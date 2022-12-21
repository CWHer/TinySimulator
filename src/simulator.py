import dataclasses
from typing import Dict, List, Set, Tuple

import tqdm

from operator_class import MemoryType, Operator
from runtime_class import RunTime
from solution_class import (COMPUTE_DECISIONS, Decision, DecisionType,
                            MemoryBlockType, decisionRank)
from utils import printError


class Simulator:
    """
    NOTE: for both forward & backward pass
        1. as our application scenario is TinyML,
            we assume each operator (partition) can saturate the underlying accelerator,
            thus, we DO NOT consider the parallelism of operators
        2. we assume the underlying machine with two levels of hierarchical memory,
            the slow memory (e.g., NVMe) CAN NOT be directly used in computation,
            but it supports full-duplex loading & storing
        3. we assume fast memory allocation & copy is immediate without any time elapse

    NOTE: for forward pass
        1. for simplicity, we DO NOT employ re-materialization
        2. we adopt operator-level (channel-wise) parallelism to further reduce the memory footprint

    NOTE: for backward pass & optimize
        1. we assume pruning is output channel wise,
            thus, roughly speaking, if an operator is not pruned entirely,
            it can pass gradient to its predecessors
        2. we interleave backward & optimizer.step() for each operator,
            and optimizer.step() is done immediately after backward completes
    """

    def __init__(self, run_time: RunTime,
                 computation_graph: List[Operator]):
        self.run_time = run_time

        # NOTE: topological sort
        self.computation_graph: List[Operator] = []
        pred_count = {op: len(op.pred_ops) for op in computation_graph}
        queue = [op for op in computation_graph if not op.pred_ops]
        while len(queue) > 0:
            op = queue.pop(0)
            self.computation_graph.append(op)
            for succ_op in op.succ_ops:
                pred_count[succ_op] -= 1
                if pred_count[succ_op] == 0:
                    queue.append(succ_op)
        printError(len(self.computation_graph) != len(computation_graph))

    def staticAnalysis(self, solution: List[Decision]):
        # NOTE:
        #   1. sequential issues
        #       for each operator, computing is in order,
        #       forward < prune < backward < optimize
        #   2. channel conflict/completeness issues
        #       e.g., multiple forward of same channel in same operator
        #       e.g., not all channels are forwarded
        #       e.g., backward of pruned channels
        #       e.g., backward of unreachable operators
        #       e.g., FIXME: invalid MemoryType.POINTER (x)
        #   3. generate commit & purge decisions
        #       commit: to simplify simulation,
        #        for each forward/backward/optimize a commit is generated
        #       purge: refer to "when to purge data"
        #   4. sort decisions by wall_time

        operators: Set[Operator] = set()
        for t in solution:
            printError(t.decision_type in
                       [DecisionType.COMMIT, DecisionType.PURGE])
            operators.add(t.operator)
        printError(operators != set(self.computation_graph))
        del operators

        # NOTE: input data are stored in slow memory
        for op in self.computation_graph:
            if not op.pred_ops:
                op.input_locations = \
                    [MemoryType.SLOW] * op.num_input_channels

        # Pass 1
        # HACK: last forward: [t, t + 10]
        #       last backward: [t + 5, t + 15]
        #   this issue is left for runtime check
        @dataclasses.dataclass
        class Interval:
            start: float = float("inf")
            end: float = float("-inf")

            def update(self, t: float):
                self.start = min(self.start, t)
                self.end = max(self.end, t)

        forward_interval: Dict[Operator, Interval] = {}
        backward_interval: Dict[Operator, Interval] = {}
        optimize_interval: Dict[Operator, Interval] = {}
        prune_interval: Dict[Operator, Interval] = {}
        intervals = {
            DecisionType.FORWARD: forward_interval,
            DecisionType.BACKWARD: backward_interval,
            DecisionType.OPTIMIZE: optimize_interval,
            DecisionType.PRUNE: prune_interval
        }
        for op in self.computation_graph:
            for interval in intervals.values():
                interval[op] = Interval()

        for t in solution:
            if t.decision_type in intervals:
                intervals[t.decision_type][t.operator].update(t.wall_time)
        for op in self.computation_graph:
            # fmt: off
            printError(not forward_interval[op].end < prune_interval[op].start)
            printError(not prune_interval[op].end < backward_interval[op].start)
            printError(not backward_interval[op].end < optimize_interval[op].start)
            # fmt: on

        # Pass 2
        forward_channels: Dict[Operator, Set[int]] = {}
        backward_channels: Dict[Operator, Set[int]] = {}
        optimize_channels: Dict[Operator, Set[int]] = {}
        pruned_channels: Dict[Operator, Set[int]] = {}
        channels = {
            DecisionType.FORWARD: forward_channels,
            DecisionType.BACKWARD: backward_channels,
            DecisionType.OPTIMIZE: optimize_channels,
            DecisionType.PRUNE: pruned_channels
        }
        for op in self.computation_graph:
            for channel in channels.values():
                channel[op] = set()

        for t in solution:
            if t.decision_type in channels:
                channel_ids = set(t.channel_ids)
                type_channels = channels[t.decision_type]
                printError(len(channel_ids & type_channels[t.operator]) > 0)
                type_channels[t.operator].update(t.channel_ids)
        for op in self.computation_graph:
            # fmt: off
            printError(forward_channels[op] != set(range(op.num_input_channels)))
            printError(not pruned_channels[op].issubset(set(range(op.num_output_channels))))
            printError(backward_channels[op] != set(range(op.num_output_channels)) - pruned_channels[op])
            printError(optimize_channels[op] != backward_channels[op])
            # fmt: on

        # NOTE: check target accuracy
        total_accuracy_loss = 0.0
        reachable_ops = [op for op in self.computation_graph
                         if not op.succ_ops and len(backward_channels[op]) > 0]
        for op in reversed(self.computation_graph):
            if len(op.succ_ops) > 0 and \
                len(backward_channels[op]) > 0 and \
                    any(succ_op in reachable_ops
                        for succ_op in op.succ_ops):
                reachable_ops.append(op)
        for op in self.computation_graph:
            if not op in reachable_ops:
                # backward of unreachable operators
                printError(len(backward_channels[op]) > 0)
                total_accuracy_loss += sum(op.output_channel_accuracy)
            else:
                total_accuracy_loss += sum(
                    op.output_channel_accuracy[channel_id]
                    for channel_id in pruned_channels[op])
        print("Accuracy after pruning: {:.2f}%".format(
            100 * (1 - total_accuracy_loss)))
        printError(1 - total_accuracy_loss
                   < self.run_time.target_accuracy)

        # Pass 3
        # is_last_[type]_commit
        self.commit_type: Dict[Decision, bool] = {}
        for t in solution:
            if t.decision_type in COMPUTE_DECISIONS:
                self.commit_type[t] = t.wall_time == \
                    intervals[t.decision_type][t.operator].end

        # (which_operator, memory_block ,chanel_ids)
        self.purge_type: Dict[Decision, List[
            Tuple[Operator, MemoryBlockType, List[int]]]] = {}
        # HACK: usage counting (purge memory block when no more usage)
        #  however, this technic can't detect useless memory allocated by user
        output_usage_count: Dict[Operator, int] = {}
        # output_ref_count: Dict[Operator, List[int]] = {} # FIXME: for POINTER
        for op in self.computation_graph:
            output_usage_count[op] = len(op.succ_ops)
            # output_ref_count[op] = [0] * op.num_output_channels

        for t in solution:
            self.purge_type[t] = []
            if t.decision_type == DecisionType.FORWARD:
                for op in t.operator.pred_ops:
                    output_usage_count[op] -= 1
                    if output_usage_count[op] == 0:
                        self.purge_type[t].append(
                            (op, MemoryBlockType.OUTPUT,
                             list(range(op.num_output_channels))))
                if t.wall_time == forward_interval[t.operator].end:
                    self.purge_type[t].append(
                        (t.operator, MemoryBlockType.PARAM,
                         list(pruned_channels[t.operator])))
                    if t.operator not in reachable_ops:
                        self.purge_type[t].append(
                            (t.operator, MemoryBlockType.INPUT,
                             list(range(t.operator.num_input_channels))))
            elif t.decision_type == DecisionType.BACKWARD:
                # NOTE: the input is concatenated from each predecessor
                for op in t.operator.succ_ops:
                    op_offset = Operator.calcOffset(t.operator, op)
                    self.purge_type[t].append(
                        (op, MemoryBlockType.PASS_GRAD,
                         [op_offset + channel_id for channel_id in t.channel_ids]))
                if t.wall_time == backward_interval[t.operator].end:
                    self.purge_type[t].append(
                        (t.operator, MemoryBlockType.INPUT,
                         list(range(t.operator.num_input_channels))))
                    # NOTE: this is sink op
                    if not t.operator.succ_ops:
                        self.purge_type[t].append(
                            (t.operator, MemoryBlockType.OUTPUT,
                             list(range(t.operator.num_output_channels))))
            elif t.decision_type == DecisionType.OPTIMIZE:
                self.purge_type[t].append(
                    (t.operator, MemoryBlockType.GRAD, t.channel_ids))

        # Pass 4
        self.solution = sorted(solution, key=decisionRank)

    def dynamicSim(self):
        # NOTE:
        #   1. dynamically issue commit & purge decisions
        #   2. simulate all decisions

        @dataclasses.dataclass
        class MemoryUsage:
            wall_time: float
            memory_usage: float
        self.memory_usages: List[MemoryUsage] = [MemoryUsage(0, 0)]

        @dataclasses.dataclass
        class Interval:
            start: float
            end: float
        self.cpu_usages: List[Interval] = [Interval(0, 0)]
        self.input_device_usages: List[Interval] = [Interval(0, 0)]
        self.output_device_usages: List[Interval] = [Interval(0, 0)]

        done_flags = {"Forward": False, "Backward": False, "Optimize": False}

        num_decision = len(self.solution) + len(self.commit_type) \
            + sum([len(self.purge_type[t]) for t in self.solution])
        for _ in tqdm.trange(num_decision, desc="Simulating"):
            t = self.solution.pop(0)
            # print("Decision: {}".format(t.decision_type))

            # optional display
            for phase, done_flag in done_flags.items():
                if not done_flag and \
                    all([getattr(op, f"is{phase}Done")()
                         for op in self.computation_graph]):
                    done_flags[phase] = True
                    # fmt: off
                    print("===== {} Done =====".format(phase))
                    print("Current Time: {:.2f} ms".format(self.cpu_usages[-1].end))
                    print("Current Memory: {:.2f} MB".format(self.memory_usages[-1].memory_usage))
                    # fmt: on

            if t.decision_type == DecisionType.LOAD:
                printError(t.wall_time < self.input_device_usages[-1].end)
                memory_delta, time_elapsed = \
                    t.operator.load(t.memory_block.value, t.channel_ids,
                                    self.run_time.cross_level_bandwidth_read)
                self.input_device_usages.append(
                    Interval(t.wall_time, t.wall_time + time_elapsed))
                memory_usage = memory_delta + \
                    self.memory_usages[-1].memory_usage
                self.memory_usages.append(
                    MemoryUsage(t.wall_time + time_elapsed, memory_usage))
                printError(memory_usage > self.run_time.memory_limit)

            elif t.decision_type == DecisionType.STORE:
                printError(t.wall_time < self.output_device_usages[-1].end)
                memory_delta, time_elapsed = \
                    t.operator.store(t.memory_block.value, t.channel_ids,
                                     self.run_time.cross_level_bandwidth_write)
                self.output_device_usages.append(
                    Interval(t.wall_time, t.wall_time + time_elapsed))
                memory_usage = - memory_delta + \
                    self.memory_usages[-1].memory_usage
                self.memory_usages.append(
                    MemoryUsage(t.wall_time + time_elapsed, memory_usage))
                printError(memory_usage < 0)

            elif t.decision_type == DecisionType.ALLOCATE:
                memory_delta = t.operator.allocate(
                    t.memory_block.value, t.channel_ids)
                memory_usage = memory_delta + \
                    self.memory_usages[-1].memory_usage
                self.memory_usages.append(
                    MemoryUsage(t.wall_time, memory_usage))
                printError(memory_usage > self.run_time.memory_limit)

            elif t.decision_type == DecisionType.PURGE:
                memory_delta = t.operator.purge(
                    t.memory_block.value, t.channel_ids)
                memory_usage = - memory_delta + \
                    self.memory_usages[-1].memory_usage
                self.memory_usages.append(
                    MemoryUsage(t.wall_time, memory_usage))
                printError(memory_usage < 0)

            # elif t.decision_type == DecisionType.REFER:
            #     # FIXME: Not implemented
            #     t.operator.refer(t.memory_block.value, t.channel_ids)

            elif t.decision_type in COMPUTE_DECISIONS:
                printError(t.wall_time < self.cpu_usages[-1].end)
                func_name = COMPUTE_DECISIONS[t.decision_type]
                memory_delta, time_elapsed = \
                    getattr(t.operator, func_name)(t.channel_ids)
                self.cpu_usages.append(
                    Interval(t.wall_time, t.wall_time + time_elapsed))
                memory_usage = memory_delta + \
                    self.memory_usages[-1].memory_usage
                self.memory_usages.append(
                    MemoryUsage(t.wall_time + time_elapsed, memory_usage))
                printError(memory_usage > self.run_time.memory_limit)

                self.solution.append(Decision(
                    wall_time=t.wall_time + time_elapsed,
                    decision_type=DecisionType.COMMIT,
                    operator=t.operator,
                    memory_block=None,
                    channel_ids=None,
                    is_last=self.commit_type[t]
                ))
                for op, memory_block, channel_ids in self.purge_type[t]:
                    self.solution.append(Decision(
                        wall_time=t.wall_time + time_elapsed,
                        decision_type=DecisionType.PURGE,
                        operator=op,
                        memory_block=memory_block,
                        channel_ids=channel_ids,
                        is_last=False
                    ))

            elif t.decision_type == DecisionType.COMMIT:
                delta_memory = t.operator.commit(t.is_last)
                memory_usage = - delta_memory + \
                    self.memory_usages[-1].memory_usage
                self.memory_usages.append(
                    MemoryUsage(t.wall_time, memory_usage))

            elif t.decision_type == DecisionType.PRUNE:
                t.operator.prune(t.channel_ids)

            else:
                raise NotImplementedError(
                    "Unknown decision type: {}".format(t.decision_type))

            # FIXME: HACK: apparently heap is better
            self.solution.sort(key=decisionRank)

        printError(len(self.solution) != 0)
        printError(not all([getattr(op, f"isAllDone")()
                            for op in self.computation_graph]))
        if self.memory_usages[-1].memory_usage != 0:
            print("This execution plan contains useless memory allocation")
            for op in self.computation_graph:
                print("Operator:")
                op.printMemory()
                print()
        self.total_time = max(
            [self.cpu_usages[-1].end,
             self.input_device_usages[-1].end,
             self.output_device_usages[-1].end]
        )
        print("===== All Done =====")
        print("Current Time: {:.2f} ms".format(self.total_time))
        print("Current Memory: {:.2f} MB".format(
            self.memory_usages[-1].memory_usage))

    def getStats(self) -> Tuple[float, float]:
        memory_peek = max([t.memory_usage for t in self.memory_usages])
        return self.total_time, memory_peek

    def plotTimeline(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot([t.wall_time for t in self.memory_usages],
                 [t.memory_usage for t in self.memory_usages],
                 color="purple", alpha=0.5)
        plt.plot([t.wall_time for t in self.memory_usages],
                 [self.run_time.memory_limit] * len(self.memory_usages),
                 color="red", alpha=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Memory (MB)")

        # use Gantt chart for better illustration
        plt.subplot(1, 2, 2)
        plt.hlines([0, 1, 2], [0] * 3, [self.total_time] * 3,
                   linestyles="dashed", color="black", alpha=0.5)
        for t in self.cpu_usages[1:]:
            plt.fill_between([t.start, t.end], 0, 1, alpha=0.2, color="red")
        for t in self.input_device_usages[1:]:
            plt.fill_between([t.start, t.end], 1, 2, alpha=0.2, color="green")
        for t in self.output_device_usages[1:]:
            plt.fill_between([t.start, t.end], 2, 3, alpha=0.2, color="blue")
        plt.yticks([0.5, 1.5, 2.5], ["CPU", "Input", "Output"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Device")

        plt.savefig("timeline.png")
