from inspect import currentframe, getframeinfo
from typing import List

import tqdm

from component_class import MemoryType, Operator, RunTime
from solution_class import Decision, DecisionType, MemoryBlockType
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
        self.computation_graph = computation_graph

    def staticProcess(self, solution: List[Decision]):
        # NOTE:
        #   1. find proper sink operators for backward pass
        #   2. sequential issues
        #       forward decisions happen before backward decisions
        #       optimize decisions happen after backward decisions
        #       prune decisions happens before backward decisions
        #   3. channel conflict/completeness issues
        #       e.g., multiple forward of same channel in same operator
        #       e.g., not all channels are forwarded
        #   4. generate commit & purge decisions
        #       commit: to simplify simulation,
        #        for each forward/backward/optimize a commit is generated
        #       purge: refer to "when to purge data"
        #   5. sort decisions by wall_time
        # TODO: prune is not implemented yet

        @dataclasses.dataclass
        class Interval:
            start: float = 1e100
            end: float = -1e100

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

        operators: Set[Operator] = set()
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

        for t in solution:
            printError(
                t.decision_type == DecisionType.COMMIT or
                t.decision_type == DecisionType.PURGE)
            # TODO
            printError(t.decision_type == DecisionType.PRUNE)
            operators.add(t.operator)
        printError(operators != set(self.computation_graph))

        for op in operators:
            for channel in channels.values():
                channel[op] = set()
            for interval in intervals.values():
                interval[op] = Interval()

        # Thing 3
        for t in solution:
            if t.decision_type in channels:
                channel_ids = set(t.channel_ids)
                type_channels = channels[t.decision_type]
                printError(len(channel_ids & type_channels[t.operator]) > 0)
                type_channels[t.operator].update(t.channel_ids)
        for op in operators:
            # fmt: off
            printError(forward_channels[op] != set(range(op.num_input_channels)))
            printError(not pruned_channels[op].issubset(set(range(op.num_output_channels))))
            printError(backward_channels[op] != set(range(op.num_output_channels)) - pruned_channels[op])
            printError(optimize_channels[op] != backward_channels[op])
            # fmt: on

        # Thing 2
        # HACK: last forward: [t, t + 10]
        #       last backward: [t + 5, t + 15]
        #   this issue is left for runtime check
        for t in solution:
            if t.decision_type in intervals:
                intervals[t.decision_type][t.operator].update(t.wall_time)
        for op in operators:
            # fmt: off
            printError(forward_interval[op].end > backward_interval[op].start)
            printError(backward_interval[op].end > optimize_interval[op].start)
            printError(prune_interval[op].end > backward_interval[op].start)
            # fmt: on

        # Thing 4
        # TODO: prune
        commit_types = {
            DecisionType.FORWARD,
            DecisionType.BACKWARD,
            DecisionType.OPTIMIZE
        }
        # is_last_[type]_commit
        self.commit_type: Dict[Decision, bool] = {}
        # (which_operator, memory_block ,chanel_ids)
        self.purge_type: Dict[Decision, List[
            Tuple[Operator, MemoryBlockType, List[int]]]] = {}
        for t in solution:
            if t.decision_type in commit_types:
                self.commit_type[t] = t.wall_time == \
                    intervals[t.decision_type][t.operator].end
        # HACK: reference counting (per memory type per channel)
        output_ref_count: Dict[Operator, List[int]] = {}
        grad_ref_count: Dict[Operator, List[int]] = {}
        pass_grad_ref_count: Dict[Operator, List[int]] = {}
        for op in operators:
            # fmt: off
            output_ref_count[op] = [len(op.succ_ops)] * op.num_output_channels
            grad_ref_count[op] = [1] * op.num_output_channels
            pass_grad_ref_count[op] = [len(op.pred_ops)] * op.num_input_channels
            # fmt: on
        # TODO: FIXME: sink and source ops
        for t in solution:
            self.purge_type[t] = []
            if t.decision_type == DecisionType.FORWARD:
                for op in t.operator.pred_ops:
                    for channel_id in range(op.num_output_channels):
                        output_ref_count[op][channel_id] -= 1
                    if output_ref_count[op][0] == 0:
                        self.purge_type[t].append(
                            (op, MemoryBlockType.OUTPUT,
                             list(range(op.num_output_channels))))
            if t.decision_type == DecisionType.BACKWARD:
                for op in t.operator.succ_ops:
                    for channel_id in range(op.num_input_channels):
                        pass_grad_ref_count[op][channel_id] -= 1
                    if pass_grad_ref_count[op][0] == 0:
                        self.purge_type[t].append(
                            (op, MemoryBlockType.PASS_GRAD,
                             list(range(op.num_input_channels))))
                if t.wall_time == backward_interval[t.operator].end:
                    self.purge_type[t].append(
                        (t.operator, MemoryBlockType.INPUT,
                         list(range(t.operator.num_input_channels))))
            if t.decision_type == DecisionType.OPTIMIZE:
                for channel_id in t.channel_ids:
                    grad_ref_count[t.operator][channel_id] -= 1
                self.purge_type[t].append(
                    (t.operator, MemoryBlockType.GRAD, t.channel_ids))

        # Thing 1
        self.source_op: List[Operator] = []
        self.sink_op: List[Operator] = []
        for op in self.computation_graph:
            if len(op.pred_ops) == 0:
                self.source_op.append(op)
                # NOTE: input data are stored in slow memory
                op.input_locations = \
                    [MemoryType.SLOW] * op.num_input_channels

            def isSink(op: Operator) -> bool:
                for succ_op in op.succ_ops:
                    if len(backward_channels[succ_op]) > 0:
                        return False
                return True
            # NOTE: if all successors are pruned, then it is a sink operator
            if isSink(op):
                self.sink_op.append(op)

        # Thing 5
        self.solution = sorted(solution, key=lambda x: x.wall_time)

    def dynamicSim(self):
        # NOTE:
        #   1. generate commit & purge decisions
        #   2. simulate all decisions
        # TODO

                if decision.decision_phase == DecisionPhase.FORWARD:
                    if decision.decision_type == DecisionType.COMPUTE:
                        assert decision.wall_time >= last_computing, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        memory_peek, time_elapsed = decision.operator.forward(
                            decision.channel_ids)
                        last_computing = decision.wall_time + time_elapsed
                        decision.operator.forward_count += len(
                            decision.channel_ids)

                        # TODO: this is not correct
                        if decision.operator.isForwardDone():
                            memory_type = MemoryType.SLOW if any(
                                [param_loc == MemoryType.SLOW
                                 for param_loc in decision.operator.param_locations]) \
                                else MemoryType.FAST
                            decision.operator.param_locations = \
                                [memory_type for _ in range(
                                    decision.operator.num_output_channels)]
                        assert current_memory + memory_peek <= self.run_time.memory, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        # NOTE: no memory resident increase as output tensor is already in memory

                    if decision.decision_type == DecisionType.LOAD:
                        assert decision.wall_time >= last_loading, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        memory_delta, time_elapsed = \
                            decision.operator.load(decision.memory_block.value,
                                                   decision.channel_ids,
                                                   self.run_time.cross_level_bandwidth_read)
                        last_loading = decision.wall_time + time_elapsed
                        current_memory += memory_delta
                        assert current_memory <= self.run_time.memory, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)

                    if decision.decision_type == DecisionType.STORE:
                        assert decision.wall_time >= last_storing, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        memory_delta, time_elapsed = \
                            decision.operator.store(decision.memory_block.value,
                                                    decision.channel_ids,
                                                    self.run_time.cross_level_bandwidth_write)
                        last_storing = decision.wall_time + time_elapsed
                        current_memory -= memory_delta
                        assert current_memory >= 0, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)

                if decision.decision_phase == DecisionPhase.BACKWARD:
                    if decision.decision_type == DecisionType.COMPUTE:
                        assert decision.wall_time >= last_computing, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        memory_peek, time_elapsed = decision.operator.backward(
                            decision.channel_ids)
                        last_computing = decision.wall_time + time_elapsed
                        decision.operator.backward_count += len(
                            decision.channel_ids)
                        assert current_memory + memory_peek <= self.run_time.memory, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        # NOTE: no memory resident increase as input tensor is already in memory

                    if decision.decision_type == DecisionType.LOAD:
                        assert decision.wall_time >= last_loading, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        memory_delta, time_elapsed = \
                            decision.operator.load(decision.memory_block.value,
                                                   decision.channel_ids,
                                                   self.run_time.cross_level_bandwidth_read)
                        last_loading = decision.wall_time + time_elapsed
                        current_memory += memory_delta
                        assert current_memory <= self.run_time.memory, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)

                    if decision.decision_type == DecisionType.STORE:
                        assert decision.wall_time >= last_storing, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)
                        memory_delta, time_elapsed = \
                            decision.operator.store(decision.memory_block.value,
                                                    decision.channel_ids,
                                                    self.run_time.cross_level_bandwidth_write)
                        last_storing = decision.wall_time + time_elapsed
                        current_memory -= memory_delta
                        assert current_memory >= 0, \
                            "file: {}, line: {}".format(
                                __file__, getframeinfo(currentframe()).lineno)

            assert self.checkBackwardDone(), \
                "file: {}, line: {}".format(
                    __file__, getframeinfo(currentframe()).lineno)
            self.total_time = last_computing
            self.is_valid = True

        except AssertionError as e:
            print(e)
            print("===== Invalid Solution =====")
            self.is_valid = False

    def checkForwardDone(self):
        for op in self.sink_op:
            if not op.isForwardDone():
                return False
        return True

    def checkBackwardDone(self):
        for op in self.source_op:
            if not op.isBackwardDone():
                return False
        return True

    def printStats(self):
        # TODO
        pass

    def isValid(self):
        return self.is_valid

    def getTime(self):
        return self.total_time
