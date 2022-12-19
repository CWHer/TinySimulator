from inspect import currentframe, getframeinfo
from typing import List

import tqdm

from component_class import MemoryType, Operator, RunTime
from solution_class import Decision, DecisionPhase, DecisionType


class Simulator:
    """
    NOTE: for both forward & backward pass
        1. as our application scenario is TinyML,
            we assume each operator (partition) can saturate the underlying accelerator,
            thus, we DO NOT consider the parallelism of operators
        2. we assume the underlying machine with two levels of hierarchical memory,
            the slow memory (e.g., NVMe) CAN NOT be directly used in computation,
            but it supports full-duplex loading & storing

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

        self.source_op: List[Operator] = []
        self.sink_op: List[Operator] = []
        for op in computation_graph:
            if len(op.pred_ops) == 0:
                self.source_op.append(op)
            if len(op.succ_ops) == 0:
                self.sink_op.append(op)

    def run(self, solution: List[Decision]):
        last_computing = 0.0
        last_loading = 0.0
        last_storing = 0.0
        current_memory = 0.0
        forward_done = False

        # TODO: drop data
        try:
            for decision in tqdm.tqdm(solution):
                if not forward_done \
                        and self.checkForwardDone():
                    forward_done = True
                    print("===== Forward Pass Done =====")
                    print("Current Time: {:.2f} s".format(last_computing))
                    print("Current Memory: {:.2f} MB".format(current_memory))

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
