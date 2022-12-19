from typing import List

from component_class import Operator, RunTime
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
                 computation_graph: List[Operator],
                 solution: List[Decision]):
        self.run_time = run_time
        self.computation_graph = computation_graph
        self.solution = solution

        self.source_op: List[Operator] = []
        self.sink_op: List[Operator] = []
        for op in computation_graph:
            if len(op.pred_ops) == 0:
                self.source_op.append(op)
            if len(op.succ_ops) == 0:
                self.sink_op.append(op)

    def run(self):
        last_computing = 0.0
        last_loading = 0.0
        last_storing = 0.0
        current_memory = 0.0
        backward_done = False

        # TODO: drop data
        try:
            for decision in self.solution:
                if self.checkForwardDone():
                    print("===== Forward Pass Done =====")
                    print("Current Time: {:.2f} s".format(last_computing))
                    print("Current Memory: {:.2f} MB".format(current_memory))

                if decision.decision_phase == DecisionPhase.FORWARD:
                    if decision.decision_type == DecisionType.COMPUTE:
                        assert decision.wall_time >= last_computing
                        memory_peek, time_elapsed = decision.operator.forward(
                            decision.channel_ids)
                        last_computing = decision.wall_time + time_elapsed
                        decision.operator.forward_count += len(
                            decision.channel_ids)
                        # TODO: this is not correct
                        assert current_memory + memory_peek <= self.run_time.memory
                        # NOTE: no memory resident increase as output tensor is already in memory

                    if decision.decision_type == DecisionType.LOAD:
                        assert decision.wall_time >= last_loading
                        memory_delta, time_elapsed = \
                            decision.operator.load(decision.memory_block.name,
                                                   decision.channel_ids,
                                                   self.run_time.cross_level_bandwidth_read)
                        last_loading = decision.wall_time + time_elapsed
                        current_memory += memory_delta
                        assert current_memory <= self.run_time.memory

                    if decision.decision_type == DecisionType.STORE:
                        assert decision.wall_time >= last_storing
                        memory_delta, time_elapsed = \
                            decision.operator.store(decision.memory_block.name,
                                                    decision.channel_ids,
                                                    self.run_time.cross_level_bandwidth_write)
                        last_storing = decision.wall_time + time_elapsed
                        current_memory -= memory_delta
                        assert current_memory >= 0

                if decision.decision_phase == DecisionPhase.BACKWARD:
                    if decision.decision_type == DecisionType.COMPUTE:
                        assert decision.wall_time >= last_computing
                        memory_peek, time_elapsed = decision.operator.backward(
                            decision.channel_ids)
                        last_computing = decision.wall_time + time_elapsed
                        decision.operator.backward_count += len(
                            decision.channel_ids)
                        assert current_memory + memory_peek <= self.run_time.memory
                        # NOTE: no memory resident increase as input tensor is already in memory

                    if decision.decision_type == DecisionType.LOAD:
                        assert decision.wall_time >= last_loading
                        memory_delta, time_elapsed = \
                            decision.operator.load(decision.memory_block.name,
                                                   decision.channel_ids,
                                                   self.run_time.cross_level_bandwidth_read)
                        last_loading = decision.wall_time + time_elapsed
                        current_memory += memory_delta
                        assert current_memory <= self.run_time.memory

                    if decision.decision_type == DecisionType.STORE:
                        assert decision.wall_time >= last_storing
                        memory_delta, time_elapsed = \
                            decision.operator.store(decision.memory_block.name,
                                                    decision.channel_ids,
                                                    self.run_time.cross_level_bandwidth_write)
                        last_storing = decision.wall_time + time_elapsed
                        current_memory -= memory_delta
                        assert current_memory >= 0

            assert self.checkBackwardDone()
            self.total_time = last_computing
            self.is_valid = backward_done

        except AssertionError as e:
            print(e)
            self.is_valid = False

    def checkForwardDone(self):
        for op in self.sink_op:
            if op.forward_count \
                    != op.num_output_channels:
                return False
        return True

    def checkBackwardDone(self):
        for op in self.source_op:
            if op.backward_count \
                    != op.num_input_channels:
                return False
        return True

    def printStats(self):
        # TODO
        pass

    def isValid(self):
        return self.is_valid

    def getCost(self):
        return self.total_time
