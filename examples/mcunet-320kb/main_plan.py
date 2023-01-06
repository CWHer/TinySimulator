# fmt: off
import sys
sys.path.append("../..")

from typing import List

from network_config import (channels_to_prune, mcu_net, op_next_to_pruned,
                            op_to_prune)

from src.operator_class import Operator
from src.runtime_class import RunTime
from src.simulator import Simulator
from src.solution_class import Decision, DecisionType, MemoryBlockType
# fmt: on


def initSequential(block: List[Operator], block_pred, block_succ):
    assert len(block) > 0, "block is empty"
    for pred, curr, succ in \
            zip([block_pred] + block, block, (block + [block_succ])[1:]):
        curr.init(pred_ops=[pred], succ_ops=[succ],
                  output_channel_accuracy=[0.] * curr.num_output_channels)  # FIXME: NOT use accuracy yet


# FIXME: shortcut haven't been implemented
# def initShortcut(block: List[Operator], block_pred: Operator, block_succ):
#     if len(block) < 3:
#         raise RuntimeError("module block isn\'t in proper shape")
#     last = block[-1]
#     last.init(pred_ops=[block[-2], block_pred], succ_ops=[block_succ],
#               output_channel_accuracy=[0.]*last.num_output_channels)
#     block_pred.succ_ops.append(last)
#     initSequential(block[:-1], block_pred, last)


# FIXME: this function may be buggy
def computeTime(decision: Decision):
    if decision.decision_type == DecisionType.LOAD:
        if decision.memory_block == MemoryBlockType.INPUT:
            return decision.operator.input_size / runtime.cross_level_bandwidth_read
        elif decision.memory_block == MemoryBlockType.PARAM:
            return decision.operator.param_size / runtime.cross_level_bandwidth_read
    elif decision.decision_type == DecisionType.STORE:
        if decision.memory_block == MemoryBlockType.INPUT:
            return decision.operator.input_size / runtime.cross_level_bandwidth_write
        elif decision.memory_block == MemoryBlockType.PARAM:
            return decision.operator.param_size / runtime.cross_level_bandwidth_write
    elif decision.decision_type == DecisionType.ALLOCATE:
        return 0.001
    elif decision.decision_type == DecisionType.FORWARD:
        return decision.operator.forward_time_elapsed
    elif decision.decision_type == DecisionType.BACKWARD:
        return decision.operator.backward_time_elapsed * len(decision.channel_ids) / decision.operator.num_output_channels
    elif decision.decision_type == DecisionType.OPTIMIZE:
        return decision.operator.optimize_time_elapsed * len(decision.channel_ids) / decision.operator.num_output_channels
    else:
        raise RuntimeError("invalid decision type")


# HACK: this class is used to compute the wall time of decisions
class Device():
    def __init__(self) -> None:
        self.wall_time = 0

    def put(self, decision: Decision):
        self.wall_time = decision.wall_time + computeTime(decision)

    def get(self) -> float:
        return self.wall_time

    def sync(self, t):
        self.wall_time = t


if __name__ == "__main__":
    computation_graph: List[Operator] = sum(mcu_net.values(), [])
    initSequential(computation_graph, None, None)

    # NOTE: model size has the metric of Byte,
    #       time metric is millisecond
    #       so bandwidth should also has the metric of Byte
    #       Read speed is 3.1MB/s => change to Byte per msec
    #       Write speed is 680kB/s => change to Byte per msec
    # FIXME: IO speed is multiplied by 10, and NOT use target_accuracy
    runtime = RunTime(memory_limit=2 * 1024 * 1024,
                      cross_level_bandwidth_read=3.1 * 1024 * 1024 / 1000 * 10,
                      cross_level_bandwidth_write=1000 * 1024 / 1000 * 10,
                      target_accuracy=1.0)

    # Compose a solution
    decisions = []
    current_wall_time = 0

    compute_device = Device()
    input_device = Device()
    output_device = Device()

    # NOTE: forward phase
    #   1. load (allocate & copy) input data
    #   2. load parameters
    #   3. allocate output data
    #   4. forward
    for op in computation_graph:
        decision_types = [
            DecisionType.LOAD if len(
                op.pred_ops) == 0 else DecisionType.ALLOCATE,
            DecisionType.LOAD,
            DecisionType.ALLOCATE,
            DecisionType.FORWARD,
            DecisionType.STORE
        ]
        memory_blocks = [
            MemoryBlockType.INPUT,
            MemoryBlockType.PARAM,
            MemoryBlockType.OUTPUT,
            None,
            MemoryBlockType.INPUT
        ]
        channel_ids_list = [
            list(range(op.num_input_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_input_channels))
        ]
        for i in range(5):
            if decision_types[i] == DecisionType.LOAD:
                tmp_wall_time = max(input_device.get(), output_device.get())
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i])
                input_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.ALLOCATE:
                tmp_wall_time = compute_device.get()
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i])
                compute_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.FORWARD:
                tmp_wall_time = max(input_device.get(), compute_device.get())
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i])
                compute_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.STORE:
                tmp_wall_time = max(output_device.get(), compute_device.get())
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i])
                output_device.put(decision)
                decisions.append(decision)

    # NOTE: generate fence instruction
    current_wall_time = max(
        input_device.get(), output_device.get(), compute_device.get())
    input_device.sync(current_wall_time)
    output_device.sync(current_wall_time)
    compute_device.sync(current_wall_time)
    print("====Forward Fence====")
    print(current_wall_time)

    # NOTE: Pruning
    for op, channel_n in zip(op_to_prune, channels_to_prune):
        decision = Decision(wall_time=current_wall_time,
                            decision_type=DecisionType.PRUNE,
                            operator=op,
                            memory_block=None,
                            channel_ids=list(range(channel_n)))
        # print("====", computation_graph.index(op), "====")
        # print("====", decision.channel_ids, "====")
        decisions.append(decision)

    # NOTE: backward phase
    #   1. allocate gradient data
    #   2. allocate pass gradient data
    #   3. backward & optimize
    for op in reversed(computation_graph):
        decision_types = [
            DecisionType.LOAD,
            DecisionType.ALLOCATE,
            DecisionType.ALLOCATE if len(
                op.pred_ops) > 0 else None,  # source op
            DecisionType.BACKWARD,
            DecisionType.OPTIMIZE,
            DecisionType.STORE
        ]
        memory_blocks = [
            MemoryBlockType.INPUT,
            MemoryBlockType.GRAD,
            MemoryBlockType.PASS_GRAD,
            None,
            None,
            MemoryBlockType.PARAM
        ]
        channel_ids_list = [
            list(range(op.num_input_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_input_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_output_channels))
        ]
        for i in range(6):
            if decision_types[i] is None:
                continue

            channel_offset = 0 if not op in op_to_prune \
                else channels_to_prune[op_to_prune.index(op)]

            if decision_types[i] == DecisionType.LOAD:
                tmp_wall_time = max(input_device.get(),
                                    output_device.get())
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i])
                input_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.ALLOCATE:
                tmp_wall_time = compute_device.get()
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i][channel_offset:]
                                    if memory_blocks[i] == MemoryBlockType.GRAD
                                    else channel_ids_list[i])
                # NOTE: DO NOT pass gradient to pruned channels
                if op in op_next_to_pruned and \
                        memory_blocks[i] == MemoryBlockType.PASS_GRAD:
                    pred_offset = channels_to_prune[
                        op_next_to_pruned.index(op)]
                    decision.channel_ids = channel_ids_list[i][pred_offset:]
                compute_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.BACKWARD:
                tmp_wall_time = max(input_device.get(),
                                    compute_device.get())
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i][channel_offset:])
                compute_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.OPTIMIZE:
                tmp_wall_time = compute_device.get()
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i][channel_offset:])
                compute_device.put(decision)
                decisions.append(decision)
            elif decision_types[i] == DecisionType.STORE:
                tmp_wall_time = max(
                    compute_device.get(), output_device.get())
                decision = Decision(wall_time=tmp_wall_time,
                                    decision_type=decision_types[i],
                                    operator=op,
                                    memory_block=memory_blocks[i],
                                    channel_ids=channel_ids_list[i][channel_offset:])
                decisions.append(decision)
                output_device.put(decision)

    simulator = Simulator(runtime, computation_graph)
    simulator.staticAnalysis(decisions)
    simulator.dynamicSim()

    total_time, peak_memory = simulator.getStats()
    print("Total time: {:<.4f}, Peak memory: {:<.4f} MB".format(
        total_time, peak_memory / 1024 / 1024), file=sys.stderr)
    simulator.plotTimeline("q_overlap_heavy-prune.png")
