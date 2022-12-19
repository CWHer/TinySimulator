from simulator import Simulator

from component_class import Operator, RunTime
from solution_class import Decision, DecisionPhase, DecisionType, MemoryBlockType


if __name__ == "__main__":
    # Let's build a simple network
    # Topology: A --> B --> D & A --> C --> D
    # A: 2 input channels, 2 output channels
    # B: 2 input channels, 3 output channels
    # C: 2 input channels, 2 output channels
    # D: 5 input channels, 1 output channel

    op_A = Operator(forward_memory_peek=4, forward_time_elapsed=8,
                    backward_memory_peek=4, backward_time_elapsed=8,
                    optimize_time_elapsed=8,
                    param_size=4, input_size=2, output_size=2,
                    grad_size=4, pass_grad_size=2,
                    num_input_channels=2, num_output_channels=2)
    op_B = Operator(forward_memory_peek=6, forward_time_elapsed=12,
                    backward_memory_peek=6, backward_time_elapsed=12,
                    optimize_time_elapsed=12,
                    param_size=6, input_size=2, output_size=3,
                    grad_size=6, pass_grad_size=2,
                    num_input_channels=2, num_output_channels=3)
    op_C = Operator(forward_memory_peek=4, forward_time_elapsed=8,
                    backward_memory_peek=4, backward_time_elapsed=8,
                    optimize_time_elapsed=8,
                    param_size=4, input_size=2, output_size=2,
                    grad_size=4, pass_grad_size=2,
                    num_input_channels=2, num_output_channels=2)
    op_D = Operator(forward_memory_peek=5, forward_time_elapsed=10,
                    backward_memory_peek=5, backward_time_elapsed=10,
                    optimize_time_elapsed=10,
                    param_size=5, input_size=5, output_size=1,
                    grad_size=5, pass_grad_size=5,
                    num_input_channels=5, num_output_channels=1)
    op_A.link(pred_ops=[], succ_ops=[op_B, op_C])
    op_B.link(pred_ops=[op_A], succ_ops=[op_D])
    op_C.link(pred_ops=[op_A], succ_ops=[op_D])
    op_D.link(pred_ops=[op_B, op_C], succ_ops=[])

    computation_graph = [op_A, op_B, op_C, op_D]
    for op in computation_graph:
        op.initMemory()

    run_time = RunTime(memory=10000,
                       cross_level_bandwidth_read=2,
                       cross_level_bandwidth_write=1)

    # Let's compose a naive solution
    decisions = []
    current_wall_time = 0
    for op in computation_graph:
        decision_types = [DecisionType.LOAD, DecisionType.LOAD,
                          DecisionType.LOAD, DecisionType.COMPUTE]
        memory_blocks = [MemoryBlockType.PARAM, MemoryBlockType.INPUT,
                         MemoryBlockType.OUTPUT, None]
        channel_ids_list = [list(range(op.num_input_channels)),
                            list(range(op.num_input_channels)),
                            list(range(op.num_output_channels)),
                            list(range(op.num_input_channels))]
        for i in range(4):
            decision = Decision(wall_time=current_wall_time,
                                decision_phase=DecisionPhase.FORWARD,
                                decision_type=decision_types[i],
                                memory_block=memory_blocks[i],
                                operator=op,
                                channel_ids=channel_ids_list[i])
            decisions.append(decision)
            current_wall_time += 50
    for op in reversed(computation_graph):
        decision_types = [DecisionType.LOAD, DecisionType.LOAD,
                          DecisionType.COMPUTE]
        memory_blocks = [MemoryBlockType.GRAD, MemoryBlockType.PASS_GRAD, None]
        channel_ids_list = [list(range(op.num_output_channels)),
                            list(range(op.num_input_channels)),
                            list(range(op.num_output_channels))]
        for i in range(3):
            decision = Decision(wall_time=current_wall_time,
                                decision_phase=DecisionPhase.BACKWARD,
                                decision_type=decision_types[i],
                                memory_block=memory_blocks[i],
                                operator=op,
                                channel_ids=channel_ids_list[i])
            decisions.append(decision)
            current_wall_time += 50

    simulator = Simulator(run_time, computation_graph)
    simulator.run(decisions)
    if simulator.isValid():
        print("Total wall time: {:.2f} s.".format(simulator.getTime()))
