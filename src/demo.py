from operator_class import ChannelType, Operator
from runtime_class import RunTime
from simulator import Simulator
from solution_class import Decision, DecisionType, MemoryBlockType

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
                    channel_type=ChannelType.AllToAll,
                    grad_size=4, pass_grad_size=2,
                    num_input_channels=2, num_output_channels=2)
    op_B = Operator(forward_memory_peek=6, forward_time_elapsed=12,
                    backward_memory_peek=6, backward_time_elapsed=12,
                    optimize_time_elapsed=12,
                    param_size=6, input_size=2, output_size=3,
                    channel_type=ChannelType.AllToAll,
                    grad_size=6, pass_grad_size=2,
                    num_input_channels=2, num_output_channels=3)
    op_C = Operator(forward_memory_peek=4, forward_time_elapsed=8,
                    backward_memory_peek=4, backward_time_elapsed=8,
                    optimize_time_elapsed=8,
                    param_size=4, input_size=2, output_size=2,
                    channel_type=ChannelType.OneToOne,
                    grad_size=4, pass_grad_size=2,
                    num_input_channels=2, num_output_channels=2)
    op_D = Operator(forward_memory_peek=5, forward_time_elapsed=10,
                    backward_memory_peek=5, backward_time_elapsed=10,
                    optimize_time_elapsed=10,
                    param_size=5, input_size=5, output_size=1,
                    channel_type=ChannelType.AllToAll,
                    grad_size=5, pass_grad_size=5,
                    num_input_channels=5, num_output_channels=1)
    op_A.init(pred_ops=[], succ_ops=[op_B, op_C],
              output_channel_accuracy=[0.05] * op_A.num_output_channels)
    op_B.init(pred_ops=[op_A], succ_ops=[op_D],
              output_channel_accuracy=[0.05] * op_B.num_output_channels)
    op_C.init(pred_ops=[op_A], succ_ops=[op_D],
              output_channel_accuracy=[0.05] * op_C.num_output_channels)
    op_D.init(pred_ops=[op_B, op_C], succ_ops=[],
              output_channel_accuracy=[0.05] * op_D.num_output_channels)

    computation_graph = [op_A, op_B, op_C, op_D]

    run_time = RunTime(memory_limit=50,
                       cross_level_bandwidth_read=2,
                       cross_level_bandwidth_write=1,
                       target_accuracy=0.8)

    # Let's compose a naive solution
    decisions = []
    current_wall_time = 0
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
            DecisionType.FORWARD
        ]
        memory_blocks = [
            MemoryBlockType.INPUT,
            MemoryBlockType.PARAM,
            MemoryBlockType.OUTPUT,
            None
        ]
        channel_ids_list = [
            list(range(op.num_input_channels)),
            list(range(op.num_input_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_input_channels))
        ]
        for i in range(4):
            decision = Decision(wall_time=current_wall_time,
                                decision_type=decision_types[i],
                                operator=op,
                                memory_block=memory_blocks[i],
                                channel_ids=channel_ids_list[i])
            decisions.append(decision)
            current_wall_time += 8

    # NOTE: prune
    decisions.append(Decision(wall_time=current_wall_time,
                              decision_type=DecisionType.PRUNE,
                              operator=computation_graph[0],
                              memory_block=None,
                              channel_ids=[0, 1]))
    for op in computation_graph[1:-1]:
        decisions.append(Decision(wall_time=current_wall_time,
                                  decision_type=DecisionType.PRUNE,
                                  operator=op,
                                  memory_block=None,
                                  channel_ids=[0]))

    # NOTE: backward phase
    #   1. allocate gradient data
    #   2. allocate pass gradient data
    #   3. backward & optimize
    for op in reversed(computation_graph[1:]):
        decision_types = [
            DecisionType.ALLOCATE,
            DecisionType.ALLOCATE,
            DecisionType.BACKWARD,
            DecisionType.OPTIMIZE
        ]
        memory_blocks = [
            MemoryBlockType.GRAD,
            MemoryBlockType.PASS_GRAD,
            None,
            None
        ]
        offset = 0 if op == computation_graph[-1] else 1
        channel_ids_list = [
            list(range(op.num_output_channels)),
            list(range(1, op.num_input_channels)),
            list(range(op.num_output_channels)),
            list(range(op.num_output_channels))
        ]
        for i in range(4):
            if op != computation_graph[-1] and \
                    memory_blocks[i] == MemoryBlockType.PASS_GRAD:
                continue
            decision = Decision(wall_time=current_wall_time,
                                decision_type=decision_types[i],
                                operator=op,
                                memory_block=memory_blocks[i],
                                channel_ids=channel_ids_list[i][offset:])
            decisions.append(decision)
            current_wall_time += 12
    # NOTE: write back parameters
    for op in reversed(computation_graph[1:]):
        offset = 0 if op == computation_graph[-1] else 1
        decision = Decision(
            wall_time=current_wall_time,
            decision_type=DecisionType.STORE,
            operator=op,
            memory_block=MemoryBlockType.PARAM,
            channel_ids=list(range(op.num_output_channels))[offset:])
        decisions.append(decision)
        current_wall_time += 6

    simulator = Simulator(run_time, computation_graph)
    simulator.staticAnalysis(decisions)
    simulator.dynamicSim()
    simulator.plotTimeline()
