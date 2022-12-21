from __future__ import annotations

import dataclasses
from typing import List, Optional, Set, Tuple

from runtime_class import MemoryType
from utils import printError


class MemoryRecord:
    def __init__(self, memory_block: List):
        # NOTE: this is a reference
        self.memory_block = memory_block
        self.channel_ids = []
        self.memory_type: List[MemoryType] = []

    def log(self, channel_id):
        self.channel_ids.append(channel_id)
        self.memory_type.append(
            self.memory_block[channel_id])

    def recover(self):
        for channel_id, memory_type in \
                zip(self.channel_ids, self.memory_type):
            self.memory_block[channel_id] = memory_type

    def lock(self):
        for channel_id in self.channel_ids:
            self.memory_block[channel_id] = MemoryType.RUNNING


class CommitInfo:
    def __init__(self, action_name: str,
                 memory_records: List[MemoryRecord],
                 delta_memory: float, delta_count: int):
        # action_name: "forward", "backward", "optimize"
        # memory_records: use to recover memory stats
        printError(action_name not in
                   ["forward", "backward", "optimize"])
        self.action_name = action_name
        self.memory_records = memory_records
        self.delta_memory = delta_memory
        self.delta_count = delta_count


@dataclasses.dataclass
class Operator:
    # computation related
    forward_memory_peek: float
    forward_time_elapsed: float
    backward_memory_peek: float
    backward_time_elapsed: float
    optimize_time_elapsed: float

    param_size: float
    input_size: float
    output_size: float
    grad_size: float
    pass_grad_size: float

    num_input_channels: int
    num_output_channels: int

    # network topology
    # NOTE:
    #   1. when an operator has multiple predecessors,
    #       the input is concatenated from each predecessor
    #   2. when an operator has multiple successors,
    #       the output is copied to each successor
    pred_ops: List[Operator] = None
    succ_ops: List[Operator] = None

    # TODO: pruning related
    # output_channel_accuracy: List[float]  # num output channels
    pruned_output_channels: Set[int] = None

    # memory related
    # NOTE:
    #   1. forward: X' = f(w, X)
    #       (pred_)output(ch) + param(in_ch) -> output(all)
    #       (pred_)output(ch) -> input(ch)
    #   2. backward: dW = dX' * df/dw, dX = dX' * df/dX
    #       (succ_)pass_grad(ch) + param(out_ch) -> pass_grad(all)
    #       (succ_)pass_grad(ch) + input(all) -> grad(ch)
    #   3. optimize: W' = W - lr * dW
    #       param(out_ch) + grad(ch)

    # (NOTE: num input channels for forward, num output channels for backward)
    param_locations: List[MemoryType] = None
    input_locations: List[MemoryType] = None        # num input channels
    output_locations: List[MemoryType] = None       # num output channels
    grad_locations: List[MemoryType] = None         # num output channels
    pass_grad_locations: List[MemoryType] = None    # num input channels
    # NOTE: when to purge data
    #   1. when output data is used by all successors (output)
    #   2. when passing gradient data to all predecessors (pass_grad)
    #   3. when input data is used by gradient computation (input)
    #   4. when optimize is done (grad)

    # forward_count == num_input_channels means forward is done
    forward_count: int = 0
    # backward_count == res_output_channels (after pruning) means backward is done
    backward_count: int = 0
    # optimize_count == res_output_channels (after pruning) means optimize is done
    optimize_count: int = 0

    commit_info: CommitInfo = None

    def __hash__(self) -> int:
        return id(self)

    def init(self, pred_ops, succ_ops):
        self.pred_ops = pred_ops
        self.succ_ops = succ_ops
        self.pruned_output_channels = set()
        self.param_locations = [MemoryType.SLOW] * self.num_input_channels
        self.input_locations = [MemoryType.NONE] * self.num_input_channels
        self.output_locations = [MemoryType.NONE] * self.num_output_channels
        self.grad_locations = [MemoryType.NONE] * self.num_output_channels
        self.pass_grad_locations = [MemoryType.NONE] * self.num_input_channels

    def checkNumChannels(self, is_forward: bool):
        # assert len(self.output_channel_accuracy) == self.num_output_channels
        printError(len(self.param_locations) != (
            self.num_input_channels if is_forward else self.num_output_channels))
        printError(len(self.input_locations) != self.num_input_channels)
        printError(len(self.output_locations) != self.num_output_channels)
        printError(len(self.grad_locations) != self.num_output_channels)
        printError(len(self.pass_grad_locations) != self.num_input_channels)

    def isAllDone(self) -> bool:
        # NOTE: all parameters should be write to slow memory (if not pruned)
        if not self.isFullPruned() and \
            any(param_loc != MemoryType.SLOW
                for param_loc in self.param_locations):
            return False
        return self.isForwardDone() and \
            self.isBackwardDone() and self.isOptimizeDone()

    def isForwardDone(self) -> bool:
        return self.forward_count == self.num_input_channels

    def isBackwardDone(self) -> bool:
        return self.backward_count == \
            self.num_output_channels - len(self.pruned_output_channels)

    def isOptimizeDone(self) -> bool:
        return self.optimize_count == \
            self.num_output_channels - len(self.pruned_output_channels)

    def isFullPruned(self) -> bool:
        return len(self.pruned_output_channels) == self.num_output_channels

    @staticmethod
    def __checkMemStat(memory_block, memory_type: Set,
                       channel_ids=None) -> Optional[MemoryRecord]:
        if channel_ids is None:
            channel_ids = range(len(memory_block))
        memory_record = MemoryRecord(memory_block)
        for channel_id in channel_ids:
            if memory_block[channel_id] not in memory_type:
                return None
            memory_record.log(channel_id)
        return memory_record

    @staticmethod
    def __findPredChannels(pred_ops: List[Operator],
                           channel_ids: List[int]) -> List[Tuple[Operator, int]]:
        # NOTE: when an operator has multiple predecessors,
        #   the input is concatenated from each predecessor
        channel_offset, pred_channels = 0, []
        for pred_op in pred_ops:
            num_output_channels = pred_op.num_output_channels
            # FIXME: binary search is more efficient
            for channel_id in channel_ids:
                pred_channel_id = channel_id - channel_offset
                if 0 <= pred_channel_id < num_output_channels:
                    pred_channels.append((pred_op, pred_channel_id))
            channel_offset += num_output_channels
        return pred_channels

    def canForward(self, channel_ids) -> Optional[List[MemoryRecord]]:
        # NOTE: when copy (pred_)output(ch) -> input(ch)
        #   indeed we can let input(ch) be a pointer to (pred_)output(ch),
        #   which will not increase the memory usage.
        # fmt: off
        args_list = [
            (self.output_locations, {MemoryType.FAST}),
            (self.param_locations, {MemoryType.FAST}, channel_ids),
            (self.input_locations, {MemoryType.FAST}, channel_ids)
            # (self.input_locations, {MemoryType.FAST, MemoryType.POINTER}, channel_ids)
        ]
        memory_records = [self.__checkMemStat(*args) for args in args_list]
        if None in memory_records: return None
        # fmt: on

        pred_channels = self.__findPredChannels(self.pred_ops, channel_ids)
        for pred_op, channel_id in pred_channels:
            if not pred_op.isForwardDone() or \
                    pred_op.output_locations[
                        channel_id] != MemoryType.FAST:
                return None
            memory_record = MemoryRecord(pred_op.output_locations)
            memory_record.log(channel_id)
            memory_records.append(memory_record)

        return memory_records

    def canBackward(self, channel_ids) -> Optional[List[MemoryRecord]]:
        args_list = [
            (self.input_locations, {MemoryType.FAST}),
            (self.grad_locations, {MemoryType.FAST}, channel_ids),
            (self.param_locations, {MemoryType.FAST}, channel_ids)
        ]
        # HACK: not need to pass grad if op is source (after pruning)
        if not all(pred_op.isFullPruned()
                   for pred_op in self.pred_ops):
            args_list.append((self.pass_grad_locations, {MemoryType.FAST}))
        # fmt: off
        memory_records = [self.__checkMemStat(*args) for args in args_list]
        if None in memory_records: return None
        # fmt: on

        # NOTE: when an operator has multiple successors,
        #   the output is copied to each successor
        for succ_op in self.succ_ops:
            for channel_id in channel_ids:
                if not succ_op.isBackwardDone() \
                    or succ_op.pass_grad_locations[
                        channel_id] != MemoryType.FAST:
                    return None
                memory_record = MemoryRecord(succ_op.pass_grad_locations)
                memory_record.log(channel_id)
                memory_records.append(memory_record)

        return memory_records

    def forward(self, channel_ids) -> Tuple[float, float]:
        memory_records = self.canForward(channel_ids)
        printError(memory_records is None)
        r = self.num_input_channels / len(channel_ids)
        memory_delta = r * self.forward_memory_peek
        time_elapsed = r * self.forward_time_elapsed
        for memory_record in memory_records:
            memory_record.lock()
        self.commit_info = CommitInfo(action_name="forward",
                                      memory_records=memory_records,
                                      delta_memory=memory_delta,
                                      delta_count=len(channel_ids))
        return memory_delta, time_elapsed

    def backward(self, channel_ids) -> Tuple[float, float]:
        printError(not self.isForwardDone())
        memory_records = self.canBackward(channel_ids)
        printError(memory_records is None)
        r = self.num_output_channels / len(channel_ids)
        memory_delta = r * self.backward_memory_peek
        time_elapsed = r * self.backward_time_elapsed
        for memory_record in memory_records:
            memory_record.lock()
        self.commit_info = CommitInfo(action_name="backward",
                                      memory_records=memory_records,
                                      delta_memory=memory_delta,
                                      delta_count=len(channel_ids))
        return memory_delta, time_elapsed

    def __memoryOp(self, memory_type: str, channel_ids: List[int],
                   prev_type: Set, new_type: MemoryType, bandwidth=0.1) -> Tuple[float, float]:
        printError(memory_type not in
                   ["param", "input", "output", "grad", "pass_grad"])
        memory_block = getattr(self, memory_type + "_locations")
        for channel_id in channel_ids:
            printError(memory_block[channel_id] not in prev_type)
            memory_block[channel_id] = new_type
        r = len(channel_ids) / len(memory_block)
        memory_delta = r * getattr(self, memory_type + '_size')
        time_elapsed = memory_delta / bandwidth
        return memory_delta, time_elapsed

    def load(self, memory_type: str, channel_ids, bandwidth) -> Tuple[float, float]:
        return self.__memoryOp(memory_type, channel_ids,
                               set([MemoryType.SLOW]), MemoryType.FAST, bandwidth)

    def store(self, memory_type: str, channel_ids, bandwidth) -> Tuple[float, float]:
        return self.__memoryOp(memory_type, channel_ids,
                               set([MemoryType.FAST]),
                               # FIXME: remove POINTER do not change memory usage
                               # set([MemoryType.FAST, MemoryType.POINTER]),
                               MemoryType.SLOW, bandwidth)

    def allocate(self, memory_type: str, channel_ids) -> float:
        return self.__memoryOp(memory_type, channel_ids,
                               set([MemoryType.NONE]), MemoryType.FAST)[0]

    def purge(self,  memory_type: str, channel_ids) -> float:
        # NOTE: HACK: this is generated by the simulator
        return self.__memoryOp(memory_type, channel_ids,
                               set([MemoryType.FAST]),
                               # FIXME: remove POINTER do not change memory usage
                               # set([MemoryType.FAST, MemoryType.POINTER]),
                               MemoryType.NONE)[0]

    # def refer(self, memory_type: str, channel_ids):
    #     # HACK: the only supported type is to point input(ch) at (pred_)output(ch)
    #     printError(memory_type != "input")
    #     pred_channels = self.__findPredChannels(self.pred_ops, channel_ids)
    #     for pred_op, channel_id in pred_channels:
    #         printError(pred_op.output_locations[channel_id] != MemoryType.FAST)
    #     for channel_id in channel_ids:
    #         self.input_locations[channel_id] = MemoryType.POINTER

    def canOptimize(self, channel_ids) -> Optional[List[MemoryRecord]]:
        args_list = [
            (self.param_locations, {MemoryType.FAST}, channel_ids),
            (self.grad_locations, {MemoryType.FAST}, channel_ids)
        ]
        memory_records = [self.__checkMemStat(*args) for args in args_list]
        return None if None in memory_records else memory_records

    def optimize(self, channel_ids) -> Tuple[float, float]:
        # NOTE: W' = W - lr * dW, DO NOT require extra memory
        printError(not self.isForwardDone())
        printError(not self.isBackwardDone())
        memory_records = self.canOptimize(channel_ids)
        printError(memory_records is None)
        r = self.num_output_channels / len(channel_ids)
        time_elapsed = r * self.optimize_time_elapsed
        for memory_record in memory_records:
            memory_record.lock()
        self.commit_info = CommitInfo(action_name="optimize",
                                      memory_records=memory_records,
                                      delta_memory=0,
                                      delta_count=len(channel_ids))
        return 0, time_elapsed

    def prune(self, channel_ids) -> None:
        self.pruned_output_channels |= set(channel_ids)

    def commit(self, is_last=False) -> float:
        # NOTE: HACK: this is generated by the simulator

        # NOTE: recover memory stats
        for memory_record in \
                self.commit_info.memory_records:
            memory_record.recover()

        action_name = self.commit_info.action_name
        if action_name == "forward":
            self.forward_count += \
                self.commit_info.delta_count
            if is_last:
                # HACK: repartition param_location to
                #   num_output_channels when last forward is done
                memory_type = MemoryType.FAST \
                    if all(param_loc == MemoryType.FAST
                           for param_loc in self.param_locations) \
                    else MemoryType.SLOW
                self.param_locations = [
                    memory_type for _ in range(self.num_output_channels)]
        elif action_name == "backward":
            self.backward_count += \
                self.commit_info.delta_count
        elif action_name == "optimize":
            self.optimize_count += \
                self.commit_info.delta_count
        return self.commit_info.delta_memory
