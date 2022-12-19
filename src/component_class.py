from __future__ import annotations

import dataclasses
from enum import Enum
from typing import List, Tuple


class MemoryType(Enum):
    """
    NOTE: we assume the underlying machine with two levels of hierarchical memory,
            the slow memory (e.g., NVMe) CAN NOT be directly used in computation,
            but it supports full-duplex loading & storing
    """
    FAST = 1
    SLOW = 2


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

    # network topology
    pred_ops: List[Operator]
    succ_ops: List[Operator]

    num_input_channels: int
    num_output_channels: int

    # TODO: pruning related
    # output_channel_accuracy: List[float]  # num output channels

    # simulator related
    # forward: X' = f(w, X)
    # backward: dW = dX' * df/dw, dX = dX' * df/dX
    # (num input channels for forward, num output channels for backward)
    param_locations: List[MemoryType]
    input_locations: List[MemoryType]  # num input channels
    output_locations: List[MemoryType]  # num output channels
    grad_locations: List[MemoryType]  # num output channels
    pass_grad_locations: List[MemoryType]  # num input channels
    # TODO: NOTE: when to drop data
    #   1. when output data is used by all successors (output)
    #   2. when passing gradient data to all predecessors (pass_grad)
    #   3. when input data is used by gradient computation (input)
    #   4. when optimize is done (grad)

    # forward_count == num_input_channels means forward is done
    forward_count: int = 0
    # backward_count == res_output_channels (pruning) means backward is done
    backward_count: int = 0

    def checkNumChannels(self, is_forward: bool):
        # assert len(self.output_channel_accuracy) == self.num_output_channels
        assert len(self.param_locations) == (
            self.num_input_channels if is_forward else self.num_output_channels)
        assert len(self.input_locations) == self.num_input_channels
        assert len(self.output_locations) == self.num_output_channels
        assert len(self.pass_grad_locations) == self.num_input_channels

    def isForwardDone(self) -> bool:
        return self.forward_count == self.num_input_channels

    def isBackwardDone(self) -> bool:
        # TODO: FIXME: prune
        return self.backward_count == self.num_output_channels

    def canForward(self, channel_ids) -> bool:
        for pred_op in self.pred_ops:
            if self.forward_count != 0 and \
                any(output_loc == MemoryType.SLOW
                    for output_loc in self.output_locations):
                return False
            for channel_id in channel_ids:
                if not pred_op.isForwardDone() \
                        or pred_op.output_locations[channel_id] == MemoryType.SLOW \
                        or self.param_locations[channel_id] == MemoryType.SLOW:
                    return False
        return True

    def canBackward(self, channel_ids) -> bool:
        for succ_op in self.succ_ops:
            # HACK: FIXME: each input channel is used by each output channel
            if any(input_loc == MemoryType.SLOW
                   for input_loc in self.input_locations):
                return False
            for channel_id in channel_ids:
                if not succ_op.isBackwardDone() \
                        or succ_op.pass_grad_locations[channel_id] == MemoryType.SLOW \
                        or self.param_locations[channel_id] == MemoryType.SLOW \
                        or self.grad_locations[channel_id] == MemoryType.SLOW:
                    return False
        return True

    def forward(self, channel_ids) -> Tuple[float, float]:
        assert self.canForward(channel_ids)
        r = self.num_input_channels / len(channel_ids)
        memory_peek = r * self.forward_memory_peek
        time_elapsed = r * self.forward_time_elapsed
        return memory_peek, time_elapsed

    def backward(self, channel_ids) -> Tuple[float, float]:
        assert self.canBackward(channel_ids)
        r = self.num_output_channels / len(channel_ids)
        memory_peek = r * self.backward_memory_peek
        time_elapsed = r * self.backward_time_elapsed
        # TODO: self.optimize_time_elapsed
        return memory_peek, time_elapsed

    def load(self, memory_type: str, channel_ids, bandwidth) -> Tuple[float, float]:
        assert memory_type in ["param", "input", "output", "pass_grad"]
        memory_block = getattr(self, memory_type + "_locations")
        r = len(channel_ids) / len(memory_block)
        for channel_id in channel_ids:
            assert memory_block[channel_id] == MemoryType.SLOW
            memory_block[channel_id] = MemoryType.FAST
        memory_delta = r * getattr(self, memory_type + '_size')
        time_elapsed = memory_delta / bandwidth
        return memory_delta, time_elapsed

    def store(self, memory_type: str, channel_ids, bandwidth) -> Tuple[float, float]:
        assert memory_type in ["param", "input", "output", "pass_grad"]
        memory_block = getattr(self, memory_type + "_locations")
        r = len(channel_ids) / len(memory_block)
        for channel_id in channel_ids:
            assert memory_block[channel_id] == MemoryType.FAST
            memory_block[channel_id] = MemoryType.SLOW
        memory_delta = r * getattr(self, memory_type + '_size')
        time_elapsed = memory_delta / bandwidth
        return memory_delta, time_elapsed


@dataclasses.dataclass
class RunTime:
    # machine related
    # NOTE: there exists two levels of hierarchical memory
    memory: float
    cross_level_bandwidth_read: float
    cross_level_bandwidth_write: float

    # TODO: accuracy related
    target_accuracy: float
