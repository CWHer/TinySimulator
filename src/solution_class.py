import dataclasses
from enum import Enum
from typing import List

from component_class import Operator


class DecisionPhase(Enum):
    FORWARD = 1
    BACKWARD = 2


class DecisionType(Enum):
    LOAD = 1
    COMPUTE = 2
    STORE = 3


class MemoryBlockType(Enum):
    PARAM = "param"
    INPUT = "input"
    OUTPUT = "output"
    GRAD = "grad"
    PASS_GRAD = "pass_grad"


@dataclasses.dataclass
class Decision:
    wall_time: float  # when this decision would happen
    decision_phase: DecisionPhase
    decision_type: DecisionType
    memory_block: MemoryBlockType  # which memory block to load/store

    operator: Operator  # apply to which operator
    channel_ids: List[int]  # apply to which channels
    # (num input channels for forward, num output channels for backward)

    # TODO: backward pass only
    pruned_channels: List[int]
