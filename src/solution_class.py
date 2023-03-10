from __future__ import annotations

import dataclasses
from enum import Enum
from typing import List

from .operator_class import Operator


class DecisionType(Enum):
    LOAD = 1        # load data from slow memory
    STORE = 2       # store data to slow memory
    ALLOCATE = 3    # allocate fast memory
    PURGE = 4       # purge fast memory (NOTE: generated by the simulator)
    # REFER = 5       # make a fast memory pointer
    FORWARD = 6     # run operator forward
    BACKWARD = 7    # run operator backward
    OPTIMIZE = 8    # optimize the parameters
    PRUNE = 9       # prune along output channels (in backward phase)
    COMMIT = 10     # commit the computation (NOTE: generated by the simulator)


ZERO_COST_DECISIONS = [
    DecisionType.COMMIT, DecisionType.PRUNE,
    DecisionType.ALLOCATE, DecisionType.PURGE,
    # DecisionType.REFER,
]


COMPUTE_DECISIONS = {
    DecisionType.FORWARD: "forward",
    DecisionType.BACKWARD: "backward",
    DecisionType.OPTIMIZE: "optimize"
}


class MemoryBlockType(Enum):
    PARAM = "param"
    INPUT = "input"
    OUTPUT = "output"
    GRAD = "grad"
    PASS_GRAD = "pass_grad"


@dataclasses.dataclass
class Decision:
    wall_time: float  # when this decision happens
    decision_type: DecisionType

    operator: Operator  # apply to which operator
    # which memory block to load/store/allocate
    memory_block: MemoryBlockType
    # (NOTE: num input channels for forward, num output channels for backward)
    channel_ids: List[int]  # apply to which channels

    def __hash__(self) -> int:
        return id(self)

    def __lt__(self, other: Decision) -> bool:
        lhs_index = len(ZERO_COST_DECISIONS) \
            if not self.decision_type in ZERO_COST_DECISIONS \
            else ZERO_COST_DECISIONS.index(self.decision_type)
        rhs_index = len(ZERO_COST_DECISIONS) \
            if not other.decision_type in ZERO_COST_DECISIONS \
            else ZERO_COST_DECISIONS.index(other.decision_type)
        return self.wall_time < other.wall_time \
            if self.wall_time != other.wall_time \
            else lhs_index < rhs_index
