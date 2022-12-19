import dataclasses
from enum import Enum
from typing import List

from constraints import Operator


class DecisionPhase(Enum):
    FORWARD = 1
    BACKWARD = 2

class DecisionType(Enum):
    LOAD = 1
    COMPUTE = 2
    STORE = 3


@dataclasses.dataclass
class Decision:
    wall_time: float  # when this decision would happen
    decision_type: DecisionType
    decision_phase: DecisionPhase

    operator: Operator  # apply to which operator
    percentage: float  # e.g. for tensor parallelism

    # backward pass only
    pruned_channels: List[int]

