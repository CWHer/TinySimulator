import dataclasses
from enum import Enum
from typing import List

from component_class import Operator


class DecisionType(Enum):
    LOAD = 1        # load data from slow memory
    STORE = 2       # store data to slow memory
    ALLOCATE = 3    # allocate fast memory
    PURGE = 4       # purge fast memory (NOTE: generated by the simulator)
    FORWARD = 5     # run operator forward
    BACKWARD = 6    # run operator backward
    OPTIMIZE = 7    # optimize the parameters
    PRUNE = 8       # prune along output channels
    COMMIT = 9      # commit the computation (NOTE: generated by the simulator)


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

    # NOTE: this is only for commit decision
    #   is this commit the last commit for forward/backward ?
    is_last: bool = False

    def __hash__(self) -> int:
        return id(self)
