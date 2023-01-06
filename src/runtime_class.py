import dataclasses
from enum import Enum


@dataclasses.dataclass
class RunTime:
    # machine related
    # unit: Byte
    memory_limit: float  # fast memory limit
    # unit: Byte/ms
    cross_level_bandwidth_read: float
    cross_level_bandwidth_write: float

    target_accuracy: float


class MemoryType(Enum):
    """
    NOTE: we assume the underlying machine with two levels of hierarchical memory,
            the slow memory (e.g., NVMe) CAN NOT be directly used in computation,
            but it supports full-duplex loading & storing
    """
    FAST = 1        # in fast memory
    SLOW = 2        # in slow memory
    NONE = 3        # not initialized
    # HACK: FIXME: for simplicity, POINTER is not implemented
    #  as far as I'm concerned, the only application scenario is
    #  to point input(ch) at (pred_)output(ch) (though this is quite important),
    #  but I DO NOT find a elegant way to implement it 
    #  (maybe an additional COPY action is required)
    # POINTER = 4     # point at fast memory
    RUNNING = 5     # occupied by operator
