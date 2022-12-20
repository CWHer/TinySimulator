import dataclasses
from enum import Enum


@dataclasses.dataclass
class RunTime:
    # machine related
    memory_limit: float  # fast memory limit
    cross_level_bandwidth_read: float
    cross_level_bandwidth_write: float

    # TODO: prune related
    # target_accuracy: float


class MemoryType(Enum):
    """
    NOTE: we assume the underlying machine with two levels of hierarchical memory,
            the slow memory (e.g., NVMe) CAN NOT be directly used in computation,
            but it supports full-duplex loading & storing
    """
    FAST = 1        # in fast memory
    SLOW = 2        # in slow memory
    NONE = 3        # not initialized
    POINTER = 4     # point at fast memory
    RUNNING = 5     # occupied by operator
