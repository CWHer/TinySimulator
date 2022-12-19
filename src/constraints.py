from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class Operator:
    # computation related
    memory_peek: float
    time_elapsed: float
    param_size: float
    output_size: float

    # network topology
    pred_ops: List[Operator]
    succ_ops: List[Operator]

    # pruning related
    channel_accuracy: List[float]


@dataclasses.dataclass
class RunTime:
    # machine related
    # NOTE: there exists two levels of hierarchical memory
    memory: float
    cross_level_bandwidth: float

    # accuracy related
    target_accuracy: float
