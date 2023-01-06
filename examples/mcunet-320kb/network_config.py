
from typing import Dict, List

from op_wrappers import Add, BatchNorm2d, Conv2d, ReLU, pBatchNorm2d, pReLU


mcu_net: Dict[str, List] = {
    "first_conv": [
        Conv2d((1, 3, 176, 176), 3, 16, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1)),
        BatchNorm2d((1, 16, 88, 88)),
        ReLU((1, 16, 88, 88))
    ],
    "block0": [
        Conv2d((1, 16, 88, 88), 16, 16, kernel_size=(3, 3),
               stride=(1, 1), padding=(1, 1), groups=16),
        pBatchNorm2d((1, 16, 88, 88)),
        pReLU((1, 16, 88, 88)),
        Conv2d((1, 16, 88, 88), 16, 8, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 8, 88, 88))
    ],
    "block1": [
        Conv2d((1, 8, 88, 88), 8, 24, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 24, 88, 88)),
        ReLU((1, 24, 88, 88)),
        Conv2d((1, 24, 88, 88), 24, 24, kernel_size=(7, 7),
               stride=(2, 2), padding=(3, 3), groups=24),
        pBatchNorm2d((1, 24, 44, 44)),
        pReLU((1, 24, 44, 44)),
        Conv2d((1, 24, 44, 44), 24, 16, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 16, 44, 44))
    ],
    "block2": [
        Conv2d((1, 16, 44, 44), 16, 80, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 80, 44, 44)),
        ReLU((1, 80, 44, 44)),
        Conv2d((1, 80, 44, 44), 80, 80, kernel_size=(3, 3),
               stride=(1, 1), padding=(1, 1), groups=80),
        pBatchNorm2d((1, 80, 44, 44)),
        pReLU((1, 80, 44, 44)),
        Conv2d((1, 80, 44, 44), 80, 16, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 16, 44, 44)),
        Add((1, 16, 44, 44))
    ],
    "block3": [
        Conv2d((1, 16, 44, 44), 16, 80, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 80, 44, 44)),
        ReLU((1, 80, 44, 44)),
        Conv2d((1, 80, 44, 44), 80, 80, kernel_size=(3, 3),
               stride=(1, 1), padding=(1, 1), groups=80),
        pBatchNorm2d((1, 80, 44, 44)),
        pReLU((1, 80, 44, 44)),
        Conv2d((1, 80, 44, 44), 80, 16, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 16, 44, 44)),
        Add((1, 16, 44, 44))
    ],
    "block4": [
        Conv2d((1, 16, 44, 44), 16, 64, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 64, 44, 44)),
        ReLU((1, 64, 44, 44)),
        Conv2d((1, 64, 44, 44), 64, 64, kernel_size=(5, 5),
               stride=(1, 1), padding=(2, 2), groups=64),
        pBatchNorm2d((1, 64, 44, 44)),
        pReLU((1, 64, 44, 44)),
        Conv2d((1, 64, 44, 44), 64, 16, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 16, 44, 44)),
        Add((1, 16, 44, 44))
    ],
    "block5": [
        Conv2d((1, 16, 44, 44), 16, 80, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 80, 44, 44)),
        ReLU((1, 80, 44, 44)),
        Conv2d((1, 80, 44, 44), 80, 80, kernel_size=(5, 5),
               stride=(2, 2), padding=(2, 2), groups=80),
        pBatchNorm2d((1, 80, 44, 44)),
        pReLU((1, 80, 44, 44)),
        Conv2d((1, 80, 22, 22), 80, 24, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 24, 22, 22))
    ],
    "block6": [
        Conv2d((1, 24, 22, 22), 24, 120, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 120, 22, 22)),
        ReLU((1, 120, 22, 22)),
        Conv2d((1, 120, 22, 22), 120, 120, kernel_size=(5, 5),
               stride=(1, 1), padding=(2, 2), groups=120),
        pBatchNorm2d((1, 120, 22, 22)),
        pReLU((1, 120, 22, 22)),
        Conv2d((1, 120, 22, 22), 120, 24, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 24, 22, 22)),
        Add((1, 24, 22, 22))
    ],
    "block7": [
        Conv2d((1, 24, 22, 22), 24, 120, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 120, 22, 22)),
        ReLU((1, 120, 22, 22)),
        Conv2d((1, 120, 22, 22), 120, 120, kernel_size=(5, 5),
               stride=(1, 1), padding=(2, 2), groups=120),
        pBatchNorm2d((1, 120, 22, 22)),
        pReLU((1, 120, 22, 22)),
        Conv2d((1, 120, 22, 22), 120, 24, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 24, 22, 22)),
        Add((1, 24, 22, 22))
    ],
    "block8": [
        Conv2d((1, 24, 22, 22), 24, 120, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 120, 22, 22)),
        ReLU((1, 120, 22, 22)),
        Conv2d((1, 120, 22, 22), 120, 120, kernel_size=(3, 3),
               stride=(2, 2), padding=(1, 1), groups=120),
        pBatchNorm2d((1, 120, 22, 22)),
        pReLU((1, 120, 22, 22)),
        Conv2d((1, 120, 11, 11), 120, 40, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 40, 11, 11))
    ],
    "block9": [
        Conv2d((1, 40, 11, 11), 40, 240, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 240, 11, 11)),
        ReLU((1, 240, 11, 11)),
        Conv2d((1, 240, 11, 11), 240, 240, kernel_size=(7, 7),
               stride=(1, 1), padding=(3, 3), groups=240),
        pBatchNorm2d((1, 240, 11, 11)),
        pReLU((1, 240, 11, 11)),
        Conv2d((1, 240, 11, 11), 240, 40, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 40, 11, 11)),
        Add((1, 40, 11, 11))
    ],
    "block10": [
        Conv2d((1, 40, 11, 11), 40, 160, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 160, 11, 11)),
        ReLU((1, 160, 11, 11)),
        Conv2d((1, 160, 11, 11), 160, 160, kernel_size=(5, 5),
               stride=(1, 1), padding=(2, 2), groups=160),
        pBatchNorm2d((1, 160, 11, 11)),
        pReLU((1, 160, 11, 11)),
        Conv2d((1, 160, 11, 11), 160, 40, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 40, 11, 11)),
        Add((1, 40, 11, 11))
    ],
    "block11": [
        Conv2d((1, 40, 11, 11), 40, 200, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 200, 11, 11)),
        ReLU((1, 200, 11, 11)),
        Conv2d((1, 200, 11, 11), 200, 200, kernel_size=(5, 5),
               stride=(1, 1), padding=(2, 2), groups=200),
        pBatchNorm2d((1, 200, 11, 11)),
        pReLU((1, 200, 11, 11)),
        Conv2d((1, 200, 11, 11), 200, 48, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 48, 11, 11))
    ],
    "block12": [
        Conv2d((1, 48, 11, 11), 48, 240, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 240, 11, 11)),
        ReLU((1, 240, 11, 11)),
        Conv2d((1, 240, 11, 11), 240, 240, kernel_size=(7, 7),
               stride=(1, 1), padding=(3, 3), groups=240),
        pBatchNorm2d((1, 240, 11, 11)),
        pReLU((1, 240, 11, 11)),
        Conv2d((1, 240, 11, 11), 240, 48, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 48, 11, 11)),
        Add((1, 48, 11, 11))
    ],
    "block13": [
        Conv2d((1, 48, 11, 11), 48, 240, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 240, 11, 11)),
        ReLU((1, 240, 11, 11)),
        Conv2d((1, 240, 11, 11), 240, 240, kernel_size=(3, 3),
               stride=(1, 1), padding=(1, 1), groups=240),
        pBatchNorm2d((1, 240, 11, 11)),
        pReLU((1, 240, 11, 11)),
        Conv2d((1, 240, 11, 11), 240, 48, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 48, 11, 11)),
        Add((1, 48, 11, 11))
    ],
    "block14": [
        Conv2d((1, 48, 11, 11), 48, 288, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 288, 11, 11)),
        ReLU((1, 288, 11, 11)),
        Conv2d((1, 288, 11, 11), 288, 288, kernel_size=(3, 3),
               stride=(2, 2), padding=(1, 1), groups=288),
        pBatchNorm2d((1, 288, 11, 11)),
        pReLU((1, 288, 11, 11)),
        Conv2d((1, 288, 6, 6), 288, 96, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 96, 6, 6))
    ],
    "block15": [
        Conv2d((1, 96, 6, 6), 96, 480, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 480, 6, 6)),
        ReLU((1, 480, 6, 6)),
        Conv2d((1, 480, 6, 6), 480, 480, kernel_size=(7, 7),
               stride=(1, 1), padding=(3, 3), groups=480),
        pBatchNorm2d((1, 480, 6, 6)),
        pReLU((1, 480, 6, 6)),
        Conv2d((1, 480, 6, 6), 480, 96, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 96, 6, 6)),
        Add((1, 96, 6, 6))
    ],
    "block16": [
        Conv2d((1, 96, 6, 6), 96, 384, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 384, 6, 6)),
        ReLU((1, 384, 6, 6)),
        Conv2d((1, 384, 6, 6), 384, 384, kernel_size=(3, 3),
               stride=(1, 1), padding=(1, 1), groups=384),
        pBatchNorm2d((1, 384, 6, 6)),
        pReLU((1, 384, 6, 6)),
        Conv2d((1, 384, 6, 6), 384, 96, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 96, 6, 6)),
        Add((1, 96, 6, 6))
    ],
    "block17": [
        Conv2d((1, 96, 6, 6), 96, 480, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 480, 6, 6)),
        ReLU((1, 480, 6, 6)),
        Conv2d((1, 480, 6, 6), 480, 480, kernel_size=(7, 7),
               stride=(1, 1), padding=(3, 3), groups=480),
        pBatchNorm2d((1, 480, 6, 6)),
        pReLU((1, 480, 6, 6)),
        Conv2d((1, 480, 6, 6), 480, 160, kernel_size=(1, 1), stride=(1, 1)),
        BatchNorm2d((1, 160, 6, 6))
    ],
}


# HACK: This is generated by script
op_to_prune = [
    mcu_net["first_conv"][0], mcu_net["first_conv"][1], mcu_net["first_conv"][2],
    mcu_net["block0"][3], mcu_net["block0"][4],
    mcu_net["block1"][0], mcu_net["block1"][1], mcu_net["block1"][2], mcu_net["block1"][6], mcu_net["block1"][7],
    mcu_net["block2"][0], mcu_net["block2"][1], mcu_net["block2"][2], mcu_net["block2"][6], mcu_net["block2"][7], mcu_net["block2"][8],
    mcu_net["block3"][0], mcu_net["block3"][1], mcu_net["block3"][2], mcu_net["block3"][6], mcu_net["block3"][7], mcu_net["block3"][8],
    mcu_net["block4"][0], mcu_net["block4"][1], mcu_net["block4"][2], mcu_net["block4"][6], mcu_net["block4"][7], mcu_net["block4"][8],
    mcu_net["block5"][0], mcu_net["block5"][1], mcu_net["block5"][2], mcu_net["block5"][6], mcu_net["block5"][7],
    mcu_net["block6"][0], mcu_net["block6"][1], mcu_net["block6"][2], mcu_net["block6"][6], mcu_net["block6"][7], mcu_net["block6"][8],
    mcu_net["block7"][0], mcu_net["block7"][1], mcu_net["block7"][2], mcu_net["block7"][6], mcu_net["block7"][7], mcu_net["block7"][8],
    mcu_net["block8"][0], mcu_net["block8"][1], mcu_net["block8"][2], mcu_net["block8"][6], mcu_net["block8"][7],
    mcu_net["block9"][0], mcu_net["block9"][1], mcu_net["block9"][2], mcu_net["block9"][6], mcu_net["block9"][7], mcu_net["block9"][8],
    mcu_net["block10"][0], mcu_net["block10"][1], mcu_net["block10"][2], mcu_net["block10"][6], mcu_net["block10"][7], mcu_net["block10"][8],
    mcu_net["block11"][0], mcu_net["block11"][1], mcu_net["block11"][2], mcu_net["block11"][6], mcu_net["block11"][7],
    mcu_net["block12"][0], mcu_net["block12"][1], mcu_net["block12"][2], mcu_net["block12"][6], mcu_net["block12"][7], mcu_net["block12"][8],
    mcu_net["block13"][0], mcu_net["block13"][1], mcu_net["block13"][2], mcu_net["block13"][6], mcu_net["block13"][7], mcu_net["block13"][8],
    mcu_net["block14"][0], mcu_net["block14"][1], mcu_net["block14"][2], mcu_net["block14"][6], mcu_net["block14"][7],
    mcu_net["block15"][0], mcu_net["block15"][1], mcu_net["block15"][2], mcu_net["block15"][6], mcu_net["block15"][7], mcu_net["block15"][8],
    mcu_net["block16"][0], mcu_net["block16"][1], mcu_net["block16"][2], mcu_net["block16"][6], mcu_net["block16"][7], mcu_net["block16"][8],
    mcu_net["block17"][0], mcu_net["block17"][1], mcu_net["block17"][2], mcu_net["block17"][6], mcu_net["block17"][7],
]


# HACK: This is generated by script
op_next_to_pruned = [
    mcu_net["first_conv"][1], mcu_net["first_conv"][2],
    mcu_net["block0"][0], mcu_net["block0"][4],
    mcu_net["block1"][0], mcu_net["block1"][1], mcu_net["block1"][2], mcu_net["block1"][3], mcu_net["block1"][7],
    mcu_net["block2"][0], mcu_net["block2"][1], mcu_net["block2"][2], mcu_net["block2"][3], mcu_net["block2"][7], mcu_net["block2"][8],
    mcu_net["block3"][0], mcu_net["block3"][1], mcu_net["block3"][2], mcu_net["block3"][3], mcu_net["block3"][7], mcu_net["block3"][8],
    mcu_net["block4"][0], mcu_net["block4"][1], mcu_net["block4"][2], mcu_net["block4"][3], mcu_net["block4"][7], mcu_net["block4"][8],
    mcu_net["block5"][0], mcu_net["block5"][1], mcu_net["block5"][2], mcu_net["block5"][3], mcu_net["block5"][7],
    mcu_net["block6"][0], mcu_net["block6"][1], mcu_net["block6"][2], mcu_net["block6"][3], mcu_net["block6"][7], mcu_net["block6"][8],
    mcu_net["block7"][0], mcu_net["block7"][1], mcu_net["block7"][2], mcu_net["block7"][3], mcu_net["block7"][7], mcu_net["block7"][8],
    mcu_net["block8"][0], mcu_net["block8"][1], mcu_net["block8"][2], mcu_net["block8"][3], mcu_net["block8"][7],
    mcu_net["block9"][0], mcu_net["block9"][1], mcu_net["block9"][2], mcu_net["block9"][3], mcu_net["block9"][7], mcu_net["block9"][8],
    mcu_net["block10"][0], mcu_net["block10"][1], mcu_net["block10"][2], mcu_net["block10"][3], mcu_net["block10"][7], mcu_net["block10"][8],
    mcu_net["block11"][0], mcu_net["block11"][1], mcu_net["block11"][2], mcu_net["block11"][3], mcu_net["block11"][7],
    mcu_net["block12"][0], mcu_net["block12"][1], mcu_net["block12"][2], mcu_net["block12"][3], mcu_net["block12"][7], mcu_net["block12"][8],
    mcu_net["block13"][0], mcu_net["block13"][1], mcu_net["block13"][2], mcu_net["block13"][3], mcu_net["block13"][7], mcu_net["block13"][8],
    mcu_net["block14"][0], mcu_net["block14"][1], mcu_net["block14"][2], mcu_net["block14"][3], mcu_net["block14"][7],
    mcu_net["block15"][0], mcu_net["block15"][1], mcu_net["block15"][2], mcu_net["block15"][3], mcu_net["block15"][7], mcu_net["block15"][8],
    mcu_net["block16"][0], mcu_net["block16"][1], mcu_net["block16"][2], mcu_net["block16"][3], mcu_net["block16"][7], mcu_net["block16"][8],
    mcu_net["block17"][0], mcu_net["block17"][1], mcu_net["block17"][2], mcu_net["block17"][3], mcu_net["block17"][7],
]


# HACK: This is generated by script
channels_to_prune = [
    15, 15, 15,
    6, 6,
    21, 21, 21, 11, 11,
    72, 72, 72, 14, 14, 14,
    79, 79, 79, 11, 11, 11,
    60, 60, 60, 14, 14, 14,
    67, 67, 67, 16, 16,
    115, 115, 115, 17, 17, 17,
    119, 119, 119, 20, 20, 20,
    105, 105, 105, 28, 28,
    238, 238, 238, 31, 31, 31,
    158, 158, 158, 37, 37, 37,
    189, 189, 189, 39, 39,
    236, 236, 236, 38, 38, 38,
    239, 239, 239, 44, 44, 44,
    275, 275, 275, 84, 84,
    477, 477, 477, 84, 84, 84,
    379, 379, 379, 85, 85, 85,
    387, 387, 387, 33, 33,
]
