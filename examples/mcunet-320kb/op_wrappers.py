# fmt: off
import sys
sys.path.append('../..')

import numpy as np
from src.operator_class import Operator, ChannelType
# fmt: on


# global settings
add_coeff = [8.42175897e-08, 2.36026058e-02]
CPU_slowdown = 7
dsize = 4


def Conv2d(input_shape, in_channel, out_channel, kernel_size,
           stride, padding=(0, 0), groups=1):
    # NOTE:
    #   input_shape: (N,C,H,W), N==1
    #   Conv2d(288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=288, bias=False)
    profile_coeff = [6.90732982e-09, 8.72233251e-02]
    N_in, C_in, H_in, W_in = input_shape
    assert (C_in == in_channel)
    H_out = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    compute_size = out_channel / groups * in_channel / groups * \
        kernel_size[0] * kernel_size[1] * H_out * W_out * groups
    pred_time_elapse = np.poly1d(profile_coeff)(compute_size)
    output_size = out_channel * H_out * W_out
    im2col_size = in_channel * H_out * W_out * \
        kernel_size[0] * kernel_size[1] / groups
    input_size = in_channel * H_in * W_in
    param_size = kernel_size[0] * kernel_size[1] * \
        in_channel * out_channel / groups
    pred_optim_elapse = np.poly1d(add_coeff)(param_size)
    # NOTE:
    #   forward_memory_peak: output && im2col
    return Operator(
        forward_memory_peak=im2col_size,
        forward_time_elapsed=pred_time_elapse * CPU_slowdown,
        backward_memory_peak=im2col_size,
        backward_time_elapsed=2 * pred_time_elapse * CPU_slowdown,
        optimize_time_elapsed=pred_optim_elapse * CPU_slowdown,
        param_size=param_size,
        input_size=input_size,
        output_size=output_size,
        grad_size=param_size * dsize,
        pass_grad_size=input_size * dsize,
        num_input_channels=in_channel,
        num_output_channels=out_channel,
        channel_type=ChannelType.AllToAll
    )


def BatchNorm2d(input_shape):
    profile_coeff = [5.57222032e-07, 1.60851888e-01]
    _, features, H_in, W_in = input_shape
    input_size = features * H_in * W_in
    param_size = 2 * features
    pred_time_elapse = np.poly1d(profile_coeff)(input_size)
    return Operator(
        forward_memory_peak=4 * features,
        forward_time_elapsed=pred_time_elapse * CPU_slowdown,
        backward_memory_peak=input_size,
        backward_time_elapsed=2 * pred_time_elapse * CPU_slowdown,
        optimize_time_elapsed=np.poly1d(add_coeff)(param_size) * CPU_slowdown,
        param_size=param_size,
        input_size=input_size,
        output_size=input_size,
        grad_size=param_size * dsize,
        pass_grad_size=input_size * dsize,
        num_input_channels=features,
        num_output_channels=features,
        channel_type=ChannelType.AllToAll
    )


def ReLU(input_shape):
    profile_coeff = [1.38615789e-07, 1.33106265e-02]
    _, channels, H_in, W_in = input_shape
    input_size = channels * H_in * W_in
    pred_time_elapse = np.poly1d(profile_coeff)(input_size)
    return Operator(
        forward_memory_peak=0,
        forward_time_elapsed=pred_time_elapse * CPU_slowdown,
        backward_memory_peak=0,
        backward_time_elapsed=2 * pred_time_elapse * CPU_slowdown,
        optimize_time_elapsed=0,
        param_size=0,
        input_size=input_size,
        output_size=input_size,
        grad_size=0,
        pass_grad_size=input_size * dsize,
        num_input_channels=channels,
        num_output_channels=channels,
        channel_type=ChannelType.AllToAll
    )


def pBatchNorm2d(input_shape):
    profile_coeff = [5.57222032e-07, 1.60851888e-01]
    _, features, H_in, W_in = input_shape
    input_size = features * H_in * W_in
    param_size = 2 * features
    pred_time_elapse = np.poly1d(profile_coeff)(input_size)
    return Operator(
        forward_memory_peak=4 * features,
        forward_time_elapsed=pred_time_elapse * CPU_slowdown,
        backward_memory_peak=input_size,
        backward_time_elapsed=2 * pred_time_elapse * CPU_slowdown,
        optimize_time_elapsed=np.poly1d(add_coeff)(param_size) * CPU_slowdown,
        param_size=param_size,
        input_size=input_size / features,
        output_size=input_size / features,
        grad_size=param_size * dsize,
        pass_grad_size=input_size * dsize / features,
        num_input_channels=features,
        num_output_channels=features,
        channel_type=ChannelType.AllToAll
    )


def pReLU(input_shape):
    profile_coeff = [1.38615789e-07, 1.33106265e-02]
    _, channels, H_in, W_in = input_shape
    input_size = channels * H_in * W_in
    pred_time_elapse = np.poly1d(profile_coeff)(input_size)
    return Operator(
        forward_memory_peak=0,
        forward_time_elapsed=pred_time_elapse * CPU_slowdown,
        backward_memory_peak=0,
        backward_time_elapsed=2 * pred_time_elapse * CPU_slowdown,
        optimize_time_elapsed=0,
        param_size=0,
        input_size=input_size / channels,
        output_size=input_size / channels,
        grad_size=0,
        pass_grad_size=input_size * dsize / channels,
        num_input_channels=channels,
        num_output_channels=channels,
        channel_type=ChannelType.AllToAll
    )


def AvgPool(input_shape):
    _, channels, H_in, W_in = input_shape
    input_size = channels * H_in * W_in
    # NOTE:
    #   singleton operator: (1, 160, 6, 6) -> (1, 160)
    return Operator(
        forward_memory_peak=3 * input_size,
        forward_time_elapsed=0.036 * CPU_slowdown,
        backward_memory_peak=0,
        backward_time_elapsed=0.012 * CPU_slowdown,
        optimize_time_elapsed=0,
        param_size=0,
        input_size=input_size,
        output_size=channels,
        grad_size=0,
        pass_grad_size=input_size * dsize,
        num_input_channels=channels,
        num_output_channels=channels,
        channel_type=ChannelType.AllToAll
    )


def Add(input_shape):
    _, channels, H_in, W_in = input_shape
    input_size = channels * H_in * W_in
    pred_time_elapse = np.poly1d(add_coeff)(input_size)
    return Operator(
        forward_memory_peak=0,
        forward_time_elapsed=pred_time_elapse * CPU_slowdown,
        backward_memory_peak=0,
        backward_time_elapsed=2 * pred_time_elapse * CPU_slowdown,
        optimize_time_elapsed=0,
        param_size=0,
        input_size=input_size * 2,
        output_size=input_size,
        grad_size=0,
        pass_grad_size=input_size * 2 * dsize,
        num_input_channels=channels,
        num_output_channels=channels,
        channel_type=ChannelType.AllToAll
    )


# def Linear(input_shape):
#     input_size = 160
#     output_size = 1000
#     param_size = 1000 + 1000 * 160
#     # NOTE:
#     #   singleton operator: (1, 160) -> (1, 2)
#     return Operator(
#         forward_memory_peak=0,
#         forward_time_elapsed=(0.0464 + 0.0065) * CPU_slowdown,
#         backward_memory_peak=0,
#         backward_time_elapsed=(2 * 0.0464 + 0.011) * CPU_slowdown,
#         optimize_time_elapsed=np.poly1d(add_coeff)(param_size) * CPU_slowdown,
#         param_size=param_size,
#         input_size=input_size,
#         output_size=output_size,
#         grad_size=param_size,
#         pass_grad_size=input_size,
#         num_input_channels=1,
#         num_output_channels=1
#     )
