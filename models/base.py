import torch
import torch.nn as nn
import torch.nn.functional as F


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


def calculate_padding(input_size, kernel_size, stride):
    def calculate_padding_1D(input_size, kernel_size, stride):
        if (input_size % stride == 0):
            pad = max(kernel_size-stride, 0)
        else:
            pad = max(kernel_size-(input_size % stride), 0)

        return pad

    if type(kernel_size) != tuple:
        kernel_size_1 = kernel_size
        kernel_size_2 = kernel_size
    else:
        kernel_size_1 = kernel_size[0]
        kernel_size_2 = kernel_size[1]

    if type(stride) != tuple:
        stride_1 = stride
        stride_2 = stride
    else:
        stride_1 = stride[0]
        stride_2 = stride[1]

    padding1 = calculate_padding_1D(input_size[0], kernel_size_1, stride_1)
    padding2 = calculate_padding_1D(input_size[1], kernel_size_2, stride_2)

    pad_top = padding1//2
    pad_bottom = padding1 - pad_top
    pad_left = padding2//2
    pad_right = padding2 - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


def SAME_padding(x, ksize, stride):
    padding = calculate_padding(x.shape[2:], ksize, stride)
    return F.pad(x, padding)


def transpose_padding_same(x, target_shape, stride):
    target_shape = torch.tensor(target_shape[2:])
    current_shape = torch.tensor(x.shape[2:])

    padding_remove = (current_shape-target_shape)
    left = padding_remove//2
    right = padding_remove//2+padding_remove % 2
    if padding_remove[0] == 0:
        return x[:, :, :, left[1]:-right[1]]
    elif padding_remove[1] == 0:
        return x[:, :, left[0]:-right[0], :]
    else:
        return x[:, :, left[0]:-right[0], left[1]:-right[1]]