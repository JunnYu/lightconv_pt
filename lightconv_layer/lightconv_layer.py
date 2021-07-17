# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lightconv_cuda
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class lightconvFunction(Function):
    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l

        outputs = lightconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = lightconv_cuda.backward(
            grad_output.contiguous(), ctx.padding_l, *ctx.saved_tensors
        )
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


class LightconvLayer(nn.Module):
    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding_l=None,
        weight_softmax=False,
        num_heads=1,
        weight_dropout=0.0,
        bias=False,
    ):
        super(LightconvLayer, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout_module = nn.Dropout(weight_dropout)

        self.weight = nn.Parameter(torch.Tensor(num_heads, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(input_size))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.permute(1, 2, 0).contiguous()
        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(self.weight, -1)
        if self.weight_dropout_module.p:
            weight = self.weight_dropout_module(weight)

        output = lightconvFunction.apply(x, weight, self.padding_l).permute(2, 0, 1)

        if self.bias is not None:
            output = output + self.bias.reshape(1, 1, -1)
        return output

    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
