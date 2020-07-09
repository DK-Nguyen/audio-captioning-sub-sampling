#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Union, List
from torch import Tensor
from torch.nn import Module, Linear, ModuleList,\
        Sequential, LeakyReLU, Dropout, Tanh, Softmax

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['Attention']


class Attention(Module):
    def __init__(self,
                 input_dim: int,
                 h_dim: int,
                 dropout_p: Optional[float] = .25,
                 layers_dim: Optional[Union[List[int], None]] = None) \
            -> None:
        """Attention layer, according to D. Bahdanau.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param h_dim: Context dimensionality.
        :type h_dim: int
        :param dropout_p: Dropout probability for extra layers,\
                          default to .25
        :type dropout_p: float, optional
        :param layers_dim: Dimensionality for layers, defaults\
                           to None.
        :type layers_dim: list[int] | None, optional
        """
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.u = Linear(in_features=self.h_dim,
                        out_features=self.input_dim,
                        bias=False)
        self.w = Linear(in_features=self.input_dim,
                        out_features=self.input_dim,
                        bias=False)
        if layers_dim is not None:
            extra_layers = []
            layers_dim = [self.input_dim] + layers_dim
            for l_dim_i, l_dim in enumerate(layers_dim[1:-1]):
                extra_layers.append(
                        Linear(in_features=layers_dim[l_dim_i],
                        out_features=l_dim))
                extra_layers.append(LeakyReLU())
                extra_layers.append(Dropout(dropout_p))
            extra_layers.append(Linear(
                in_features=layers_dim[-2],
                out_features=layers_dim[-1],
                bias=False))
            self.extra_layers = Sequential(*extra_layers)
            self.output_dim = layers_dim[-1]
        else:
            self.extra_layers = None
            self.output_dim = self.input_dim
        self.output_layer = Linear(in_features=self.output_dim,
                                   out_features=1,
                                   bias=False)
        self.tanh = Tanh()
        self.leaky_relu = LeakyReLU()
        self.softmax = Softmax(dim=1)

    def forward(self,
                x: Tensor,
                context: Tensor) \
        -> Tensor:
        """Forward function of attention.

        :param x: Input tensor.
        :type x: torch.Tensor
        :param context: Context tensor.
        :type context: torch.Tensor
        :return: Attention weight.
        :rtype: torch.Tensor
        """
        context_h = self.u(context)
        _h = self.w(x) + context_h.unsqueeze(1).expand(-1, x.size()[1], -1)
        if self.extra_layers is not None:
            h = self.extra_layers(self.leaky_relu(_h))
        else:
            h = self.tanh(_h)
        out = self.softmax(self.output_layer(h))
        return out
# EOF
