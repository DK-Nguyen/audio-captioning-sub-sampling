#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, GRU, Dropout

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['SubSamplingEncoder']


class SubSamplingEncoder(Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_p: float,
                 nb_layers: int,
                 sub_sampling_factor: int) \
            -> None:
        """Encoder module.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param hidden_dim: Hidden dimensionality.
        :type hidden_dim: int
        :param output_dim: Output dimensionality.
        :type output_dim: int
        :param dropout_p: Dropout.
        :type dropout_p: float
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout: Dropout = Dropout(p=dropout_p)

        self.sub_sampling_factor = sub_sampling_factor

        self.input_layer: GRU = GRU(
            input_size=self.input_dim, hidden_size=self.hidden_dim,
            num_layers=1, bias=True, batch_first=True, bidirectional=True)

        self.layers = [GRU(
            input_size=self.hidden_dim*2, hidden_size=self.hidden_dim,
            num_layers=1, bias=True, batch_first=True, bidirectional=True)
            for _ in range(1, nb_layers)]

        self.input_layer.flatten_parameters()

        [self.layers[i].flatten_parameters() for i in range(len(self.layers))]

    def _l_pass(self,
                layer: Module,
                layer_input: Tensor) \
            -> Tensor:
        """Does the forward passing for a GRU layer.

        :param layer: GRU layer for forward passing.
        :type layer: torch.nn.Module
        :param layer_input: Input to the GRU layer.
        :type layer_input: torch.Tensor
        :return: Output of the GRU layer.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = layer_input.size()
        h = layer(layer_input)[0].view(b_size, t_steps, 2, -1)
        return self.dropout(cat([h[:, :, 0, :], h[:, :, 1, :]], dim=-1))

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """
        self.input_layer.flatten_parameters()
        [self.layers[i].flatten_parameters() for i in range(len(self.layers))]

        h = self._l_pass(self.input_layer, x)[:, ::self.sub_sampling_factor, :]

        if h.size()[-1]//2 == x.size()[-1]:
            h += cat([x, x][:, ::self.sub_sampling_factor, :], dim=-1)

        for a_layer in self.layers:
            h_ = self._l_pass(a_layer, h)[:, ::self.sub_sampling_factor, :]
            h = h[:, ::self.sub_sampling_factor, :] + h_ \
                if h.size()[-1] == h_.size()[-1] else h_

        return h

# EOF
