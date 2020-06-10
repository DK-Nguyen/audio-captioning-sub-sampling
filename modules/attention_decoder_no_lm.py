#!/usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
from typing import Optional

from torch import zeros, cat, Tensor
from torch.nn import Module, GRUCell, Linear, Dropout, Sequential, LeakyReLU
from torch.nn.functional import softmax

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['AttentionDecoder']


class AttentionDecoder(Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nb_classes: int,
                 dropout_p: float,
                 num_attn_layers: int = 0,
                 first_attn_layer_output_dim: int = 0,
                 max_out_t_steps: Optional[int] = 22) \
            -> None:
        """Attention decoder for the baseline audio captioning method.

        :param input_dim: Input dimensionality for the RNN.
        :type input_dim: int
        :param output_dim: Output dimensionality for the RNN.
        :type output_dim: int
        :param nb_classes: Amount of amount classes.
        :type nb_classes: int
        :param dropout_p: RNN dropout.
        :type dropout_p: float
        :param max_out_t_steps: Maximum output time steps during inference.
        :type max_out_t_steps: int
        """
        logger_inner = logger.bind(is_caption=False, indent=1)
        logger_inner.info(f'Decoder with attention, no lm, {num_attn_layers} attn layers, '
                          f'first output attn layer dim: {first_attn_layer_output_dim}')

        super(AttentionDecoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_classes = nb_classes
        self.max_out_t_steps = max_out_t_steps
        self.first_attn_layer_output_dim = first_attn_layer_output_dim

        self.dropout: Module = Dropout(p=dropout_p)

        if num_attn_layers == 1:
            self.attention: Module = Linear(in_features=self.input_dim + self.output_dim,
                                            out_features=1,
                                            bias=True)
        elif num_attn_layers == 2:
            self.attention: Module = Sequential(
                Linear(in_features=self.input_dim + self.output_dim,
                       out_features=self.first_attn_layer_output_dim,
                       bias=True),
                LeakyReLU(),
                Linear(in_features=self.first_attn_layer_output_dim,
                       out_features=1,
                       bias=True)
            )
        elif num_attn_layers == 3:
            self.attention: Module = Sequential(
                Linear(in_features=self.input_dim + self.output_dim,
                       out_features=self.first_attn_layer_output_dim,
                       bias=True),
                Linear(in_features=first_attn_layer_output_dim,
                       out_features=int(self.first_attn_layer_output_dim/2),
                       bias=True),
                Linear(in_features=int(self.first_attn_layer_output_dim/2),
                       out_features=1,
                       bias=True)
            )
        else:
            logger_inner.info(f'No more than 3 layers for attention')

        self.gru: Module = GRUCell(self.input_dim, self.output_dim)
        self.classifier: Module = Linear(self.output_dim, self.nb_classes)

    def _attention(self,
                   h_i: Tensor,
                   h_h: Tensor) \
            -> Tensor:
        """Application of attention.

        :param h_i: Input sequence.
        :type h_i: torch.Tensor
        :param h_h: Previous hidden state.
        :type h_h: torch.Tensor
        :return: Attention weights.
        :rtype: torch.Tensor
        """
        h: Tensor = cat([
            h_i,
            h_h.unsqueeze(1).expand(-1, h_i.size()[1], -1)
        ], dim=-1)

        return softmax(self.attention(h), dim=1)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the attention decoder.

        :param x: Input sequence to the decoder.
        :type x: torch.Tensor
        :param y: Target values.
        :type y: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """
        h_in = self.dropout(x)
        device = h_in.device
        b_size = h_in.size()[0]

        h = zeros(b_size, self.output_dim).to(device)
        outputs = zeros(b_size, self.max_out_t_steps, self.nb_classes).to(device)

        for t_step in range(self.max_out_t_steps):
            att_weights = self._attention(h_in, h)
            weighted_h_in = h_in.mul(att_weights).sum(dim=1)

            h = self.gru(weighted_h_in, h)
            outputs[:, t_step, :] = self.classifier(h)

        return outputs

# EOF
