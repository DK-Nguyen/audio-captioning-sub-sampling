#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Union, List
from torch import zeros, cat, Tensor
from torch.nn import Module, GRUCell, Linear, Dropout
from torch.nn.functional import softmax
from .attention import Attention

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['AttentionDecoder']


class AttentionDecoder(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nb_classes: int,
                 dropout_p: float,
                 max_out_t_steps: Optional[int] = 22,
                 attention_dropout: Optional[float] = .25,
                 attention_dim: Optional[Union[List[int], None]] = None) \
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
        :param attention_dropout: Dropout for attention, defaults to .25.
        :type attention_dropout: float, optional
        :param attention_dim: Dimensionality of attention layers, defaults to None.
        :type attention_dim: list[int] | None, optional
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_classes = nb_classes
        self.max_out_t_steps = max_out_t_steps
        self.dropout: Module = Dropout(p=dropout_p)
        self.attention: Attention = Attention(
                input_dim=self.input_dim,
                h_dim=self.output_dim,
                dropout_p=attention_dropout,
                layers_dim=attention_dim)
        self.gru: Module = GRUCell(self.input_dim, self.output_dim)
        self.classifier: Module = Linear(self.output_dim, self.nb_classes)

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
        outputs = []
        for t_step in range(self.max_out_t_steps):
            att_weights = self.attention(h_in, h)
            weighted_h_in = h_in.mul(att_weights).sum(dim=1)
            h = self.gru(weighted_h_in, h)
            outputs.append(self.classifier(h).unsqueeze(1))
        return cat(outputs, dim=1)


# EOF
