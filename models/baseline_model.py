#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor

from modules.encoder import Encoder
from modules.attention_decoder_no_lm import AttentionDecoder

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['BaselineModel']


class BaselineModel(Module):

    def __init__(self,
                 input_dim_encoder: int,
                 hidden_dim_encoder: int,
                 output_dim_encoder: int,
                 dropout_p_encoder: float,
                 output_dim_h_decoder: int,
                 nb_classes: int,
                 dropout_p_decoder: float,
                 max_out_t_steps: int) \
            -> None:
        """Baseline method for audio captioning with Clotho dataset.

        :param input_dim_encoder: Input dimensionality of the encoder.
        :type input_dim_encoder: int
        :param hidden_dim_encoder: Hidden dimensionality of the encoder.
        :type hidden_dim_encoder: int
        :param output_dim_encoder: Output dimensionality of the encoder.
        :type output_dim_encoder: int
        :param dropout_p_encoder: Encoder RNN dropout.
        :type dropout_p_encoder: float
        :param output_dim_h_decoder: Hidden output dimensionality of the decoder.
        :type output_dim_h_decoder: int
        :param nb_classes: Amount of output classes.
        :type nb_classes: int
        :param dropout_p_decoder: Decoder RNN dropout.
        :type dropout_p_decoder: float
        :param max_out_t_steps: Maximum output time-steps of the decoder.
        :type max_out_t_steps: int
        """
        super().__init__()

        self.encoder: Module = Encoder(
            input_dim=input_dim_encoder,
            hidden_dim=hidden_dim_encoder,
            output_dim=output_dim_encoder,
            dropout_p=dropout_p_encoder)

        self.decoder: Module = AttentionDecoder(
            input_dim=output_dim_encoder * 2,
            output_dim=output_dim_h_decoder,
            nb_classes=nb_classes,
            dropout_p=dropout_p_decoder,
            max_out_t_steps=max_out_t_steps)

    def forward(self,
                x: Tensor,
                x_2: None) \
            -> Tensor:
        """Forward pass of the baseline method.

        :param x: Input features.
        :type x: torch.Tensor
        :param x_2: Place holder for compatibility.
        :type x_2: None
        :return: Predicted values.
        :rtype: torch.Tensor
        """
        h_encoder: Tensor = self.encoder(x)
        return self.decoder(h_encoder)

# EOF
