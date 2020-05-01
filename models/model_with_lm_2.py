#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch import Tensor
from torch.nn import Module

from modules.encoder import Encoder
from modules.attention_decoder_lm_extra_layers \
    import AttentionDecoderWithExtraLayers as AttentionDecoder

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ModelWithLM2']


class ModelWithLM2(Module):

    def __init__(self,
                 input_dim_encoder: int,
                 hidden_dim_encoder: int,
                 output_dim_encoder: int,
                 dropout_p_encoder: float,
                 output_dim_h_decoder: int,
                 dropout_p_decoder: float,
                 gamma_factor: float,
                 mul_factor: float,
                 min_prob: float,
                 max_prob: float,
                 max_out_t_steps: int,
                 nb_classes: int,
                 nb_gru_layers: Optional[int] = 0) \
            -> None:
        """

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
        :param dropout_p_decoder: Decoder RNN dropout.
        :type dropout_p_decoder: float
        :param gamma_factor: Gamma factor for teacher forcing.
        :type gamma_factor: float
        :param mul_factor: Multiplication factor for teacher forcing.
        :type mul_factor: float
        :param min_prob: Minimum probability for teacher forcing.
        :type min_prob: float
        :param max_prob: Maximum probability for teacher forcing.
        :type max_prob: float
        :param max_out_t_steps: Maximum output steps of classifier.
        :type max_out_t_steps: int
        :param nb_classes: Amount of classes.
        :type nb_classes: int
        :param nb_gru_layers: Amount of GRU layers to use.
        :type nb_gru_layers: int
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
            gamma_factor=gamma_factor,
            mul_factor=mul_factor,
            min_prob=min_prob,
            max_prob=max_prob,
            max_out_t_steps=max_out_t_steps,
            nb_gru_layers=nb_gru_layers)

    @property
    def batch_counter(self) \
            -> int:
        """Getter for the batch counter.

        :return: Batch count.
        :rtype: int
        """
        return self.decoder.batch_counter

    @batch_counter.setter
    def batch_counter(self,
                      value: int) \
            -> None:
        """Setter for batch counter.

        :param value: New value for batch count.
        :type value: int
        """
        self.decoder.batch_counter = value

    @property
    def min_prob(self) \
            -> float:
        """Getter for the min_prob attribute.

        :return: The minimum probability for\
                 selecting predictions.
        :rtype: float
        """
        return self.decoder.min_prob

    @min_prob.setter
    def min_prob(self,
                 value: float) \
            -> None:
        """Setter for the min_prob attribute.

        :param value: The new value of the min_prob.
        :type value: float
        """
        self.decoder.min_prob(value)

    def forward(self,
                x: Tensor,
                x_2: Tensor) \
            -> Tensor:
        """Forward pass of the LM based model.

        :param x: Input to the model
        :type x: torch.Tensor
        :param x_2: Teacher forcing input to the model.
        :type x_2: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        h = self.encoder(x)
        return self.decoder(h, x_2)

# EOF
