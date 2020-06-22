#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

from torch.nn import Module, GRU, Dropout
from torch import Tensor, cat

__docformat__ = 'reStructuredText'
__all__ = ['SubSamplingEncoder']


class SubSamplingEncoder(Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_p: float,
                 subsample_factor: int) \
            -> None:
        """Encoder with GRU and attention

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param hidden_dim: Hidden dimensionality.
        :type hidden_dim: int
        :param output_dim: Output dimensionality.
        :type output_dim: int
        :param dropout_p: Dropout.
        :type dropout_p: float
        :param subsample_factor: sub-sampling rate.
        :type subsample_factor: int
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        self.dropout: Module = Dropout(p=dropout_p)
        self.sub_sampling_rate: int = subsample_factor

        rnn_common_args = {
            'num_layers': 1,
            'bias': True,
            'batch_first': True,
            'bidirectional': True}

        self.gru_1: Module = GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            **rnn_common_args
        )

        self.gru_2: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.hidden_dim,
            **rnn_common_args
        )

        self.gru_3: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.output_dim,
            **rnn_common_args
        )

    def _sub_sampling(self,
                      x: Tensor,
                      dimension: int = 1) \
            -> Tensor:
        """
        Doing sub-sampling for the input tensor x

        :param x: the input Tensor of shape [batch_size, time_steps, num_features]
        :param dimension: the dimension along which to do sub-sampling.
               default value = 1, which sub-samples along the time_steps.
        :return the output Tensor.
        :rtype torch.Tensor of shape
                    [batch_size, time_steps/sub_sampling_rate, num_features]
        """
        if dimension == 1:
            y = x[:, 0::self.sub_sampling_rate, :]
        elif dimension == 2:
            y = x[:, :, 0::self.sub_sampling_rate]
        return y

    # noinspection DuplicatedCode
    def forward(self,
                x: Tensor) \
            -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor of shape [batch_size, time_steps (varies), input_dim_encoder]
        :return tuple [output_3, hidden_3]
        :rtype output_3 (output of the encoder): torch.Tensor of shape
                    [batch_size, time_steps/(sub_sampling_rate^2), output_dim_encoder*2]
               hidden_3 (the last hidden state of the encoder, can be used as the first hidden for the attention):
                    torch.Tensor of shape [batch_size, 1, output_dim_encoder*2]
        """

        # let's take an example input x with size [16, 2584, 64], sub_sampling rate = 4

        # down_x = self._sub_sampling(x)  # [16, 646, 512]
        # batch_size, first_layer_time_steps, _ = down_x.shape
        batch_size, first_layer_time_steps, _ = x.shape

        # first layer gru
        output_1 = self.gru_1(x)[0]
        output_1 = output_1.view(batch_size, first_layer_time_steps, 2, -1)  # unpacking: [16, 2584, 2, 256]
        output_1 = self.dropout(cat([output_1[:, :, 0, :], output_1[:, :, 1, :]], dim=-1))
        output_1 = self._sub_sampling(output_1)  # [16, 646, 512]

        # second layer gru
        _, second_layer_time_steps, _ = output_1.shape
        output_2 = self.gru_2(output_1)[0]
        output_2 = output_2.view(batch_size, second_layer_time_steps, 2, -1)  # unpacking: [16, 646, 2, 256]
        output_2 = self.dropout(cat([output_2[:, :, 0, :], output_2[:, :, 1, :]], dim=-1))
        output_2 = self._sub_sampling(output_2)  # [16, 162, 512]

        # third layer gru
        _, third_layer_time_steps, _ = output_2.shape
        input_third_layer = output_2 + self._sub_sampling(output_1)
        output_3, hidden_3 = self.gru_3(input_third_layer)
        output_3 = output_3.view(batch_size, third_layer_time_steps, 2, -1)  # unpacking: [16, 162, 2, 256]
        output_3 = self.dropout(cat([output_3[:, :, 0, :], output_3[:, :, 1, :]], dim=-1))  # [16, 162, 512]
        hidden_3 = hidden_3.view(batch_size, 1, -1)

        return output_3, hidden_3


if __name__ == '__main__':
    import torch

    encoder_input = torch.rand((16, 2584, 64)).cuda()
    print(f'Input to encoder sub-sampling simp shape: {encoder_input.shape}')
    encoder = SubSamplingEncoder(input_dim=64,
                                 hidden_dim=256,
                                 output_dim=256,
                                 dropout_p=0.25,
                                 subsample_factor=4).cuda()
    print(encoder)
    encoder_outputs, encoder_hidden = encoder(encoder_input)
    print(f'Output of encoder sub-sampling simp shape: {encoder_outputs.shape}')
    print(f'Hidden of encoder sub-sampling simp shape: {encoder_hidden.shape}')

# EOF

