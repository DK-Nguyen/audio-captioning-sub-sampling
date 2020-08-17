#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple
from loguru import logger

import torch
from torch.nn import Module, GRU, Dropout, Linear, Sequential, LeakyReLU
from torch import Tensor

__docformat__ = 'reStructuredText'
__all__ = ['SubSamplingEncoder']


class SubSamplingEncoder(Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_p: float,
                 sub_sampling_factor: int,
                 sub_sampling_mode: int = 0) \
            -> None:
        """Encoder with sub-sampling, GRU and attention

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param hidden_dim: Hidden dimensionality.
        :type hidden_dim: int
        :param output_dim: Output dimensionality.
        :type output_dim: int
        :param dropout_p: Dropout.
        :type dropout_p: float
        :param sub_sampling_factor: sub-sampling rate.
        :type sub_sampling_factor: int
        :param sub_sampling_mode: if value is 0, just drop the vectors along the time-step according to subsample_factor
                                  if value is 1, using a linear layer as a sub-sampling mechanism
                                  if value is 2, using a linear maxout sub-sampling
                                  if value is 3, using rnn sub-sampling
        :type sub_sampling_mode: int
        """
        super().__init__()

        logger_inner = logger.bind(is_caption=False, indent=1)

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.sub_sampling_factor: int = sub_sampling_factor
        self.sub_sampling_mode: int = sub_sampling_mode

        self.dropout: Module = Dropout(p=dropout_p)

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

        if self.sub_sampling_mode == 0:
            logger_inner.info(f'Using dropping sub-sampling with factor {self.sub_sampling_factor}')

        elif self.sub_sampling_mode == 1:
            logger_inner.info(f'Using linear sub-sampling with factor {self.sub_sampling_factor}')

            self.linear_subsampling_1: Module = Sequential(
                Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                       out_features=self.hidden_dim*2),
                LeakyReLU()
            )
            self.linear_subsampling_2: Module = Sequential(
                Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                       out_features=self.hidden_dim*2),
                LeakyReLU()
            )

        elif self.sub_sampling_mode == 2:
            logger_inner.info(f'Using  maxout sub-sampling with factor {self.sub_sampling_factor}')
            # first maxout sub_sampling layer
            self.maxout_linear_1_1: Module = Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                                                    out_features=self.hidden_dim*2)
            self.maxout_linear_1_2: Module = Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                                                    out_features=self.hidden_dim*2)
            # second maxout sub_sampling layer
            self.maxout_linear_2_1: Module = Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                                                    out_features=self.hidden_dim*2)
            self.maxout_linear_2_2: Module = Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                                                    out_features=self.hidden_dim*2)

        elif self.sub_sampling_mode == 3:
            logger_inner.info(f'Using rnn sub-sampling with factor {self.sub_sampling_factor}')

            self.rnn_sub_sampling_1: Module = GRU(
                input_size=self.hidden_dim*2,
                hidden_size=self.output_dim,
                batch_first=True,
                bidirectional=True
            )
            self.rnn_sub_sampling_2: Module = GRU(
                input_size=self.hidden_dim*2,
                hidden_size=self.output_dim,
                batch_first=True,
                bidirectional=True
            )

    def _sub_sampling(self,
                      x: Tensor,
                      layer: int) \
            -> Tensor:
        """
        Doing sub-sampling for the input tensor x along the time-step

        :param x: the input Tensor of shape [batch_size, time_steps, num_features]
        :param layer: the layer that calls sub_sampling, we need this for learnable sub_sampling, as
                       different sub_sampling layers have different parameters to learn
        :return the output Tensor.
        :rtype torch.Tensor of shape
                    [batch_size, time_steps/sub_sampling_rate, num_features]
        """

        # Removing redundant feature vectors at the end of the input matrix according to the sampling factor
        batch_size, num_time_steps, num_features = x.shape
        trimmed_x: Tensor = x[:, :(num_time_steps - num_time_steps % self.sub_sampling_factor), :]
        num_chunks: int = int(trimmed_x.shape[1] / self.sub_sampling_factor)
        y: Tensor = torch.empty(batch_size, num_chunks, num_features).cuda()

        if self.sub_sampling_mode == 0:  # drop feature vectors (non_learnable sub_sampling)
            y = trimmed_x[:, 0::self.sub_sampling_factor, :]

        elif self.sub_sampling_mode == 1:  # using a linear with leaky relu as the learnable sub_sampling
            num_new_feat: int = int(num_features * self.sub_sampling_factor)
            x_: Tensor = trimmed_x.view(batch_size, num_chunks, num_new_feat)
            if layer == 1:
                y = self.linear_subsampling_1(x_)
            elif layer == 2:
                y = self.linear_subsampling_2(x_)

        elif self.sub_sampling_mode == 2:  # using maxout as the learnable sub_sampling
            num_new_feat: int = int(num_features * self.sub_sampling_factor)
            x_: Tensor = trimmed_x.view(batch_size, num_chunks, num_new_feat)
            if layer == 1:
                y = torch.max(
                    self.maxout_linear_1_1(x_),
                    self.maxout_linear_1_2(x_)
                )
            elif layer == 2:
                y = torch.max(
                    self.maxout_linear_2_1(x_),
                    self.maxout_linear_2_2(x_)
                )

        elif self.sub_sampling_mode == 3:  # using rnn as the learnable sub_sampling
            x_: Tensor = torch.reshape(trimmed_x,
                                       (-1, self.sub_sampling_factor, num_features))
            if layer == 1:
                h: Tensor = self.rnn_sub_sampling_1(x_)[-1]
                y = h.view(batch_size, -1, num_features)
            elif layer == 2:
                h: Tensor = self.rnn_sub_sampling_2(x_)[-1]
                y = h.view(batch_size, -1, num_features)

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

        # refer to https://arxiv.org/abs/2007.02676 for more information in details
        # let's take an example input x with size [16, 2584, 64], sub_sampling rate = 4

        # batch_size, first_layer_time_steps, _ = down_x.shape
        batch_size, first_layer_time_steps, _ = x.shape

        # first layer gru
        h1 = self.gru_1(x)[0]  # H1: [16, 2584, 512]
        # unpacking then concatenate the output of gru layer according to
        # https://pytorch.org/docs/master/generated/torch.nn.GRU.html
        h1 = h1.view(batch_size, first_layer_time_steps, 2, -1)  # unpacking: [16, 2584, 2, 256]
        h1 = self.dropout(torch.cat([h1[:, :, 0, :], h1[:, :, 1, :]], dim=-1))  # H1: [16, 2584, 512]
        h1_pp = self._sub_sampling(h1, layer=1)  # H1'': [16, 646, 512]

        # second layer gru
        _, second_layer_time_steps, _ = h1_pp.shape
        h2_p = self.gru_2(h1_pp)[0]  # H2': [16, 646, 512]
        h2_p = h2_p.view(batch_size, second_layer_time_steps, 2, -1)  # unpacking: [16, 646, 2, 256]
        h2_p = self.dropout(torch.cat([h2_p[:, :, 0, :], h2_p[:, :, 1, :]], dim=-1))  # H2': [16, 646, 512]
        h2 = h1_pp + h2_p  # residual connection
        h2_pp = self._sub_sampling(h2, layer=2)  # H2'': [16, 161, 512]

        # third layer gru
        _, third_layer_time_steps, _ = h2_pp.shape
        h3_p, _ = self.gru_3(h2_pp)
        h3_p = h3_p.view(batch_size, third_layer_time_steps, 2, -1)  # unpacking: [16, 161, 2, 256]
        h3 = self.dropout(torch.cat([h3_p[:, :, 0, :], h3_p[:, :, 1, :]], dim=-1))  # H3: [16, 161, 512]

        return h3


if __name__ == '__main__':

    encoder_input: Tensor = torch.rand((16, 2584, 64)).cuda()
    encoder: Module = SubSamplingEncoder(input_dim=64,
                                         hidden_dim=256,
                                         output_dim=256,
                                         dropout_p=0.25,
                                         sub_sampling_factor=1,
                                         sub_sampling_mode=0).cuda()
    print(encoder)
    print(f'Input to encoder sub-sampling shape: {encoder_input.shape}')
    encoder_outputs = encoder(encoder_input)
    print(f'Output of encoder sub-sampling shape: {encoder_outputs.shape}')

# EOF

