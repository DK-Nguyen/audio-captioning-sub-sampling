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
            self.linear_subsampling: Module = Sequential(
                Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                       out_features=self.hidden_dim*2),
                LeakyReLU()
            )

        elif self.sub_sampling_mode == 2:
            logger_inner.info(f'Using linear maxout sub-sampling with factor {self.sub_sampling_factor}')
            self.maxout_linear_1: Module = Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                                                  out_features=self.hidden_dim*2)
            self.maxout_linear_2: Module = Linear(in_features=self.hidden_dim*2*self.sub_sampling_factor,
                                                  out_features=self.hidden_dim*2)

        elif self.sub_sampling_mode == 3:
            logger_inner.info(f'Using rnn sub-sampling with factor {self.sub_sampling_factor}')
            self.rnn_sub_sampling: Module = GRU(
                input_size=self.hidden_dim*2,
                hidden_size=self.output_dim,
                batch_first=True,
                bidirectional=True
            )

    def _sub_sampling(self,
                      x: Tensor) \
            -> Tensor:
        """
        Doing sub-sampling for the input tensor x along the time-step

        :param x: the input Tensor of shape [batch_size, time_steps, num_features]
        :return the output Tensor.
        :rtype torch.Tensor of shape
                    [batch_size, time_steps/sub_sampling_rate, num_features]
        """

        # Removing redundant feature vectors at the end of the input matrix according to the sampling factor
        batch_size, num_time_steps, num_features = x.shape
        trimmed_x: Tensor = x[:, :(num_time_steps - num_time_steps % self.sub_sampling_factor), :]
        num_chunks: int = int(trimmed_x.shape[1] / self.sub_sampling_factor)
        # print(f'trimmed x shape: {trimmed_x.shape}')
        y: Tensor = torch.empty(batch_size, num_chunks, num_features).cuda()

        if self.sub_sampling_mode == 0:  # drop
            y = trimmed_x[:, 0::self.sub_sampling_factor, :]

        elif self.sub_sampling_mode == 1:  # using linear layer
            num_new_feat: int = int(num_features * self.sub_sampling_factor)
            concat_x: Tensor = trimmed_x.view(batch_size, num_chunks, num_new_feat)
            y: Tensor = self.linear_subsampling(concat_x)

            # Looping method
            # chunked_x: Tuple = torch.chunk(trimmed_x, num_chunks, dim=1)
            # for chunk in chunked_x:
            #     concatenated_chunk: Tensor = torch.reshape(chunk, (batch_size, 1, -1))
            #     # print(f'concatenated_chunk shape: {concatenated_chunk.shape}')
            #     sub_sampled_chunk: Tensor = self.linear_subsampling(concatenated_chunk)
            #     # print(f'sub_sampled_chunk shape: {sub_sampled_chunk.shape}')
            #     torch.cat((y, sub_sampled_chunk), dim=1)
            #     # print(y.shape)

        elif self.sub_sampling_mode == 2:  # using maxout linear for sub-sampling
            num_new_feat: int = int(num_features * self.sub_sampling_factor)
            concat_x: Tensor = trimmed_x.view(batch_size, num_chunks, num_new_feat)
            y: Tensor = torch.max(
                self.maxout_linear_1(concat_x),
                self.maxout_linear_2(concat_x)
            )

            # Looping method
            # chunked_x: Tuple = torch.chunk(trimmed_x, num_chunks, dim=1)
            # for chunk in chunked_x:
            #     concatenated_chunk: Tensor = torch.reshape(chunk, (batch_size, 1, -1))
            #     sub_sampled_1: Tensor = self.maxout_linear_1(concatenated_chunk)
            #     sub_sampled_2: Tensor = self.maxout_linear_2(concatenated_chunk)
            #     max_sub_sampled: Tensor = torch.max(sub_sampled_1, sub_sampled_2)
            #     torch.cat((y, max_sub_sampled), dim=1)
            #     print(f'sub_sampled_1 shape: {sub_sampled_1.shape}')
            #     print(f'sub_sampled_2 shape: {sub_sampled_2.shape}')
            #     print(f'max_sub_sampled shape: {max_sub_sampled.shape}')

        elif self.sub_sampling_mode == 3:  # using rnn
            # x_ = trimmed_x.view(-1, self.sub_sampling_factor, num_features)
            x_ = torch.reshape(trimmed_x, (-1, self.sub_sampling_factor, num_features))
            h = self.rnn_sub_sampling(x_)[-1]
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

        # let's take an example input x with size [16, 2584, 64], sub_sampling rate = 4
        # batch_size, first_layer_time_steps, _ = down_x.shape
        batch_size, first_layer_time_steps, _ = x.shape

        # first layer gru
        output_1 = self.gru_1(x)[0]  # H1: [16, 2584, 512]
        output_1 = output_1.view(batch_size, first_layer_time_steps, 2, -1)  # unpacking: [16, 2584, 2, 256]
        output_1 = self.dropout(torch.cat([output_1[:, :, 0, :], output_1[:, :, 1, :]], dim=-1))
        output_1 = self._sub_sampling(output_1)  # H1'': [16, 646, 512]
        # print(f'output_1 shape: {output_1.shape}')

        # second layer gru
        _, second_layer_time_steps, _ = output_1.shape
        output_2 = self.gru_2(output_1)[0]
        output_2 = output_2.view(batch_size, second_layer_time_steps, 2, -1)  # unpacking: [16, 646, 2, 256]
        output_2 = self.dropout(torch.cat([output_2[:, :, 0, :], output_2[:, :, 1, :]], dim=-1))  # H2': [16, 646, 512]
        output_2 = self._sub_sampling(output_2)  # [16, 162, 512]

        # third layer gru
        _, third_layer_time_steps, _ = output_2.shape
        input_third_layer = output_2 + self._sub_sampling(output_1)  # H2'': [16, 162, 512]
        output_3, hidden_3 = self.gru_3(input_third_layer)
        output_3 = output_3.view(batch_size, third_layer_time_steps, 2, -1)  # unpacking: [16, 162, 2, 256]
        output_3 = self.dropout(torch.cat([output_3[:, :, 0, :], output_3[:, :, 1, :]], dim=-1))  # H3: [16, 162, 512]
        hidden_3 = hidden_3.view(batch_size, 1, -1)

        # note: H2'' = sub_sampling(H2') + sub_sampling(H1'') [line 123]
        #            = sub_sampling(H2' + H1'') [illustrated in figure 1 of the paper & thesis]
        # in other words, sub_sampling is a linear function
        return output_3, hidden_3


if __name__ == '__main__':

    encoder_input: Tensor = torch.rand((16, 2584, 64)).cuda()
    encoder: Module = SubSamplingEncoder(input_dim=64,
                                         hidden_dim=256,
                                         output_dim=256,
                                         dropout_p=0.25,
                                         sub_sampling_factor=2,
                                         sub_sampling_mode=2).cuda()  # same params with the DCASE2020 baseline system
    print(encoder)
    print(f'Input to encoder sub-sampling shape: {encoder_input.shape}')
    encoder_outputs, encoder_hidden = encoder(encoder_input)
    print(f'Output of encoder sub-sampling shape: {encoder_outputs.shape}')
    # print(f'Hidden of encoder sub-sampling shape: {encoder_hidden.shape}')
    # print(encoder_outputs)

# EOF

