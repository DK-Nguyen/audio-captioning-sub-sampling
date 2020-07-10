#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger
from torch import Tensor
from torch.nn import Module

from modules.encoder_subsampling import SubSamplingEncoder
from modules.decoder_no_attention import DecoderNoAttention
from modules.attention_decoder_no_lm import AttentionDecoder

__docformat__ = 'reStructuredText'
__all__ = ['SubSamplingAttentionModel']


class SubSamplingAttentionModel(Module):

    def __init__(self,
                 input_dim_encoder: int,
                 hidden_dim_encoder: int,
                 output_dim_encoder: int,
                 dropout_p_encoder: float,
                 sub_sampling_factor_encoder: int,
                 sub_sampling_mode: int,
                 output_dim_h_decoder: int,
                 nb_classes: int,
                 dropout_p_decoder: float,
                 max_out_t_steps: int,
                 mode: int) \
            -> None:
        """
        Recurrent Neural Network with bi-directional GRU and attention
        for audio captioning on Clotho Dataset

        :param input_dim_encoder: Input dimensionality of the encoder.
        :type input_dim_encoder: int
        :param hidden_dim_encoder: Hidden dimensionality of the encoder.
        :type hidden_dim_encoder: int
        :param output_dim_encoder: Output dimensionality of the encoder.
        :type output_dim_encoder: int
        :param dropout_p_encoder: Encoder RNN dropout.
        :type dropout_p_encoder: float
        :param sub_sampling_factor_encoder: Sub-sampling rate for the encoder
        :type sub_sampling_factor_encoder: int
        :param output_dim_h_decoder: Hidden output dimensionality of the decoder.
        :type output_dim_h_decoder: int
        :param nb_classes: Amount of output classes.
        :type nb_classes: int
        :param dropout_p_decoder: Decoder RNN dropout.
        :type dropout_p_decoder: float
        :param max_out_t_steps: Maximum output time-steps of the decoder.
        :type max_out_t_steps: int
        :param mode: if mode is 0, use decoder without attention,
                     if mode is 1, use decoder with attention
        :type mode: int
        """
        super().__init__()

        logger_inner = logger.bind(is_caption=False, indent=1)
        if mode == 0:
            logger_inner.info(f'Sub sampling attention model {mode} - no attention')
        elif mode == 1:
            logger_inner.info(f'Sub sampling attention model {mode} - use attention')

        self.mode = mode
        self.max_out_t_steps: int = max_out_t_steps
        self.nb_classes: int = nb_classes

        self.encoder: Module = SubSamplingEncoder(
            input_dim=input_dim_encoder,
            hidden_dim=hidden_dim_encoder,
            output_dim=output_dim_encoder,
            dropout_p=dropout_p_encoder,
            sub_sampling_factor=sub_sampling_factor_encoder,
            sub_sampling_mode=sub_sampling_mode
        )

        if self.mode == 0:
            self.decoder_alzheimer: Module = DecoderNoAttention(
                input_dim=output_dim_encoder * 2,
                output_dim=output_dim_h_decoder,
                nb_classes=nb_classes,
                dropout_p=dropout_p_decoder
            )

        elif self.mode == 1:
            self.decoder_attention: Module = AttentionDecoder(
                input_dim=output_dim_encoder * 2,
                output_dim=output_dim_h_decoder,
                nb_classes=nb_classes,
                dropout_p=dropout_p_decoder
            )

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """
        Forward pass of the model

        :param x: Input features.
        :type x: torch.Tensor of size [batch_size, time_steps, num_input_features]
        :return predicted values.
        :rtype torch.Tensor of size [batch_size, max_out_t_steps, nb_classes]
        """

        batch_size, _, _ = x.shape

        if self.mode == 0:  # sub-sampling encoder, no attention decoder
            encoder_outputs: Tensor = self.encoder(x)[0]
            encoder_outputs = encoder_outputs[:, -1, :]  # get the last time step (fixed context vector)
            encoder_outputs = encoder_outputs.unsqueeze(1)\
                .expand(-1, self.max_out_t_steps, -1)  # modify the shape
            output = self.decoder_alzheimer(encoder_outputs)

        else:  # sub-sampling encoder, attention decoder
            # print(f'input shape: {x.shape}')
            encoder_outputs: Tensor = self.encoder(x)[0]
            # print(f'encoder output shape: {encoder_outputs.shape}')
            output = self.decoder_attention(encoder_outputs)  # the attention uses the whole outputs of the encoder

        assert output.shape == (batch_size, self.max_out_t_steps, self.nb_classes), \
            'output shape of the network is not of the right shape'

        return output


if __name__ == '__main__':
    import torch

    x = torch.rand((1, 2584, 64)).cuda()
    print(f'Input to SubSamplingAttentionModel shape: {x.shape}')
    model = SubSamplingAttentionModel(input_dim_encoder=64,
                                      hidden_dim_encoder=256,
                                      output_dim_encoder=256,
                                      dropout_p_encoder=0.25,
                                      sub_sampling_factor_encoder=4,
                                      sub_sampling_mode=1,
                                      output_dim_h_decoder=256,
                                      nb_classes=4367,
                                      dropout_p_decoder=0.25,
                                      max_out_t_steps=22,
                                      mode=0).cuda()
    print(model)
    y = model(x)
    y_numpy = y.cpu().data.numpy()
    print(f'Output of SubSamplingAttentionModel shape: {y.shape}')

# EOF
