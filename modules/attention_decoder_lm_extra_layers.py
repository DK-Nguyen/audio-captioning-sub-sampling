#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch import zeros, cat, Tensor, ones_like, device as torch_device, zeros_like
from torch.nn import Module, GRUCell, Linear, Dropout, Sequential, \
    LeakyReLU, BatchNorm1d
from torch.nn.functional import softmax

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['AttentionDecoderWithExtraLayers']


class AttentionDecoderWithExtraLayers(Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nb_classes: int,
                 dropout_p: float,
                 gamma_factor: float,
                 mul_factor: float,
                 min_prob: float,
                 max_prob: float,
                 max_out_t_steps: Optional[int] = 22,
                 nb_gru_layers: Optional[int] = 0) \
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
        :param gamma_factor: Gamma factor for scheduled sampling.
        :type gamma_factor: float
        :param mul_factor: Multiplication factor for scheduled sampling.
        :type mul_factor: float
        :param min_prob: Minimum probability for selecting predictions.
        :type min_prob: float
        :param max_prob: Maximum probability for selecting predictions.
        :type max_prob: float
        :param max_out_t_steps: Maximum output time steps during\
                                inference, defaults to 22.
        :type max_out_t_steps: int, optional
        :param nb_gru_layers: Amount of GRU layers to use, default to 0.
        :type nb_gru_layers: int, optional
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.nb_classes: int = nb_classes

        self._batch_counter: int = 0
        self.gamma_factor: float = gamma_factor/mul_factor
        self._min_prob: float = 1 - min_prob
        self.max_prob: float = max_prob
        self.iteration: int = 0

        self.max_out_t_steps: int = max_out_t_steps

        self.dropout: Dropout = Dropout(dropout_p)

        self.attention: Sequential = Sequential(
            Linear(in_features=self.input_dim + self.output_dim,
                   out_features=512),
            LeakyReLU(), Dropout(.25),
            Linear(in_features=512, out_features=256),
            LeakyReLU(), Dropout(.25),
            Linear(in_features=256, out_features=1))

        self.rnn: GRUCell = GRUCell(
            input_size=self.input_dim + self.nb_classes,
            hidden_size=self.output_dim)

        self.rnn_2: GRUCell = GRUCell(
            input_size=self.output_dim + self.input_dim,
            hidden_size=self.output_dim)

        self.rnn_3: GRUCell = GRUCell(
            input_size=self.output_dim + self.input_dim,
            hidden_size=self.output_dim)

        self.classifier: Linear = Linear(
            in_features=self.output_dim,
            out_features=self.nb_classes)

    @property
    def batch_counter(self) \
            -> int:
        """Getter for the batch counter.

        :return: Batch count.
        :rtype: int
        """
        return self._batch_counter

    @batch_counter.setter
    def batch_counter(self,
                      value: int) \
            -> None:
        """Setter for batch counter.

        :param value: New value for batch count.
        :type value: int
        """
        self._batch_counter = value

    @property
    def min_prob(self) \
            -> float:
        """Getter for the min_prob attribute.

        :return: The minimum probability for\
                 selecting predictions.
        :rtype: float
        """
        return 1 - self._min_prob

    @min_prob.setter
    def min_prob(self,
                 value: float) \
            -> None:
        """Setter for the min_prob attribute.

        :param value: The new value of the min_prob.
        :type value: float
        """
        self._min_prob = 1 - value

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

    def scheduled_sampling(self) \
            -> float:
        """Returns the probability to select
        the predicted value.

        :return: The current probability of\
                 selecting predictions.
        :rtype: float
        """
        p: float = self.iteration/self.batch_counter
        d: float = Tensor([-self.gamma_factor * p]).exp().item()

        return min(
            self.max_prob,
            1 - min(self.min_prob, (2 / (1 + d)) - 1))

    def forward(self,
                x: Tensor,
                y: Tensor) \
            -> Tensor:
        """Forward pass of the attention decoder.

        :param x: Input sequence to the decoder.
        :type x: torch.Tensor
        :param y: Target values.
        :type y: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """
        device = x.device

        return self._inference_pass(x, device) if y is None \
            else self._training_pass(x, y, device)

    def _training_pass(self,
                       x: Tensor,
                       y: Tensor,
                       device: torch_device) \
            -> Tensor:
        """Forward pass of the attention decoder.

        :param x: Input tensor.
        :type x: torch.Tensor
        :param y: Target values.
        :type y: torch.Tensor
        :param device: Device to use.
        :type device: torch.device
        :return: Output of the attention decoder.
        :rtype: torch.Tensor
        """
        b_size, max_len = y.size()

        y_onehot = zeros(
            b_size, max_len,
            self.nb_classes
        ).to(device)

        y_onehot.scatter_(
            -1,
            y.unsqueeze(-1),
            ones_like(y).float().unsqueeze(-1))

        tf = zeros_like(y_onehot[:, 0, :]).to(device)

        y_onehot = y_onehot[:, 1:, :]
        max_len -= 1

        h_1 = zeros(b_size, self.output_dim).to(device)
        h_2 = zeros(b_size, self.output_dim).to(device)
        h_3 = zeros(b_size, self.output_dim).to(device)
        flags = zeros(b_size, max_len).to(device)

        outputs = zeros(
            b_size, max_len,
            self.nb_classes
        ).to(device)

        h_in = self.dropout(x)

        for t_step in range(max_len):
            prob = self.scheduled_sampling()
            flags.random_(0, 1001).div_(1000).lt_(prob).float()

            att_weights = self._attention(h_in, h_1)
            weighted_h_in = h_in.mul(att_weights).sum(dim=1)

            tf_input = cat([weighted_h_in, tf], dim=-1)
            h_1 = self.rnn(tf_input, h_1)

            _h = self.dropout(cat([h_1, weighted_h_in], dim=-1))
            h_2 = self.rnn_2(_h, h_2)

            _h = self.dropout(cat([h_1 + h_2, weighted_h_in], dim=-1))
            h_3 = self.rnn_2(_h, h_3)

            _h = h_3 + h_2

            cls_out = self.classifier(_h)
            sig_out = cls_out.softmax(dim=-1)

            batch_flags = flags.unsqueeze(-1).expand(
                -1, -1, y_onehot.size()[-1])

            tf = y_onehot[:, t_step, :].mul(batch_flags[:, t_step, :]).add(
                sig_out.mul(batch_flags[:, t_step, :].eq(0).float()))

            outputs[:, t_step, :] = cls_out

        self.iteration += 1

        return outputs

    def _inference_pass(self,
                        x: Tensor,
                        device: torch_device) \
            -> Tensor:
        """Inference  pass of the attention decoder.

        :param x: Input tensor.
        :type x: torch.Tensor
        :param device: Device to use.
        :type device: torch.device
        :return: Output of the attention decoder.
        :rtype: torch.Tensor
        """
        b_size = x.size()[0]

        h_in = self.dropout(x)

        h = zeros(b_size, self.output_dim).to(device)
        tf = zeros(b_size, self.nb_classes).to(device)
        outputs = zeros(b_size, self.max_out_t_steps,
                        self.nb_classes).to(device)

        if len(self.extra_rnn_layers) > 0:
            h_extra = [zeros(b_size, self.output_dim).to(device)
                       for _ in range(len(self.extra_rnn_layers) + 1)]

        for t_step in range(self.max_out_t_steps):
            att_weights = self._attention(h_in, h)
            weighted_h_in = h_in.mul(att_weights).sum(dim=1)

            tf_input = cat([weighted_h_in, tf], dim=-1)

            h = self.rnn(tf_input, h)

            if len(self.extra_rnn_layers) > 0:
                h_extra[0] = h

                for index_h, rnn in enumerate(self.extra_rnn_layers):
                    h_extra[index_h+1] = rnn(
                        self.dropout(h_extra[index_h]),
                        h_extra[index_h+1])

                hh = h_extra[-1]
            else:
                hh = h

            cls_out = self.classifier(hh)

            tf[:, :] = cls_out.softmax(dim=-1)
            outputs[:, t_step, :] = cls_out

        return outputs

# EOF
