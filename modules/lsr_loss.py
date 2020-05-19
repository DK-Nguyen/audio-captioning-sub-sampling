#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch import zeros, Tensor
from torch.nn import Module, KLDivLoss

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['LabelSmoothingRegularization']


class LabelSmoothingRegularization(Module):

    def __init__(self,
                 nb_classes: int,
                 eps: float,
                 reduction: Optional[str] = 'batchmean') \
            -> None:
        super().__init__()

        self.smoothing_factor = 1 - eps
        self.nb_classes = nb_classes
        self.loss_f = KLDivLoss(reduction=reduction)

    def forward(self,
                y_hat: Tensor,
                y: Tensor) \
            -> Tensor:
        """LSR loss.

        :param y_hat: Predictions.
        :type y_hat: torch.Tensor
        :param y: Ground truth.
        :type y: torch.Tensor
        :return: KL loss with smooth labelling.
        :rtype: torch.Tensor
        """
        b_size = y.size()[0]
        device = y.device

        y_onehot = zeros(
            b_size,
            self.nb_classes
        ).to(device)

        y_onehot = y_onehot.scatter(-1, y.unsqueeze(-1), 1).float()
        y_onehot = (y_onehot * self.smoothing_factor) + \
                   (self.smoothing_factor/self.nb_classes)

        return self.loss_f(y_hat.log_softmax(dim=-1), y_onehot)

# EOF
