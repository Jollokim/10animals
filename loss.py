import torch.nn as nn
import torch.nn.functional as F
import torch


class InceptionLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_aux1, y_aux2, y_hat, y):
        """
            Hey sucka!
        """
        aux1_loss = F.cross_entropy(y_aux1, y)
        aux2_loss = F.cross_entropy(y_aux2, y)
        y_hat_loss = F.cross_entropy(y_hat, y)

        loss = (aux1_loss * 0.3) + (aux2_loss * 0.3) + (y_hat_loss * 0.4)

        return loss