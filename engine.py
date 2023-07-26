from typing import Any
import pytorch_lightning as pl
from model import Inception
import torch

from loss import InceptionLoss

from torchmetrics import Metric

from torch import Tensor


class LightningInception(pl.LightningModule):
    def __init__(self, n_classes, batch_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = Inception(3, n_classes)
        self.loss = InceptionLoss()

        self.batch_size = batch_size

        self.validation_acc = ClassificationAccuracy()
        self.test_acc = ClassificationAccuracy()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-5)
        return opt

    def training_step(self, batch, batch_idx):
        X, y, _ = batch

        aux1, aux2, y_hat = self.model(X)

        loss = self.loss(aux1, aux2, y_hat, y)

        self.log("train_loss", loss, batch_size=self.batch_size, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, _ = batch

        aux1, aux2, y_hat = self.model(X)

        acc = self.validation_acc(y_hat, y)

        self.log("validation_acc", acc, batch_size=self.batch_size, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        X, y, _ = batch

        aux1, aux2, y_hat = self.model(X)

        acc = self.validation_acc(y_hat, y)

        self.log("test_acc", acc, batch_size=self.batch_size, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)



class ClassificationAccuracy(Metric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape

        preds_argmax = torch.argmax(preds, dim=1)
        target_argmax = torch.argmax(target, dim=1)

        self.correct += torch.sum(preds_argmax == target_argmax)
        self.total += torch.tensor(target.shape[0])

    def compute(self):
        return self.correct / self.total


def dummy_forward(model: pl.LightningModule, X=None):
    """
        When using pytorch lazy modules, use this function to initialize the weights.
    """
    if X is None:
        X = torch.randn((1, 3, 200, 200))

    model.freeze()
    model(X)
    model.unfreeze()
