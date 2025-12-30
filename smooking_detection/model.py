import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from omegaconf import DictConfig, OmegaConf

from models.conv_net import ConvNet
from models.vit_model import ViTModel


class SmookingBinaryClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_str = getattr(
            cfg.model, "vit_model_name", "WinKawaks/vit-tiny-patch16-224"
        )
        id2label = getattr(cfg.model, "id2label", {0: "not_smoking", 1: "smoking"})
        label2id = getattr(cfg.model, "label2id", {"not_smoking": 0, "smoking": 1})
        if isinstance(id2label, DictConfig):
            id2label = OmegaConf.to_container(id2label, resolve=True)

        if isinstance(label2id, DictConfig):
            label2id = OmegaConf.to_container(label2id, resolve=True)

        if cfg.model.model_name == "conv_net":
            self.model = ConvNet()
        elif "vit" in cfg.model.model_name.lower():
            self.model = ViTModel(
                model_name=model_str, id2label=id2label, label2id=label2id
            )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_config = cfg.optimizer

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_f1 = F1Score(task="binary", num_classes=2, average="macro")
        self.val_f1 = F1Score(task="binary", num_classes=2, average="macro")
        self.test_f1 = F1Score(task="binary", num_classes=2, average="macro")

        self.save_hyperparameters()

    def _get_predictions(self, logits):
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = self._get_predictions(logits)
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = self._get_predictions(logits)
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = self._get_predictions(logits)

        self.test_acc(preds, y)
        self.test_f1(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

        return loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
