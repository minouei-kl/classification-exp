import torch
import datetime
import json
from copy import deepcopy
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, MaxMetric, MeanMetric, MetricCollection
import math
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch, verbose=False)


class LightModel(pl.LightningModule):
    def __init__(self, args, model, loss_fn):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

        self.history = {}
        self.num_class = args.num_class
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.num_class)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_class)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_class)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.loss_fn = loss_fn
        # self.training_step_outputs = []
        # self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds.detach(), y.detach()

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.shared_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train_loss", self.train_loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True,
                 on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.shared_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val_loss", self.val_loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=True,
                 on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.shared_step(batch)
        self.test_step_outputs.append(
            {'loss': loss, 'preds': preds, 'targets': targets})
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=True,
                 on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])
        else:
            return self(batch)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.print(self.trainer.callback_metrics)
        self.print_bar()
        epoch = self.trainer.current_epoch
        self.history[epoch] = str(self.trainer.callback_metrics)

    # def on_training_epoch_end(self):
    #     self.log("train_acc", self.train_acc.compute(),
    #              on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        self.plotcm(self.test_step_outputs)

    def plotcm(self, outputs):

        y_pred = torch.cat([output['preds'].argmax(axis=1)
                            for output in outputs]).cpu()
        y_true = torch.cat([output['targets']for output in outputs]).cpu()

        _fig, _ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred, normalize=True, ax=_ax)
        plt.savefig('confusion_matrix.png', format='png')
        report = classification_report(y_true, y_pred, output_dict=True)
        self.print(json.dumps(report, indent=4))
        matrix = confusion_matrix(y_true, y_pred)
        class_acc = matrix.diagonal()/matrix.sum(axis=1)
        self.print(class_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup_steps = 2 * (
            self.trainer.estimated_stepping_batches
            // self.trainer.max_epochs  # type:ignore
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s" % nowtime)
