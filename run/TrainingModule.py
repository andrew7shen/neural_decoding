# Script that holds TrainingModule code for Lightning module

# Import packages
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score

class TrainingModule(LightningModule):

    def __init__(self, model, lr, record, type):
        super().__init__()
        self.model = model
        self.lr = lr
        self.record = record
        self.dataset_type = type
        self.training_step_labels = []
        self.training_step_preds = []
        self.val_step_labels = []
        self.val_step_preds = []

    def forward(self, x):
        return self.model.forward(x)

    
    def training_step(self, batch):

        # Read in batch
        features = batch["m1"]
        labels = batch["emg"]

        # Generate predictions
        labels_hat = self.model(features).squeeze()
        if self.dataset_type == "behavioral":
            train_loss = F.cross_entropy(labels_hat, labels)
        elif self.dataset_type == "emg":
            train_loss = F.mse_loss(labels_hat, labels)
        self.log("train_loss", train_loss, on_step=True)
        self.training_step_labels += labels.tolist()
        self.training_step_preds += labels_hat.tolist()

        return train_loss
    

    def validation_step(self, batch):

        # Read in batch
        features = batch["m1"]
        labels = batch["emg"]

        # Generate predictions
        labels_hat = self.model(features).squeeze()
        if self.dataset_type == "behavioral":
            val_loss = F.cross_entropy(labels_hat, labels)
        elif self.dataset_type == "emg":
            val_loss = F.mse_loss(labels_hat, labels)
        self.log("val_loss", val_loss, on_epoch=True)
        self.val_step_labels += labels.tolist()
        self.val_step_preds += labels_hat.tolist()

        return val_loss
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    

class Callback(pl.Callback):
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.epoch_number = 0


    def on_train_epoch_end(self, trainer, pl_module):
        
        # Calculate loss over the epoch
        labels = torch.Tensor(pl_module.training_step_labels)
        labels_hat = torch.Tensor(pl_module.training_step_preds)
        if self.dataset.dataset_type == "behavioral":
            train_loss_epoch = F.cross_entropy(labels_hat, labels)
        elif self.dataset.dataset_type == "emg":
            train_loss_epoch = F.mse_loss(labels_hat, labels)

        # Calculate R^2 metric
        train_r2 = r2_score(labels, labels_hat)
        pl_module.log("train_r2", train_r2)

        # Reset stored labels and preds for current epoch
        pl_module.training_step_labels = []
        pl_module.training_step_preds = []
        self.epoch_number += 1
        

    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Calculate loss over the epoch
        labels = torch.Tensor(pl_module.val_step_labels)
        labels_hat = torch.Tensor(pl_module.val_step_preds)
        if self.dataset.dataset_type == "behavioral":
            val_loss_epoch = F.cross_entropy(labels_hat, labels)
        elif self.dataset.dataset_type == "emg":
            val_loss_epoch = F.mse_loss(labels_hat, labels)

        # Calculate R^2 metric
        val_r2 = r2_score(labels, labels_hat)
        pl_module.log("val_r2", val_r2)

        # Reset stored labels and preds for current epoch
        pl_module.val_step_labels = []
        pl_module.val_step_preds = []
