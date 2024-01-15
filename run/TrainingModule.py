# Script that holds TrainingModule code for Lightning module

# Import packages
from pytorch_lightning import LightningModule
import torch.optim as optim
import torch.nn.functional as F
import wandb

class TrainingModule(LightningModule):

    def __init__(self, model, lr, record):
        super().__init__()
        self.model = model
        self.lr = lr
        self.record = record
    
    def training_step(self, batch):
        # Read in batch
        features = batch["features"]
        labels = batch["labels"]

        # Generate predictions
        label_hat = self.model(features).squeeze()
        # train_loss = F.mse_loss(label_hat, labels)
        train_loss = F.binary_cross_entropy(label_hat, labels)
        self.log("train_loss", train_loss, on_step=True)

        return train_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer