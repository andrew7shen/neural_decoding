# Script that holds TrainingModule code for Lightning module

# Import packages
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from sklearn.metrics import r2_score

class TrainingModule(LightningModule):

    def __init__(self, model, lr, weight_decay, record, type, 
                 temperature, anneal_temperature, num_epochs, end_temperature, lambda_val, l1_lambda_val, overlap_lambda_val):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.record = record
        self.dataset_type = type
        self.training_step_labels = []
        self.training_step_preds = []
        self.val_step_labels = []
        self.val_step_preds = []
        self.temperature = temperature
        self.anneal_temperature = anneal_temperature
        self.end_temperature = end_temperature
        self.num_epochs = num_epochs
        self.max_cluster_probs = []
        self.lambda_val = lambda_val
        self.l1_lambda_val = l1_lambda_val
        self.overlap_lambda_val = overlap_lambda_val


    def forward(self, x):
        return self.model.forward(x, self.temperature)

    
    def training_step(self, batch):

        # Read in batch
        features = batch["m1"]
        labels = batch["emg"]

        # Generate predictions
        # labels_hat = self.model(features, self.temperature)
        # TODO: Return clustering probs and final output
        cluster_probs, labels_hat = self.model(features, self.temperature)
        # TODO: Add fix for batch size of 1
        self.max_cluster_probs += list(torch.max(cluster_probs.squeeze(1),dim=1).values)
        
        if labels_hat.shape[0] == 1:
            pass
            # TODO: Commented out for generalizability grooming experiment because was erroring and unnecessary
            # labels_hat = labels_hat.squeeze(2) # We want to squeeze out last dim but not first dim if batch_size=1 
        else:
            labels_hat = labels_hat.squeeze() 
        if self.dataset_type == "behavioral":
            train_loss = F.cross_entropy(labels_hat, labels)
        elif self.dataset_type == "emg":
            train_loss = F.mse_loss(labels_hat, labels)

        # Add L2 regularization of decoder weights
        l2_norm = sum(p[1].pow(2).sum() for p in self.model.dm.named_parameters() if "weight" in p[0])
        train_loss += self.lambda_val * l2_norm

        # Add L1 regularization of final outputs
        outputs = self.model.cm(batch["m1"], self.temperature) * self.model.dm(batch["m1"])
        l1_norm = sum(p.abs().sum() for p in outputs)
        train_loss += self.l1_lambda_val * l1_norm

        # Add overlap penalty of final outputs
        inner_product = torch.abs(outputs.mT@outputs)
        train_loss += self.overlap_lambda_val * torch.sum(inner_product)

        self.log("train_loss", train_loss, on_step=True)
        self.training_step_labels += labels.tolist()
        self.training_step_preds += labels_hat.tolist()

        return train_loss
    

    def validation_step(self, batch):

        # Read in batch
        features = batch["m1"]
        labels = batch["emg"]

        # Generate predictions
        # labels_hat = self.model(features, self.temperature)
        # TODO: Return clustering probs and final output
        cluster_probs, labels_hat = self.model(features, self.temperature)

        if labels_hat.shape[0] == 1:
            pass
            # TODO: Commented out for generalizability grooming experiment because was erroring and unnecessary
            # labels_hat = labels_hat.squeeze(2) # We want to squeeze out last dim but not first dim if batch_size=1 
        else:
            labels_hat = labels_hat.squeeze() 
        if self.dataset_type == "behavioral":
            val_loss = F.cross_entropy(labels_hat, labels)
        elif self.dataset_type == "emg":
            val_loss = F.mse_loss(labels_hat, labels)
        self.log("val_loss", val_loss, on_epoch=True)
        self.val_step_labels += labels.tolist()
        self.val_step_preds += labels_hat.tolist()

        return val_loss
    

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # TODO: clarify l2 vs weight_decay
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
    

class Callback(pl.Callback):
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.epoch_number = 0
        self.initial_temp = None


    def on_train_epoch_end(self, trainer, pl_module):

        # Set initial temperature
        if self.epoch_number == 0:
            self.initial_temp = pl_module.temperature
        
        # Calculate loss over the epoch
        labels = torch.Tensor(pl_module.training_step_labels)
        labels_hat = torch.Tensor(pl_module.training_step_preds)
        if self.dataset.dataset_type == "behavioral":
            train_loss_epoch = F.cross_entropy(labels_hat, labels)
        elif self.dataset.dataset_type == "emg":
            train_loss_epoch = F.mse_loss(labels_hat, labels)
        # print(train_loss_epoch.item())

        # Calculate R^2 metric
        train_r2 = r2_score(labels.numpy(), labels_hat.numpy())
        pl_module.log("train_r2", train_r2)

        # Reset stored labels and preds for current epoch
        pl_module.training_step_labels = []
        pl_module.training_step_preds = []
        self.epoch_number += 1

        # Calculate discreteness metric
        limit = 0.9
        discreteness_metric = sum(max_val > limit for max_val in pl_module.max_cluster_probs)/len(pl_module.max_cluster_probs)
        pl_module.log("discreteness", discreteness_metric.item())
        pl_module.max_cluster_probs = []

        # Anneal temperature
        pl_module.log("temperature", pl_module.temperature)
        if pl_module.anneal_temperature == "linear":
            pl_module.temperature = self.linearTemp(self.epoch_number, self.initial_temp)
        elif pl_module.anneal_temperature == "cosine":
            pl_module.temperature = self.cosineTemp(self.epoch_number, self.initial_temp, pl_module.end_temperature)
        elif pl_module.anneal_temperature == "cosine_flatten":
            pl_module.temperature = self.cosineFlattenTemp(self.epoch_number, self.initial_temp, pl_module.end_temperature)
        elif pl_module.anneal_temperature == "sine":
            pl_module.temperature = self.sineTemp(self.epoch_number, self.initial_temp, pl_module.end_temperature)
        elif pl_module.anneal_temperature == "sine_flatten":
            pl_module.temperature = self.sineFlattenTemp(self.epoch_number, self.initial_temp, pl_module.end_temperature)
        elif pl_module.anneal_temperature == "inverse":
            print("ERROR: choose valid annealing parameter")
            exit()
        elif pl_module.anneal_temperature == "none":
            pass
        else:
            print("ERROR: choose valid annealing parameter")
            exit()
        # print(pl_module.temperature)

    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Calculate loss over the epoch
        labels = torch.Tensor(pl_module.val_step_labels)
        labels_hat = torch.Tensor(pl_module.val_step_preds)
        if self.dataset.dataset_type == "behavioral":
            val_loss_epoch = F.cross_entropy(labels_hat, labels)
        elif self.dataset.dataset_type == "emg":
            val_loss_epoch = F.mse_loss(labels_hat, labels)

        # Calculate R^2 metric
        val_r2 = r2_score(labels.numpy(), labels_hat.numpy())
        pl_module.log("val_r2", val_r2)

        # Reset stored labels and preds for current epoch
        pl_module.val_step_labels = []
        pl_module.val_step_preds = []


    def linearTemp(self, epoch, initial_temp):
        end_temp = 0.01
        # end_temp = 0.001
        num_epochs = 500

        # Anneal temperature at linear rate
        curr_temp = initial_temp - epoch*(initial_temp-end_temp)/(num_epochs)

        return curr_temp
    
    
    def cosineTemp(self, epoch, initial_temp, end_temp):
        # end_temp = 0.01
        # end_temp = 0.001
        num_epochs = 500

        # TODO: temp for continued training
        # epoch = epoch + 473

        # Anneal temperature at cosine rate
        # From Pytorch documentation for CosineAnnealingLR (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
        
        curr_temp = end_temp+0.5*(initial_temp-end_temp)*(1+math.cos(epoch*math.pi/num_epochs))

        return curr_temp
    

    def cosineFlattenTemp(self, epoch, initial_temp, end_temp):
        # end_temp = 0.01
        # end_temp = 0.001
        num_epochs = 500

        # Anneal temperature at cosine rate
        # From Pytorch documentation for CosineAnnealingLR (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
        if epoch < num_epochs:
            curr_temp = end_temp+0.5*(initial_temp-end_temp)*(1+math.cos(epoch*math.pi/num_epochs))
        else:
            curr_temp = end_temp

        return curr_temp
    

    def sineTemp(self, epoch, initial_temp, end_temp):
        # end_temp = 0.01
        # end_temp = 0.001
        num_epochs = 500

        # TODO: Trying to re-derive cosine annealer from scratch
        # trig_term = 0.5*(math.cos(math.pi*epoch/num_epochs)+1)
        # curr_temp = end_temp+trig_term*(initial_temp-end_temp)

        # TODO: Can try sin annealer
        trig_term = -math.sin(0.5*math.pi*epoch/num_epochs)+1
        curr_temp = end_temp+trig_term*(initial_temp-end_temp)

        # if epoch > 450:
        #     print(curr_temp)
        # import pdb; pdb.set_trace()
        return curr_temp
    
    def sineFlattenTemp(self, epoch, initial_temp, end_temp):
        # end_temp = 0.01
        # end_temp = 0.001
        num_epochs = 500

        # Anneal temperature at sine rate
        if epoch < num_epochs:
            trig_term = -math.sin(0.5*math.pi*epoch/num_epochs)+1
            curr_temp = end_temp+trig_term*(initial_temp-end_temp)

        else:
            curr_temp = end_temp

        return curr_temp
    
