# Script for the LightningDataModule objects

# Import packages
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class M1Dataset_Toy(pl.LightningDataModule):
    
    def __init__(self, num_samples, num_neurons, batch_size):
        super().__init__()
        self.num_samples = num_samples
        self.num_neurons = num_neurons
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.num_modes = 2

        # Generate toy features and labels
        self.train_dataset = self.generate_toy_dataset(self.num_samples, self.num_neurons)
        self.val_dataset = self.generate_toy_dataset(self.num_samples*0.2, self.num_neurons)

    def generate_helper(self, num_samples, num_neurons, rate, mode):
        """
        Helper function to generates data for a specific mode
        
        Input: (int) num_samples: number of samples
            (int) num_neurons: number of neurons
            (int) rate: value of the features in this mode
            (str) mode: which mode to generate
        Output: ([num_samples, num_neurons] tensor) features: output features
                ([num_samples] tensor) labels: output behavioral labels 
        """
        if mode == "mode1":
            labels = F.one_hot(torch.zeros(num_samples, dtype=int), self.num_modes)
        elif mode == "mode2":
            labels = F.one_hot(torch.ones(num_samples, dtype=int), self.num_modes)

        features = torch.full((num_samples, num_neurons), rate)
            
        return features, labels


    def generate_toy_dataset(self, num_samples, num_neurons):
        """
        Generates the final toy dataset
        Input: (int) num_samples: number of samples
               (int) num_neurons: number of neurons
        Output: ((feature, label) tuple) dataset: output dataset
        """

        # Generate datasets
        num_samples_total = num_samples
        num_samples = int(num_samples_total/2)
        rate_mode1 = 10.0
        rate_mode2 = 0.0
        features_mode1, labels_mode1 = self.generate_helper(num_samples, num_neurons, rate=rate_mode1, mode="mode1")
        features_mode2, labels_mode2 = self.generate_helper(num_samples, num_neurons, rate=rate_mode2, mode="mode2")
        
        # Format datasets in pairs of (feature, label)
        dataset_mode1 = [(features_mode1[i], labels_mode1[i]) for i in range(len(features_mode1))]
        dataset_mode2 = [(features_mode2[i], labels_mode2[i]) for i in range(len(features_mode2))]
        dataset = dataset_mode1 + dataset_mode2

        return dataset
    

    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):

        # Not used for the training
        return self.train_dataset[index]


    def collate_fn(self, batch):
        
        final_batch = {}
        X = []
        Y = []
        for sample in batch:
            X.append(sample[0])
            Y.append(sample[1])
        final_batch["features"] = torch.stack(X)
        final_batch["labels"] = torch.stack(Y).float()

        return final_batch


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)


