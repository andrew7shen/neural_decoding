# Script for the LightningDataModule objects

# Import packages
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(42)


class M1_EMG_Dataset_Toy(pl.LightningDataModule):
    
    def __init__(self, num_samples, num_neurons, num_muscles, batch_size, dataset_type):
        super().__init__()
        self.num_samples = num_samples
        self.num_neurons = num_neurons
        self.num_muscles = num_muscles # THIS is a bad variable name, just temporary until know what M is
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.num_modes = 2
        self.dataset_type = dataset_type
        self.decoder_mode1 = torch.randn(self.num_neurons, self.num_muscles) # temporary just for toy dataset testing
        self.decoder_mode2 = torch.randn(self.num_neurons, self.num_muscles) # temporary just for toy dataset testing

        # Generate toy features and labels
        self.train_dataset = self.generate_toy_dataset(self.num_samples)
        self.val_dataset = self.generate_toy_dataset(self.num_samples*0.2)


    def generate_behavioral(self, num_samples, m1_val, mode):
        """
        Helper function to generate data with behavioral labels for testing ClusterModel sub-model
        
        Input: (int) num_samples: number of samples
            (float) m1_val: value of the features in this mode
            (str) mode: which mode to generate
        Output: ([num_samples, num_neurons] tensor) features: output features
                ([num_samples] tensor) labels: output behavioral labels 
        """
        if mode == "mode1":
            labels = F.one_hot(torch.zeros(num_samples, dtype=int), self.num_modes)
        elif mode == "mode2":
            labels = F.one_hot(torch.ones(num_samples, dtype=int), self.num_modes)

        features = torch.full((num_samples, self.num_neurons), m1_val) + torch.randn(num_samples, self.num_neurons)
            
        return features, labels
    
    
    def generate_emg(self, num_samples, m1_val, decoder):
        """
        Helper function to generate data with EMG labels for testing the full CombinedModel
        
        Input: (int) num_samples: number of samples
            (float) m1_val: value of the features in this mode
        Output: ([num_samples, num_neurons] tensor) features: output features
                ([num_samples, num_muscles] tensor) labels: output EMG labels 
        """

        features = torch.full((num_samples, self.num_neurons), m1_val) + torch.randn(num_samples, self.num_neurons)
        labels = torch.matmul(features, decoder)

        return features, labels


    def generate_toy_dataset(self, num_samples):
        """
        Generates the final toy dataset
        Input: (int) num_samples: number of samples
        Output: ((feature, label) tuple) dataset: output dataset
        """
        
        # Dataset parameters
        num_samples_total = num_samples
        num_samples = int(num_samples_total/2)
        m1_val_mode1 = 10.0
        m1_val_mode2 = 0.0

        # Generate toy dataset with behavioral labels to test out ClusterModel
        if self.dataset_type == "behavioral":
            features_mode1, labels_mode1 = self.generate_behavioral(num_samples, m1_val_mode1, "mode1")
            features_mode2, labels_mode2 = self.generate_behavioral(num_samples, m1_val_mode2, "mode2")
        
        # Generate toy dataset with EMG labels to test out full CombinedModel
        elif self.dataset_type == "emg":
            features_mode1, labels_mode1 = self.generate_emg(num_samples, m1_val_mode1, self.decoder_mode1)
            features_mode2, labels_mode2 = self.generate_emg(num_samples, m1_val_mode2, self.decoder_mode2)
        
        # Format datasets in pairs of (feature, label)
        dataset_mode1 = [(features_mode1[i], labels_mode1[i]) for i in range(len(features_mode1))]
        dataset_mode2 = [(features_mode2[i], labels_mode2[i]) for i in range(len(features_mode2))]
        dataset = dataset_mode1 + dataset_mode2

        return dataset
    

    def __len__(self):
        """
        Returns number of samples in the training set.
        """
        
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
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


