# Script for the LightningDataModule objects

# Import packages
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class Cage_Dataset(pl.LightningDataModule):
    
    def __init__(self, m1_path, emg_path, behavioral_path, num_modes, batch_size, dataset_type, seed, kmeans_cluster):
        super().__init__()
        self.batch_size = batch_size
        self.num_modes = num_modes
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_type = dataset_type
        self.N = None
        self.M = None
        self.seed = seed
        self.kmeans_cluster = kmeans_cluster

        # Set manual seed
        torch.manual_seed(seed)

        # TODO: temp code to train single linear model
        # mode_name = ["crawl", "precision", "power"]
        # for mode in mode_name:
        #     curr_m1_path = "%s%s.npy" % (m1_path[:-9], mode)
        #     curr_emg_path = "%s%s.npy" % (emg_path[:-9], mode)
        #     curr_behavioral_path = "%s%s.npy" % (behavioral_path[:-9], mode)
        #     m1 = torch.Tensor(np.load(curr_m1_path))
        #     emg = torch.Tensor(np.load(curr_emg_path))
        #     behavioral = np.load(curr_behavioral_path)
        #     labels = [[emg[i], behavioral[i]] for i in range(len(emg))]
        #     X_train, X_val, y_train, y_val = train_test_split(m1, labels, test_size=0.2, random_state=42)

        #     # Reformat back into timestamps
        #     X_train = torch.reshape(X_train, (X_train.size()[0]*X_train.size()[1], X_train.size()[2]))
        #     X_val = torch.reshape(X_val, (X_val.size()[0]*X_val.size()[1], X_val.size()[2]))
        #     y_train_emg = torch.stack([v[0] for v in y_train])
        #     y_train_emg = torch.reshape(y_train_emg, (y_train_emg.size()[0]*y_train_emg.size()[1], y_train_emg.size()[2]))
        #     y_train_behavioral = np.stack([v[1] for v in y_train])
        #     y_train_behavioral = np.reshape(y_train_behavioral, (y_train_behavioral.shape[0]*y_train_behavioral.shape[1]))
        #     y_val_emg = torch.stack([v[0] for v in y_val])
        #     y_val_emg = torch.reshape(y_val_emg, (y_val_emg.size()[0]*y_val_emg.size()[1], y_val_emg.size()[2]))
        #     y_val_behavioral = np.stack([v[1] for v in y_val])
        #     y_val_behavioral = np.reshape(y_val_behavioral, (y_val_behavioral.shape[0]*y_val_behavioral.shape[1]))

        #     if mode == "crawl":
        #         self.train_dataset = [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
        #         self.val_dataset = [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]
        #         self.N = m1.size()[2]
        #         self.M = emg.size()[2]
        #     else:
        #         self.train_dataset = self.train_dataset + [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
        #         self.val_dataset = self.val_dataset + [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]

        # If using kmeans split data for sanity check
        if "data/set2_data/kmeans_split" in m1_path:
            path = m1_path
            m1_train = torch.Tensor(np.load("%s/m1_train_%s.npy" % (path, self.kmeans_cluster)))
            emg_train = torch.Tensor(np.load("%s/emg_train_%s.npy" % (path, self.kmeans_cluster)))
            labels_train = np.load("%s/labels_train_%s.npy" % (path, self.kmeans_cluster))
            m1_val = torch.Tensor(np.load("%s/m1_val_%s.npy" % (path, self.kmeans_cluster)))
            emg_val = torch.Tensor(np.load("%s/emg_val_%s.npy" % (path, self.kmeans_cluster)))
            labels_val = np.load("%s/labels_val_%s.npy" % (path, self.kmeans_cluster))

            self.train_dataset = [(m1_train[i], emg_train[i], labels_train[i]) for i in range(len(m1_train))]
            self.val_dataset = [(m1_val[i], emg_val[i], labels_val[i]) for i in range(len(m1_val))]
            self.N = m1_train.shape[1]
            self.M = emg_train.shape[1]

        else:
            # Read in data
            m1 = torch.Tensor(np.load(m1_path))
            emg = torch.Tensor(np.load(emg_path))
            behavioral = np.load(behavioral_path)
            labels = [[emg[i], behavioral[i]] for i in range(len(emg))]
            X_train, X_val, y_train, y_val = train_test_split(m1, labels, test_size=0.2, random_state=42)

            # Reformat back into timestamps
            X_train = torch.reshape(X_train, (X_train.size()[0]*X_train.size()[1], X_train.size()[2]))
            X_val = torch.reshape(X_val, (X_val.size()[0]*X_val.size()[1], X_val.size()[2]))
            y_train_emg = torch.stack([v[0] for v in y_train])
            y_train_emg = torch.reshape(y_train_emg, (y_train_emg.size()[0]*y_train_emg.size()[1], y_train_emg.size()[2]))
            y_train_behavioral = np.stack([v[1] for v in y_train])
            y_train_behavioral = np.reshape(y_train_behavioral, (y_train_behavioral.shape[0]*y_train_behavioral.shape[1]))
            y_val_emg = torch.stack([v[0] for v in y_val])
            y_val_emg = torch.reshape(y_val_emg, (y_val_emg.size()[0]*y_val_emg.size()[1], y_val_emg.size()[2]))
            y_val_behavioral = np.stack([v[1] for v in y_val])
            y_val_behavioral = np.reshape(y_val_behavioral, (y_val_behavioral.shape[0]*y_val_behavioral.shape[1]))

            self.train_dataset = [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
            self.val_dataset = [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]
            self.N = m1.size()[2]
            self.M = emg.size()[2]
            

    def __len__(self):
        """
        Returns number of samples in the training set.
        """
        
        return len(self.train_dataset) + len(self.val_dataset)


    def __getitem__(self, index):

        # Not used for the training
        return self.train_dataset[index]


    def collate_fn(self, batch):
        
        final_batch = {}
        X = []  # m1
        Y1 = []  # emg
        Y2 = []  # behavioral
        for sample in batch:
            X.append(sample[0])
            Y1.append(sample[1])
            Y2.append(sample[2])
        final_batch["m1"] = torch.stack(X)
        final_batch["emg"] = torch.stack(Y1).float()
        final_batch["behavioral"] = Y2

        return final_batch


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)



class M1_EMG_Dataset_Toy(pl.LightningDataModule):
    
    def __init__(self, num_samples, num_neurons, num_muscles, num_modes, batch_size, dataset_type):
        super().__init__()
        self.num_samples = num_samples
        self.num_neurons = num_neurons
        self.num_muscles = num_muscles # THIS is a bad variable name, just temporary until know what M is
        self.num_modes = num_modes
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
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
        final_batch["m1"] = torch.stack(X)
        final_batch["emg"] = torch.stack(Y).float()

        return final_batch


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


