# Script for the LightningDataModule objects

# Import packages
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cage_data
import pickle as pickle

class Cage_Dataset(pl.LightningDataModule):
    
    def __init__(self, m1_path, emg_path, behavioral_path, 
                 num_modes, batch_size, dataset_type, 
                 seed, kmeans_cluster, label_type,
                 remove_zeros, scale_outputs):
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
        self.label_type = label_type
        self.remove_zeros = remove_zeros
        self.scale_outputs = scale_outputs

        # Set manual seed
        torch.manual_seed(seed)

        # Load data with Set1 labels
        if "set1" in self.label_type:
            # If loading data for just one label
            if len(self.label_type) > 4:
                curr_label = self.label_type[5:]
                self.format_set1_data(curr_label)
            # If loading data for all labels
            else:
                self.format_set1_data("all")

        # Load data with Set2 labels
        elif self.label_type == "set2":
            # TODO: Refactor to format input data all within data.py

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

                # Perform min-max scaling
                if self.scale_outputs:
                    plot_variance = False
                    scaler = MinMaxScaler()
                    scaler.fit(y_train_emg)
                    if plot_variance:
                        y_train_emg_unscaled = y_train_emg
                    y_train_emg = torch.Tensor(scaler.transform(y_train_emg))
                    y_val_emg = torch.Tensor(scaler.transform(y_val_emg))
                    # Create plots for variance in EMG data before and after scaling
                    if plot_variance:
                        save_fig = True
                        unscaled_var_vals = torch.var(y_train_emg_unscaled, dim=0)
                        scaled_var_vals = torch.var(y_train_emg, dim=0)
                        fig, ax = plt.subplots(1,2, figsize=(10,5))
                        # fig.tight_layout()
                        fig.suptitle("Unscaled vs Scaled Variance in EMG")
                        ax[0].plot(unscaled_var_vals, label="unscaled")
                        ax[0].legend()
                        ax[0].set_xlabel("Muscles")
                        ax[0].set_ylabel("Variance")
                        ax[1].plot(scaled_var_vals, label="scaled")
                        ax[1].legend()
                        ax[1].set_xlabel("Muscles")
                        ax[1].set_ylabel("Variance")
                        fig.tight_layout()
                        if save_fig:
                            curr_dir = os.getcwd()
                            plt.savefig("%s/figures/misc/scaled_var_diff.png" % curr_dir)
                        else:
                            plt.show()

                self.train_dataset = [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
                self.val_dataset = [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]
                self.N = m1.size()[2]
                self.M = emg.size()[2]
    
        # Load data with full dataset without labels
        elif self.label_type == "none":

            # If using kmeans split data for sanity check
            if "data/none_data/kmeans_split" in m1_path:
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
                self.format_none_data()

        # Remove samples with M1 data of all zeros
        if self.remove_zeros:
            self.remove_zeros_from_dataset()

    def format_set1_data(self, labels_to_use):

        np.set_printoptions(suppress=True)

        def find_start_end(N, timeframe):
            segment_range = np.where((timeframe>=my_cage_data.behave_tags['start_time'][N]) & (timeframe<=my_cage_data.behave_tags['end_time'][N]))[0]
            start_idx = segment_range[0]
            end_idx = segment_range[-1]
            return start_idx, end_idx

        # Load in raw dataset files
        curr_dir = os.getcwd()
        data_path = "%s/data/pickle_files/" % curr_dir
        file_name = 'Pop_20210709_Cage_004.pkl'
        with open(data_path+file_name, 'rb') as fp:
            my_cage_data = pickle.load(fp)
        m1 = np.transpose(my_cage_data.binned['spikes'])
        emg = np.transpose(my_cage_data.binned['filtered_EMG'])
        timeframe = my_cage_data.binned['timeframe']

        # Format into trials
        m1_trials = []
        emg_trials = []
        behavior_trials = []
        for N in range(len(my_cage_data.behave_tags['tag'])):
            curr_behavior = my_cage_data.behave_tags['tag'][N]
            if curr_behavior == "grooming":
                    continue
            start_idx, end_idx = find_start_end(N, timeframe)
            m1_trials.append(m1[start_idx:end_idx+1])
            emg_trials.append(emg[start_idx:end_idx+1])
            behavior_trials.append([curr_behavior]*(end_idx-start_idx+1))
            
        # Create train/val splits
        labels_trials = [[emg_trials[i], behavior_trials[i]] for i in range(len(emg_trials))]
        X_train, X_val, y_train, y_val = train_test_split(m1_trials, labels_trials, test_size=0.2, random_state=42)

        # If only loading data with one label
        if labels_to_use != "all":
            # For each dataset split of train and val
            for split in ["train", "val"]:
                new_X = []
                new_y = []
                if split == "train":
                    curr_X = X_train
                    curr_y = y_train
                elif split == "val":
                    curr_X = X_val
                    curr_y = y_val
                # For each trial in dataset
                for i in range(len(curr_y)):
                    curr_trial_X = curr_X[i]
                    curr_trial_y = curr_y[i]
                    curr_label = curr_trial_y[1][0]
                    if curr_label == labels_to_use:
                        new_X.append(curr_trial_X)
                        new_y.append(curr_trial_y)
                # Save new dataset with only data from one label
                if split == "train":
                    X_train = new_X
                    y_train = new_y
                elif split == "val":
                    X_val = new_X
                    y_val = new_y

        # Format back into time stamps
        X_train = torch.Tensor(np.concatenate(X_train))
        X_val = torch.Tensor(np.concatenate(X_val))
        y_train_emg = torch.Tensor(np.concatenate([y[0] for y in y_train]))
        y_train_behavioral = np.concatenate([y[1] for y in y_train])
        y_val_emg = torch.Tensor(np.concatenate([y[0] for y in y_val]))
        y_val_behavioral = np.concatenate([y[1] for y in y_val])
        self.train_dataset = [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
        self.val_dataset = [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]
        self.N = m1.shape[1]
        self.M = emg.shape[1]

    def format_none_data(self):

        np.set_printoptions(suppress=True)

        def find_start_end(N, timeframe):
            segment_range = np.where((timeframe>=my_cage_data.behave_tags['start_time'][N]) & (timeframe<=my_cage_data.behave_tags['end_time'][N]))[0]
            start_idx = segment_range[0]
            end_idx = segment_range[-1]
            return start_idx, end_idx

        # Load in raw dataset files
        curr_dir = os.getcwd()
        data_path = "%s/data/pickle_files/" % curr_dir
        file_name = 'Pop_20210709_Cage_004.pkl'
        with open(data_path+file_name, 'rb') as fp:
            my_cage_data = pickle.load(fp)
        m1 = np.transpose(my_cage_data.binned['spikes'])
        emg = np.transpose(my_cage_data.binned['filtered_EMG'])
        timeframe = my_cage_data.binned['timeframe']

        # Format labels for labeled and unlabeled timestamps
        behavior = ["none"] * m1.shape[0]
        for N in range(len(my_cage_data.behave_tags['tag'])):
            curr_behavior = my_cage_data.behave_tags['tag'][N]
            if curr_behavior == "grooming":
                    continue
            start_idx, end_idx = find_start_end(N, timeframe)
            behavior[start_idx:end_idx+1] = [curr_behavior] * (end_idx+1-start_idx)

        # Format into trials of length 100
        m1_trials = []
        emg_trials = []
        behavior_trials = []
        for i in range(0, m1.shape[0], 100):
            m1_trials.append(m1[i:i+100])
            emg_trials.append(emg[i:i+100])
            behavior_trials.append(behavior[i:i+100])

        # Create train/val splits
        labels_trials = [[emg_trials[i], behavior_trials[i]] for i in range(len(emg_trials))]
        X_train, X_val, y_train, y_val = train_test_split(m1_trials, labels_trials, test_size=0.2, random_state=42)

        # Format back into time stamps
        X_train = torch.Tensor(np.concatenate(X_train))
        X_val = torch.Tensor(np.concatenate(X_val))
        y_train_emg = torch.Tensor(np.concatenate([y[0] for y in y_train]))
        y_train_behavioral = np.concatenate([y[1] for y in y_train])
        y_val_emg = torch.Tensor(np.concatenate([y[0] for y in y_val]))
        y_val_behavioral = np.concatenate([y[1] for y in y_val])

        # Perform min-max scaling
        if self.scale_outputs:
            scaler = MinMaxScaler()
            scaler.fit(y_train_emg)
            y_train_emg = torch.Tensor(scaler.transform(y_train_emg))
            y_val_emg = torch.Tensor(scaler.transform(y_val_emg))

        self.train_dataset = [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
        self.val_dataset = [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]
        self.N = m1.shape[1]
        self.M = emg.shape[1]
        
    def remove_zeros_from_dataset(self):
        train_nozeros = []
        val_nozeros = []
        for dataset in [self.train_dataset, self.val_dataset]:
            num_zeros = 0
            zeros_tensor = torch.Tensor([0.0]*95)
            for i in range(len(dataset)):
                sample = dataset[i]
                curr_m1 = sample[0]
                if torch.equal(curr_m1, zeros_tensor):
                    num_zeros += 1
                else:
                    # Keep only samples without all zeros
                    if dataset == self.train_dataset:
                        train_nozeros.append(sample)
                    elif dataset == self.val_dataset:
                        val_nozeros.append(sample)
            # if dataset == self.train_dataset:
            #     print(f"{num_zeros} zero samples out of {len(dataset)} in train set")
            # elif dataset == self.val_dataset:
            #     print(f"{num_zeros} zero samples out of {len(dataset)} in val set")
        self.train_dataset = train_nozeros
        self.val_dataset = val_nozeros

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


