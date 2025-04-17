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
import h5py
from tqdm import tqdm

class Simulated_Dataset(pl.LightningDataModule):

    def __init__(self, T, N, M, m1_path, emg_path, behavioral_path, 
                    num_modes, batch_size, dataset_type, 
                    seed, kmeans_cluster, label_type,
                    remove_zeros, scale_outputs, mean_centering):
        super().__init__()

        # Assign class variables
        self.batch_size = batch_size
        self.num_modes = num_modes
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_type = dataset_type
        self.T = T
        self.N = N
        self.M = M
        self.seed = seed
        self.kmeans_cluster = kmeans_cluster
        self.label_type = label_type
        self.remove_zeros = remove_zeros
        self.scale_outputs = scale_outputs
        self.mean_centering = mean_centering

        # Set manual seed
        torch.manual_seed(seed)
        
        # Generate simulated dataset
        dataset = self.generate_simulated_dataset()
        timestamps = dataset[1]
        m1 = np.stack([dataset[1], dataset[2]]).T
        emg = dataset[3]

        # Create train/val splits
        X_train, X_val, y_train, y_val = train_test_split(m1, emg, test_size=0.2, random_state=42)
        
        # Perform mean-centering
        # Mean center M1 data
        if "m1" in self.mean_centering:
            train_m1_mean = X_train.mean(axis=0)
            X_train = X_train - train_m1_mean
            X_val = X_val - train_m1_mean
        # Mean center EMG data
        if "emg" in self.mean_centering:
            y_train_mean = y_train.mean(axis=0)
            y_train = y_train - y_train_mean
            y_val = y_val - y_train_mean

        # Final datasets
        self.train_dataset = [(torch.Tensor(X_train[i]), torch.Tensor([y_train[i]]), "no_label") for i in range(len(X_train))]
        self.val_dataset = [(torch.Tensor(X_val[i]), torch.Tensor([y_val[i]]), "no_label") for i in range(len(X_val))]

    
    def generate_simulated_dataset(self):

        # Parameters to adjust
        t_single = np.linspace(0, 2 * np.pi, 300) # number of timepoints in each section
        freq1 = 5
        freq2 = 15
        num_trials = 5
        # Trial-specific mean offsets and magnitudes
        input_means = [100, 90, 50, 10, 0]
        input_magnitudes = [10, 10, 10, 10, 10] # josh said 5-10 relative to 100?
        # Input signals (same for all trials)
        neuron1 = np.sin(freq1 * t_single)
        neuron2 = np.sin(freq2 * t_single)
        # Weight combinations
        weights = [1, 0.9, 0.5, 0.1, 0]
        w_pairs = [(1 - w, w) for w in weights]  # list of (w1, w2)
        # Create full signals
        full_time = []
        full_n1 = []
        full_n2 = []
        full_output = []
        for i, ((w1, w2), mean, mag) in enumerate(zip(w_pairs, input_means, input_magnitudes)):
            offset = i * 2 * np.pi
            n1 = mag * np.sin(freq1 * t_single) + mean
            n2 = mag * np.sin(freq2 * t_single) + mean
            out = w1 * n1 + w2 * n2
            full_time.append(t_single + offset)
            full_n1.append(n1)
            full_n2.append(n2)
            full_output.append(out)
        # Concatenate all trials
        t_all = np.concatenate(full_time)
        n1_all = np.concatenate(full_n1)
        n2_all = np.concatenate(full_n2)
        output_all = np.concatenate(full_output)

        return t_all, n1_all, n2_all, output_all


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


class Mouse_Dataset(pl.LightningDataModule):

    def __init__(self, m1_path, emg_path, behavioral_path, 
                    num_modes, batch_size, dataset_type, 
                    seed, kmeans_cluster, label_type,
                    remove_zeros, scale_outputs, mean_centering):
        super().__init__()

        # Assign class variables
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
        self.mean_centering = mean_centering

        # Set manual seed
        torch.manual_seed(seed)

        # If loading kmeans split data
        if "kmeans_split" in m1_path:
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

        # If loading data for just one label
        elif len(self.label_type) > 5:
            curr_label = float(self.label_type[6:])
            self.format_mouse_data(curr_label)
            
        # If loading data for all labels
        else:
            self.format_mouse_data("all")


    def format_mouse_data(self, labels_to_use):

        # Format and save data if files don't exist
        curr_dir = os.getcwd()
        if not os.path.exists(f"{curr_dir}/data/mouse_files/m1.npy"):
            self.format_and_save()

        # Load pre-saved partially formatted mouse data files
        m1 = np.load(f"{curr_dir}/data/mouse_files/m1.npy")
        emg = np.load(f"{curr_dir}/data/mouse_files/emg.npy")
        behavior = np.load(f"{curr_dir}/data/mouse_files/behavior.npy")

        # Format into trials
        m1_trials = []
        emg_trials = []
        behavior_trials = []
        # Calculate length of each behavior
        last_behavior = behavior[0]
        curr_m1_trial = []
        curr_emg_trial = []
        curr_behavior_trial = []
        for i in range(len(behavior)):
            curr_behavior = behavior[i]
            # Still within the same behavior
            if curr_behavior == last_behavior:
                curr_m1_trial.append(m1[i])
                curr_emg_trial.append(emg[i])
                curr_behavior_trial.append(behavior[i])
            # Reached start of new behavior
            else:
                # Update last behavior
                last_behavior = curr_behavior
                m1_trials.append(curr_m1_trial)
                emg_trials.append(curr_emg_trial)
                behavior_trials.append(curr_behavior_trial)
                # Reset buffers
                curr_m1_trial = [m1[i]]
                curr_emg_trial = [emg[i]]
                curr_behavior_trial = [behavior[i]]
        # Clear buffer one last time
        m1_trials.append(curr_m1_trial)
        emg_trials.append(curr_emg_trial)
        behavior_trials.append(curr_behavior_trial)

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
        if len(X_train) != 0:
            X_train = torch.Tensor(np.concatenate(X_train))
            y_train_emg = torch.Tensor(np.concatenate([y[0] for y in y_train]))
            y_train_behavioral = np.concatenate([y[1] for y in y_train])
        if len(X_val) != 0:
            X_val = torch.Tensor(np.concatenate(X_val))
            y_val_emg = torch.Tensor(np.concatenate([y[0] for y in y_val]))
            y_val_behavioral = np.concatenate([y[1] for y in y_val])
        
        # Extract max value across each muscle channel for initializing scalevector
        max_vals = torch.max(torch.cat((y_train_emg, y_val_emg)), dim=0)[0]
        # print(max_vals)
        # Extract mean value across each muscle channel for initializing global bias vector
        mean_vals = torch.mean(torch.cat((y_train_emg, y_val_emg)), dim=0)
        # print(mean_vals)

        # Mean center M1 data
        if "m1" in self.mean_centering:
            train_m1_mean = X_train.mean(axis=0)
            X_train = X_train - train_m1_mean
            X_val = X_val - train_m1_mean

        # Mean center EMG data
        if "emg" in self.mean_centering:
            train_emg_mean = y_train_emg.mean(axis=0)
            y_train_emg = y_train_emg - train_emg_mean
            y_val_emg = y_val_emg - train_emg_mean

        # Create final datasets for training
        self.train_dataset = [(X_train[i], y_train_emg[i], y_train_behavioral[i]) for i in range(len(X_train))]
        self.val_dataset = [(X_val[i], y_val_emg[i], y_val_behavioral[i]) for i in range(len(X_val))]
        self.N = m1.shape[1]
        self.M = emg.shape[1]


    def format_and_save(self):

        # Load in mouse dataset from .mat file
        curr_dir = os.getcwd()
        filepath = f"{curr_dir}/data/mouse_files/D020DataForDecoding.mat"
        f = h5py.File(filepath)
        arrays = {}
        for k, v in tqdm(f.items()):
            arrays[k] = np.array(v) 

        # Extract M1, EMG, and labels
        neural = arrays["firingRates"]
        cortexInds = arrays["cortexInds"]
        m1 = neural[:,149:] # Extract dimensions 150 to 232 for M1 data
        emg = arrays["emg"]
        behavior = arrays["humanBehaviorLabels"]

        # Condense data into 25ms timestamps
        m1_25ms = []
        emg_25ms = []
        behavior_25ms = []
        for i in range(0, len(m1), 25):
            m1_25ms.append(np.mean(m1[i:i+25], axis=0))
            emg_25ms.append(np.mean(emg[i:i+25], axis=0))
            # Take majority behavior label
            behavior_counts = np.bincount(behavior[i:i+25].squeeze(1).astype(np.int64))
            majority_behavior = np.argmax(behavior_counts)
            behavior_25ms.append(float(majority_behavior))
        # Convert back to arrays
        m1 = np.array(m1_25ms)
        emg = np.array(emg_25ms)
        behavior = np.array(behavior_25ms)

        # Format into bins of B=10
        B = 10
        m1_b10 = []
        for i in range(len(m1)-B):
            # Extract current timestamp and the B-1 subsequent timestamps
            m1_b10.append(np.concatenate(m1[i:i+10]))
        # Convert back to arrays
        m1 = np.stack(m1_b10)
        emg = np.array(emg[:-B]) # In order to match the B=10 formatting for M1
        behavior = np.array(behavior[:-B]) # In order to match the B=10 formatting for M1

        # Remove features with "nan" values
        nan_indices = np.where(np.any(np.isnan(m1), axis=1))[0]
        m1 = np.delete(m1, nan_indices, axis=0)
        emg = np.delete(emg, nan_indices, axis=0)
        behavior = np.delete(behavior, nan_indices, axis=0)

        # Save formatted M1, EMG, and behavior numpy arrays
        np.save(f"{curr_dir}/data/mouse_files/m1.npy", m1)
        np.save(f"{curr_dir}/data/mouse_files/emg.npy", emg)
        np.save(f"{curr_dir}/data/mouse_files/behavior.npy", behavior)


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


class Cage_Dataset(pl.LightningDataModule):
    
    def __init__(self, m1_path, emg_path, behavioral_path, 
                 num_modes, batch_size, dataset_type, 
                 seed, kmeans_cluster, label_type,
                 remove_zeros, scale_outputs, mean_centering):
        super().__init__()

        # Assign class variables
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
        self.mean_centering = mean_centering

        # Set manual seed
        torch.manual_seed(seed)

        # Load data with Set1 labels
        if "set1" in self.label_type:

            # If loading kmeans split data
            if "kmeans_split" in m1_path:
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

            # If loading data to test generalizability
            elif "generalizability" in self.label_type:

                # Load correct generalizability experiment
                if self.label_type == "set1_generalizability_grooming":
                    experiment_types = ["grooming"]
                elif self.label_type == "set1_generalizability_grooming_sitting_still":
                    experiment_types = ["grooming", "sitting_still"]

                self.format_set1_data_generalizability(experiment_types)
                
            # If loading data for just one label
            elif len(self.label_type) > 4:
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

                # Extract max value across each muscle channel for initializing scalevector
                max_vals = torch.max(torch.cat((y_train_emg, y_val_emg)), dim=0)[0]
                # print(max_vals)
                # Extract mean value across each muscle channel for initializing global bias vector
                mean_vals = torch.mean(torch.cat((y_train_emg, y_val_emg)), dim=0)
                # print(mean_vals)

                # Mean center M1 data
                if "m1" in self.mean_centering:
                    train_m1_mean = X_train.mean(axis=0)
                    X_train = X_train - train_m1_mean
                    X_val = X_val - train_m1_mean

                # Mean center EMG data
                if "emg" in self.mean_centering:
                    train_emg_mean = y_train_emg.mean(axis=0)
                    y_train_emg = y_train_emg - train_emg_mean
                    y_val_emg = y_val_emg - train_emg_mean
                
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
        elif "none" in self.label_type:

            # If using kmeans split data for sanity check
            if "kmeans_split" in m1_path:
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
            
            # If loading data to test generalizability
            elif "generalizability" in self.label_type:

                # Load correct generalizability experiment
                if self.label_type == "none_generalizability_unlabeled":
                    experiment_types = ["unlabeled"]
                elif self.label_type == "none_generalizability_labeled":
                    experiment_types = ["labeled"]

                self.format_none_data_generalizability(experiment_types)

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


    def format_set1_data_generalizability(self, experiment_types):

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

        # Format into train and val sets
        m1_train = []
        emg_train = []
        behavior_train = []
        m1_val = []
        emg_val = []
        behavior_val = []
        for N in range(len(my_cage_data.behave_tags['tag'])):
            curr_behavior = my_cage_data.behave_tags['tag'][N]
            start_idx, end_idx = find_start_end(N, timeframe)

            # If running "grooming" or "sitting_still" generalizability experiment
            if curr_behavior in experiment_types:
                m1_val.append(m1[start_idx:end_idx+1])
                emg_val.append(emg[start_idx:end_idx+1])
                behavior_val.append([curr_behavior]*(end_idx-start_idx+1))
            else:
                m1_train.append(m1[start_idx:end_idx+1])
                emg_train.append(emg[start_idx:end_idx+1])
                behavior_train.append([curr_behavior]*(end_idx-start_idx+1))

        # Format back into time stamps
        X_train = torch.Tensor(np.concatenate(m1_train))
        y_train_emg = torch.Tensor(np.concatenate(emg_train))
        y_train_behavioral = np.concatenate(behavior_train)
        X_val = torch.Tensor(np.concatenate(m1_val))
        y_val_emg = torch.Tensor(np.concatenate(emg_val))
        y_val_behavioral = np.concatenate(behavior_val)

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
        

    def format_none_data_generalizability(self, experiment_types):

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
        
        # Format into train and val sets
        m1_train = []
        emg_train = []
        behavior_train = []
        m1_val = []
        emg_val = []
        behavior_val = []
        for i in range(len(m1)):
            curr_m1 = m1[i]
            curr_emg = emg[i]
            curr_behavior = behavior[i]

            # If currently an unlabeled timestamp:
            if curr_behavior == "none":
                # Use either unlabeled or labeled timestamps as validation set
                if experiment_types == ["unlabeled"]:
                    m1_val.append(curr_m1)
                    emg_val.append(curr_emg)
                    behavior_val.append(curr_behavior)
                elif experiment_types == ["labeled"]:
                    m1_train.append(curr_m1)
                    emg_train.append(curr_emg)
                    behavior_train.append(curr_behavior)
            # If currently a labeled timestamp:
            else:
                # Use either unlabeled or labeled timestamps as validation set
                if experiment_types == ["unlabeled"]:
                    m1_train.append(curr_m1)
                    emg_train.append(curr_emg)
                    behavior_train.append(curr_behavior)
                elif experiment_types == ["labeled"]:
                    m1_val.append(curr_m1)
                    emg_val.append(curr_emg)
                    behavior_val.append(curr_behavior)

        # Format back into time stamps
        X_train = torch.Tensor(np.array(m1_train))
        y_train_emg = torch.Tensor(np.array(emg_train))
        y_train_behavioral = np.array(behavior_train)
        X_val = torch.Tensor(np.array(m1_val))
        y_val_emg = torch.Tensor(np.array(emg_val))
        y_val_behavioral = np.array(behavior_val)

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


