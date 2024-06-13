
# Import packages
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
cwd = os.getcwd()
sys.path.append(cwd)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from data import Cage_Dataset
from utils.constants import *

"""
Run script using command: "python3 data/kmeans_split.py configs/t100_configs/configs_cage_t100.yaml"
"""

config = load_config()
    
dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type, seed=config.seed, kmeans_cluster=config.kmeans_cluster)

m1_train = torch.stack([v[0] for v in dataset.train_dataset])
m1_val = torch.stack([v[0] for v in dataset.val_dataset])
emg_train = torch.stack([v[1] for v in dataset.train_dataset])
emg_val = torch.stack([v[1] for v in dataset.val_dataset])
labels_train = [v[2] for v in dataset.train_dataset]
labels_val = [v[2] for v in dataset.val_dataset]

# Apply kmeans to M1 training data
k = 6
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(m1_train)
preds = kmeans.labels_
val_preds = kmeans.predict(m1_val)

# Create save directory path
save_files = False
if "b10" in config.m1_path:
    save_dir = "k%s_b10" % k
elif "b10" not in config.m1_path:
    save_dir = "k%s" % k

# Create three separate train/val datasets for m1/emg/labels for each cluster
# Should end up with 6 files per cluster for train and val m1/emg/labels
# Initialize all dictionaries
m1_train_clusters = {}
emg_train_clusters = {}
labels_train_clusters = {}
m1_val_clusters = {}
emg_val_clusters = {}
labels_val_clusters = {}
for i in range(k):
    for dict in [m1_train_clusters, emg_train_clusters, labels_train_clusters, m1_val_clusters, emg_val_clusters, labels_val_clusters]:
        dict[i] = []
# Split up input data into each cluster
for i in range(len(m1_train)):
    curr_cluster = preds[i]
    m1_train_clusters[curr_cluster].append(m1_train[i])
    emg_train_clusters[curr_cluster].append(emg_train[i])
    labels_train_clusters[curr_cluster].append(labels_train[i])
for i in range(len(m1_val)):
    curr_cluster = val_preds[i]
    m1_val_clusters[curr_cluster].append(m1_val[i])
    emg_val_clusters[curr_cluster].append(emg_val[i])
    labels_val_clusters[curr_cluster].append(labels_val[i])

# Convert all to numpy arrays
for dict in [m1_train_clusters, emg_train_clusters, labels_train_clusters,
             m1_val_clusters, emg_val_clusters, labels_val_clusters]:
    for key in dict.keys():
        dict[key] = np.array(dict[key])

# Save split files as numpy arrays
out_path = "/Users/andrewshen/Desktop/neural_decoding/data/set2_data/kmeans_split/%s/" % save_dir
name_dict = {"m1_train": m1_train_clusters, "emg_train": emg_train_clusters, "labels_train": labels_train_clusters,
             "m1_val": m1_val_clusters, "emg_val": emg_val_clusters, "labels_val": labels_val_clusters}
for name in name_dict.keys():
    dict = name_dict[name]
    for key in dict.keys():
        if save_files:
            np.save("%s%s_%s" % (out_path, name, key), dict[key])

# Plot PCA of kmeans on training data
pca = PCA(n_components=2)
# pca_result = pca.fit_transform(m1_train)
# pca1 = pca_result[:,0]
# pca2 = pca_result[:,1]
# preds_color_dict = {0: "green", 1: "blue", 2: "red"}
# colors_preds = [preds_color_dict[v] for v in preds]
# plt.figure(figsize=(12,7))
# plt.scatter(pca1, pca2, s=8, c=colors_preds)
# plt.title("PCA Results from Kmeans on Training Data")
# plt.xlabel("PCA1")
# plt.ylabel("PCA2")
# green_patch = mpatches.Patch(color='green', label='0')
# blue_patch = mpatches.Patch(color='blue', label='1')
# red_patch = mpatches.Patch(color='red', label='2')
# plt.legend(handles=[green_patch, blue_patch, red_patch])
# commented out saving of files because already saved
# plt.show()
# plt.savefig("figures/pca_M1_train.png")

# Perform linear regression on each cluster dataset
weights_list = []
for cluster_id in m1_train_clusters.keys():
    curr_m1_train = m1_train_clusters[cluster_id]
    curr_emg_train = emg_train_clusters[cluster_id]
    curr_reg = LinearRegression().fit(curr_m1_train, curr_emg_train)
    curr_weights = curr_reg.coef_
    weights_list.append(np.array(curr_weights))
    
# Save model weights for each cluster
out_path = "/Users/andrewshen/Desktop/neural_decoding/data/set2_data/decoder_weights/%s/" % save_dir
for i in range(len(weights_list)):
    weights = weights_list[i]
    if save_files:
        np.save("%sweights_%s" % (out_path, i), weights)
