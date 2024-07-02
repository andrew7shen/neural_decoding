
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
from sklearn.model_selection import train_test_split
from data import Cage_Dataset
from utils.constants import *
import random

"""
Run script using command: "python3 data/kmeans_split.py configs/t100_configs/configs_cage_t100.yaml"
"""

config = load_config()
curr_dir = os.getcwd()
    
dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type, seed=config.seed, kmeans_cluster=config.kmeans_cluster)

m1_train = torch.stack([v[0] for v in dataset.train_dataset])
m1_val = torch.stack([v[0] for v in dataset.val_dataset])
emg_train = torch.stack([v[1] for v in dataset.train_dataset])
emg_val = torch.stack([v[1] for v in dataset.val_dataset])
labels_train = [v[2] for v in dataset.train_dataset]
labels_val = [v[2] for v in dataset.val_dataset]

# Create initial clusters
# cluster_type = "kmeans"
cluster_type = "random"
k = 6
# Apply kmeans to M1 training data
if cluster_type == "kmeans":
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(m1_train)
    preds = kmeans.labels_
    val_preds = kmeans.predict(m1_val)
elif cluster_type == "random":
    random.seed(0)

    # Create initial list of clusters from 0 to k
    preds = []
    val_preds = []
    for dataset, clusters in [[m1_train, preds], [m1_val, val_preds]]:
        curr_cluster = 0
        for i in range(len(dataset)):
            if curr_cluster == k:
                curr_cluster = 0
            clusters.append(curr_cluster)
            curr_cluster += 1
    
    # Assign random clusters
    random.shuffle(preds)
    random.shuffle(val_preds)

# Create save directory path
save_files = True
if "b10" in config.m1_path:
    if cluster_type == "kmeans":
        save_dir = "k%s_b10" % k
    elif cluster_type == "random": 
        save_dir = "k%s_b10_random" % k
elif "b10" not in config.m1_path:
    if cluster_type == "kmeans":
        save_dir = "k%s" % k
    elif cluster_type == "random": 
        save_dir = "k%s_random" % k

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
out_path = "%s/data/set2_data/kmeans_split/%s/" % (curr_dir, save_dir)
if not os.path.exists(out_path):
    os.mkdir(out_path)
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
out_path = "%s/data/set2_data/decoder_weights/%s/" % (curr_dir, save_dir)
if not os.path.exists(out_path):
    os.mkdir(out_path)
for i in range(len(weights_list)):
    weights = weights_list[i]
    if save_files:
        np.save("%sweights_%s" % (out_path, i), weights)
