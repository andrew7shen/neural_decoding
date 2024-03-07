# Script to evaluate performance of trained models

import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, r2_score
import umap.umap_ as umap
cwd = os.getcwd()
sys.path.append(cwd)

from data.data import *
from model.model import *
from run.TrainingModule import *
from utils.constants import *

# Run script using command "python3 eval/evaluate.py configs/configs_cage.yaml" in home directory

def dataset_statistics(dataset, verbose):
    train_timestamps = len(dataset.train_dataset)
    val_timestamps = len(dataset.val_dataset)
    total_timestamps = train_timestamps + val_timestamps
    train_mode_timestamps_dict = {}
    val_mode_timestamps_dict = {}
    for mode in [v[2] for v in dataset.train_dataset]:
        if mode not in train_mode_timestamps_dict.keys():
            train_mode_timestamps_dict[mode] = 1
        else:
            train_mode_timestamps_dict[mode] += 1
    for mode in [v[2] for v in dataset.val_dataset]:
        if mode not in val_mode_timestamps_dict.keys():
            val_mode_timestamps_dict[mode] = 1
        else:
            val_mode_timestamps_dict[mode] += 1

    train_crawl = train_mode_timestamps_dict["crawl"]
    train_precision = train_mode_timestamps_dict["precision"]
    train_power = train_mode_timestamps_dict["power"]
    val_crawl = val_mode_timestamps_dict["crawl"]
    val_precision = val_mode_timestamps_dict["precision"]
    val_power = val_mode_timestamps_dict["power"]
    tot_crawl = train_crawl + val_crawl
    tot_precision = train_precision + val_precision
    tot_power = train_power + val_power
    if verbose:
        print("Train timestamps: %s" % train_timestamps)
        print("\tTrain crawl timestamps: %s (%s)" % (train_crawl, (train_crawl/train_timestamps)))
        print("\tTrain precision timestamps: %s (%s)" % (train_precision, (train_precision/train_timestamps)))
        print("\tTrain power timestamps: %s (%s)" % (train_power, (train_power/train_timestamps)))
        print("Valid timestamps: %s" % val_timestamps)
        print("\tValid crawl timestamps: %s (%s)" % (val_crawl, (val_crawl/val_timestamps)))
        print("\tValid precision timestamps: %s (%s)" % (val_precision, (val_precision/val_timestamps)))
        print("\tValid power timestamps: %s (%s)" % (val_power, (val_power/val_timestamps)))
        print("Total timestamps: %s" % total_timestamps)
        print("\tTotal crawl timestamps: %s (%s)" % (tot_crawl, (tot_crawl/total_timestamps)))
        print("\tTotal precision timestamps: %s (%s)" % (tot_precision, (tot_precision/total_timestamps)))
        print("\tTotal power timestamps: %s (%s)" % (tot_power, (tot_power/total_timestamps)))
        print("N: %s" % dataset.N)
        print("M: %s" % dataset.M)

def check_clustering(model_path, num_to_print, dataset, config, verbose):
    """
    Checks clustering for trained model by comparing behavioral labels to learned clusters.
    Input: (str) model_path: path to model to evaluate
           (int) num_to_print: number of samples to print
    """
    
    # Access train dataset
    train = dataset.train_dataset
    train_emg_1 = torch.stack([val[1][0] for val in train]).tolist()
    train_behavioral_labels = [val[2] for val in train]

    # Convert ground truth behavioral labels into numbers
    id_dict = {}
    curr_id = 1
    for label in train_behavioral_labels:
        if label not in id_dict:
            id_dict[label] = curr_id
            curr_id +=1
    timestamps = range(0, len(train_behavioral_labels))
    ids = [id_dict[val] for val in train_behavioral_labels]

    # Print ids of ground truth behavioral labels
    # plt.bar(timestamps[:num_to_print], ids[:num_to_print], width=1.0, alpha=0.5)

    # Load trained model
    eval_mode = True
    model = CombinedModel(input_dim=dataset.N,
                              output_dim=dataset.M,
                              num_modes=config.d, 
                              temperature=config.temperature,
                              ev=eval_mode)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    model = TrainingModule(model, config.lr, config.record, config.type)
    model.load_state_dict(state_dict)

    # Print learned cluster labels
    cluster_probs = []
    for sample in train:
        x = sample[0].unsqueeze(0)
        curr_probs = model.forward(x)
        cluster_probs.append(curr_probs.squeeze())
    cluster_probs = torch.stack(cluster_probs).tolist()
    torch.set_printoptions(sci_mode=False)
    cluster_ids = [val.index(max(val))+1 for val in cluster_probs]
    # plt.bar(timestamps[:num_to_print], cluster_ids[:num_to_print], width=1.0, alpha=0.5)
    # plt.title("Behavioral Labels and Majority Cluster Mode over Time")
    # plt.xlabel("Timestamp")
    # plt.ylabel("Mode ID")

    # Graph of EMG with behavioral label and learned cluster labels overlaid with color
    ids_array = np.array(ids)
    cluster_ids_array = np.array(cluster_ids)
    num_timestamps = len(ids_array)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,3))
    ax1.plot(timestamps, train_emg_1, color="black")
    cmap = ListedColormap(["yellow","green","blue"], name='from_list', N=None)
    ax1.imshow(np.expand_dims(ids_array, 0),
               cmap=cmap,
               alpha=1.0,
               extent=[0, num_timestamps, -200, 200])
    ax1.title.set_text("Behavioral Labels")
    ax2.plot(timestamps, train_emg_1, color="black")
    ax2.imshow(np.expand_dims(cluster_ids_array, 0),
               cmap=cmap,
               alpha=1.0,
               extent=[0, num_timestamps, -200, 200])
    ax2.title.set_text("Learned Cluster Labels")

    if verbose:
        plt.show()

        # Print confusion matrix
        print("Printing confusion matrix, C[i,j] is number of observations in group 'i' but predicted to be 'j'")
        print(confusion_matrix(ids, cluster_ids))



def full_R2(dataset, config, verbose):
    """
    Calculates full R^2 value over the three separately trained linear decoders on Set2 labels.
    """

    # Load in trained models for each behavioral label
    model_crawl = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint58_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model_crawl = TrainingModule(model_crawl, config.lr, config.record, config.type)
    model_crawl.load_state_dict(state_dict)
    
    model_precision = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint59_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model_precision = TrainingModule(model_precision, config.lr, config.record, config.type)
    model_precision.load_state_dict(state_dict)

    model_power = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint60_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model_power = TrainingModule(model_power, config.lr, config.record, config.type)
    model_power.load_state_dict(state_dict)

    # Generate predicted value for each input training sample
    r2_list = []
    splits = ["train", "val"]
    out_str = "Full R2 values:\n\n"
    for split in splits:
        emgs = []
        preds = []
        if split == "train":
            curr_dataset = dataset.train_dataset
        elif split == "val":
            curr_dataset = dataset.val_dataset
        for val in curr_dataset:
            curr_m1 = val[0].unsqueeze(0)
            curr_emg = val[1]
            curr_behavioral = val[2]
            emgs.append(curr_emg)

            # Get predicted value from pretrained model depending on behavioral label
            if curr_behavioral == "crawl":
                preds.append(model_crawl.forward(curr_m1).squeeze())
            elif curr_behavioral == "precision":
                preds.append(model_precision.forward(curr_m1).squeeze())
            elif curr_behavioral == "power":
                preds.append(model_power.forward(curr_m1).squeeze())
        
        # Calculate final R^2 value
        emgs = torch.stack(emgs)
        preds = torch.stack(preds).detach()
        r2 = r2_score(emgs, preds)
        r2_list.append(r2)

        # Format output string
        out_str += "%s\n%s\n\n" % (split, r2)
    
    if verbose:
        print(out_str)

    return r2_list



def sep_R2(dataset, model_path, config, verbose):
    """
    Calculates separate R^2 values for each of the individual behavioral labels in Set2 for our model.
    """

    # Load in trained model
    model = CombinedModel(input_dim=dataset.N,
                              output_dim=dataset.M,
                              num_modes=config.d, 
                              temperature=config.temperature,
                              ev=config.ev)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    model = TrainingModule(model, config.lr, config.record, config.type)
    model.load_state_dict(state_dict)

    # Generate predicted value for each input training sample
    r2_list = []
    splits = ["train", "val"]
    for split in splits:
        emg_dict = {}
        preds_dict = {}
        if split == "train":
            curr_dataset = dataset.train_dataset
        elif split == "val":
            curr_dataset = dataset.val_dataset
        for val in curr_dataset:
            curr_m1 = val[0].unsqueeze(0)
            curr_emg = val[1]
            curr_behavioral = val[2]

            # Store the emg labels and predicted values in dicts
            if curr_behavioral not in emg_dict.keys():
                emg_dict[curr_behavioral] = [curr_emg]
                preds_dict[curr_behavioral] = [model.forward(curr_m1).squeeze()]
            else:
                emg_dict[curr_behavioral].append(curr_emg)
                preds_dict[curr_behavioral].append(model.forward(curr_m1).squeeze())

        # Calculate separate R^2 values
        r2_values = []
        out_str = ("Separate R2 values:\n\n%s\n" % split)
        for key in emg_dict.keys():
            emg_dict[key] = torch.stack(emg_dict[key])
            preds_dict[key] = torch.stack(preds_dict[key]).detach()
            curr_r2 = r2_score(emg_dict[key], preds_dict[key])
            r2_values.append(curr_r2)
            out_str += ("%s: %s\n" % (key, curr_r2))
        out_str += "\n"
        if verbose:
            print(out_str)

        r2_list.append(r2_values)

    return r2_list


def run_umap(dataset, verbose):
    # print(dataset.shape)
    reducer = umap.UMAP()
    # embedding = reducer.fit_transform(scaled_penguin_data)
    if verbose:
        print("umap")
        





if __name__ == "__main__":
    
    print("\nRunning 'evaluate.py'...\n")

    # Read in configs and dataset
    config = load_config()
    dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type, seed=config.seed)

    # Print dataset statistics
    dataset_statistics(dataset=dataset, verbose=False)
    
    # Perform UMAP on input dataset
    # run_umap(dataset=dataset, verbose=False)

    # Evaluate model clustering 
    model_id = 93
    model_path = "checkpoints/checkpoint%s_epoch=499.ckpt" % model_id
    num_to_print = 300
    check_clustering(dataset=dataset, model_path=model_path, num_to_print=num_to_print, config=config, verbose=True)

    # Calculate full R^2 over separate models
    full_r2_list = full_R2(dataset=dataset, config=config, verbose=False)

    # Calculate separate R^2 for each behavioral label in our model
    sep_r2_list = sep_R2(dataset=dataset, model_path=model_path, config=config, verbose=False)




