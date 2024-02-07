# Script to evaluate performance of trained models

import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score
cwd = os.getcwd()
sys.path.append(cwd)

from data.data import *
from model.model import *
from run.TrainingModule import *
from utils.constants import *

# Run script using command "python3 eval/evaluate.py configs/configs_cage.yaml" in home directory

def check_clustering(model_path, num_to_print, dataset, config):
    """
    Checks clustering for trained model by comparing behavioral labels to learned clusters.
    Input: (str) model_path: path to model to evaluate
           (int) num_to_print: number of samples to print
    """
    
    # Access train dataset
    train = dataset.train_dataset
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
    # print(ids[:num_to_print])
    plt.bar(timestamps[:num_to_print], ids[:num_to_print], width=1.0, alpha=0.5)
    # plt.show()

    # Load trained model
    config.ev = True
    model = CombinedModel(dataset.N, dataset.M, config.d, config.ev)
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
    # print(cluster_probs[:num_to_print])
    cluster_ids = [val.index(max(val))+1 for val in cluster_probs]
    # print(cluster_ids[:num_to_print])
    plt.bar(timestamps[:num_to_print], cluster_ids[:num_to_print], width=1.0, alpha=0.5)
    plt.title("Behavioral Labels and Majority Cluster Mode over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Mode ID")
    plt.show()

    # Print confusion matrix
    print("Printing confusion matrix, C[i,j] is number of observations in group 'i' but predicted to be 'j'")
    print(confusion_matrix(ids, cluster_ids))



def full_R2(dataset, config):
    """
    Calculates full R^2 value over the three separately trained linear decoders on Set2 labels.
    """

    # Load in trained models for each behavioral label
    model_crawl = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint33_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model_crawl = TrainingModule(model_crawl, config.lr, config.record, config.type)
    model_crawl.load_state_dict(state_dict)
    
    model_precision = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint34_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model_precision = TrainingModule(model_precision, config.lr, config.record, config.type)
    model_precision.load_state_dict(state_dict)

    model_power = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint35_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model_power = TrainingModule(model_power, config.lr, config.record, config.type)
    model_power.load_state_dict(state_dict)

    # Generate predicted value for each input training sample
    emgs = []
    preds = []
    for val in dataset.train_dataset:
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

    return r2



def sep_R2(dataset, config):
    """
    Calculates separate R^2 values for each of the individual behavioral labels in Set2 for our model.
    """

    # Load in trained model
    model = CombinedModel(dataset.N, dataset.M, config.d, config.ev)
    checkpoint = torch.load("checkpoints/checkpoint32_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model = TrainingModule(model, config.lr, config.record, config.type)
    model.load_state_dict(state_dict)

    # Generate predicted value for each input training sample
    emg_dict = {}
    preds_dict = {}
    for val in dataset.train_dataset:
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
    for key in emg_dict.keys():
        emg_dict[key] = torch.stack(emg_dict[key])
        preds_dict[key] = torch.stack(preds_dict[key]).detach()
        curr_r2 = r2_score(emg_dict[key], preds_dict[key])
        r2_values.append(curr_r2)

    return r2_values
        





if __name__ == "__main__":
    
    print("\nRunning 'evaluate.py'...\n")

    # Read in configs and dataset
    config = load_config()
    dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type)

    # Evaluate model clustering 
    model_path = "checkpoints/checkpoint32_epoch=499.ckpt"
    num_to_print = 300
    # check_clustering(dataset=dataset, model_path=model_path, num_to_print=num_to_print)

    # Calculate full R^2 over separate models
    full_r2_value = full_R2(dataset=dataset, config=config)
    # print(full_r2_value)

    # Calculate separate R^2 for each behavioral label in our model
    sep_r2_values = sep_R2(dataset=dataset, config=config)
    # print(sep_r2_values)




