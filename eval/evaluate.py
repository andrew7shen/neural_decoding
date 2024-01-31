# Script to evaluate performance of trained models

import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/andrewshen/Desktop/neural_decoding')

from data.data import *
from model.model import *
from run.TrainingModule import *
from utils.constants import *

# Run script using command "python3 eval/evaluate.py configs/configs_cage.yaml" in home directory

def check_clustering(model_path):
    """
    Checks clustering for trained model by comparing behavioral labels to learned clusters.
    """

    # Load configs
    config = load_config()

    # Load in training set
    dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type)
    train = dataset.train_dataset
    train_behavioral_labels = [val[2] for val in train]

    # Convert behavioral labels into numbers
    id_dict = {}
    curr_id = 1
    for label in train_behavioral_labels:
        if label not in id_dict:
            id_dict[label] = curr_id
            curr_id +=1
    timestamps = range(0, len(train_behavioral_labels))
    ids = [id_dict[val] for val in train_behavioral_labels]

    # Print ids of behavioral labels
    num_to_print = 10
    print(ids[:num_to_print])
    # plt.bar(timestamps[:num_to_print], ids[:num_to_print], width=1.0)
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
    print(cluster_ids[:num_to_print])
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    
    print("Running 'evaluate.py'...")

    model_path = "checkpoints/checkpoint32_epoch=499.ckpt"

    check_clustering(model_path=model_path)