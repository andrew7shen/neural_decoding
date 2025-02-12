# Script to evaluate performance of trained models

import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.patches as mpatches
cwd = os.getcwd()
sys.path.append(cwd)

from data.data import *
from model.model import *
from run.TrainingModule import *
from utils.constants import *

"""
Run script using command "python3 eval/evaluate.py configs/configs_cage.yaml" in home directory
Expanded trial range: "python3 eval/evaluate.py configs/t100_configs/configs_cage_t100.yaml"
Expanded trial range (B=10): "python3 eval/evaluate.py configs/b_configs/configs_cage_t100_b10.yaml"
Expanded trial range (d=4): "python3 eval/evaluate.py configs/d_configs/configs_cage_t100_d4.yaml"
Set 1 data: "python3 eval/evaluate.py configs/t100_configs/configs_cage_t100_set1.yaml"
Kmeans clusters #0 for None data: "python3 eval/evaluate.py configs/kmeans_split_configs/configs_cage_t100_kmeans0_none.yaml"
"""


def dataset_statistics(dataset, verbose):
    
    if not verbose:
        return
    
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
    
    # Extract number of samples for each behavioral label
    labels = train_mode_timestamps_dict.keys()
    count_dict = {}
    for label in labels:
        train_curr = train_mode_timestamps_dict[label]
        val_curr = val_mode_timestamps_dict[label]
        tot_curr = train_curr + val_curr
        count_dict[label] = [train_curr, val_curr, tot_curr]

    # Print out calculated numbers of samples
    print("Train timestamps: %s" % train_timestamps)
    for label in labels:
        curr_counts = count_dict[label]
        print("\tTrain %s timestamps: %s (%s)" % (label, curr_counts[0], (curr_counts[0]/train_timestamps)))
    print("Valid timestamps: %s" % val_timestamps)
    for label in labels:
        curr_counts = count_dict[label]
        print("\tValid %s timestamps: %s (%s)" % (label, curr_counts[1], (curr_counts[1]/val_timestamps)))
    print("Total timestamps: %s" % total_timestamps)
    for label in labels:
        curr_counts = count_dict[label]
        print("\tTotal %s timestamps: %s (%s)" % (label, curr_counts[2], (curr_counts[2]/total_timestamps)))
    print("N: %s" % dataset.N)
    print("M: %s" % dataset.M)


def check_clustering(model_path, num_to_print, dataset, config, plot_type, model_id, verbose):
    """
    Checks clustering for trained model by comparing behavioral labels to learned clusters.
    Input: (str) model_path: path to model to evaluate
           (int) num_to_print: number of samples to print
    """

    if not verbose:
        return
    
    # Access train dataset
    train = dataset.train_dataset
    train_m1 = torch.stack([val[0] for val in train])
    train_m1_1 = train_m1[:,0].tolist()
    train_emg = torch.stack([val[1] for val in train])
    train_emg_1 = train_emg[:,0].tolist()
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

    # Load trained model
    eval_mode = True
    model = CombinedModel(input_dim=dataset.N,
                          hidden_dim=config.hidden_dim,
                              output_dim=dataset.M,
                              num_modes=config.d, 
                              ev=eval_mode,
                              cluster_model_type=config.cluster_model_type,
                              decoder_model_type=config.decoder_model_type)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    model = TrainingModule(model=model,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type,
                           temperature=config.temperature,
                           anneal_temperature=config.anneal_temperature,
                           num_epochs=config.epochs,
                           end_temperature=config.end_temperature)
    model.load_state_dict(state_dict)

    # Calculate learned cluster labels
    cluster_probs = []
    sample_modes = []
    for sample in train:
        x = sample[0].unsqueeze(0)
        curr_output = model.forward(x)
        curr_probs = curr_output[0]
        curr_mode = sample[2]
        cluster_probs.append(curr_probs.squeeze())
        sample_modes.append(curr_mode)
    cluster_probs = torch.stack(cluster_probs).tolist()
    torch.set_printoptions(sci_mode=False)
    cluster_ids = [val.index(max(val))+1 for val in cluster_probs]

    # Create color map and label options for multiple modes
    cmap_colors = ["yellow","green","blue","red","purple","orange","pink","brown","gray","olive","cyan"]
    number_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"] 

    # Graph of EMG with behavioral label and learned cluster labels overlaid with color
    if plot_type == "majority":

        # Determine whether plotting M1 or EMG
        dataset_to_plot = "m1"
        # dataset_to_plot = "emg"
        if dataset_to_plot == "m1":
            train_dataset = train_m1 * 10
            train_dataset_1 = train_m1_1 * 10
        elif dataset_to_plot == "emg":
            train_dataset = train_emg
            train_dataset_1 = train_emg_1


        # Plot EMG info on top of clustering info
        plot_pcs = True
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(train_dataset)
        pca1 = pca_result[:,0]
        pca2 = pca_result[:,1]

        ids_array = np.array(ids)[:num_to_print]
        cluster_ids_array = np.array(cluster_ids)[:num_to_print]
        num_timestamps = num_to_print
        # TODO: Original code
        # num_timestamps = len(ids_array)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10,3))

        # Plot EMG info 
        if plot_pcs: 
            ax1.plot(timestamps[:num_to_print], pca1[:num_to_print], color="black")
            ax1.plot(timestamps[:num_to_print], pca2[:num_to_print], color="white")
        else:
            ax1.plot(timestamps[:num_to_print], train_dataset_1[:num_to_print], color="black")
        
        cmap = ListedColormap(cmap_colors, name='from_list', N=None)
        cmap_preds = ListedColormap(cmap_colors[:config.d], name='from_list', N=None)
        # TODO: Original code
        # cmap = ListedColormap(["yellow","green","blue"], name='from_list', N=None)
        # cmap_preds = ListedColormap(cmap_colors[:config.d], name='from_list', N=None)
        # Plot ground truth majority labels
        ax1.imshow(np.expand_dims(ids_array, 0),
                cmap=cmap,
                alpha=1.0,
                extent=[0, num_timestamps, -200, 200])
        ax1.title.set_text("Behavioral Labels")

        # Plot EMG info 
        if plot_pcs: 
            ax2.plot(timestamps[:num_to_print], pca1[:num_to_print], color="black")
            ax2.plot(timestamps[:num_to_print], pca2[:num_to_print], color="white")
        else:
            ax2.plot(timestamps[:num_to_print], train_dataset_1[:num_to_print], color="black")

        # Plot predicted majority labels
        ax2.imshow(np.expand_dims(cluster_ids_array, 0),
                cmap=cmap_preds,
                alpha=1.0,
                extent=[0, num_timestamps, -200, 200])
        ax2.title.set_text("Learned Cluster Labels")
        plt.tight_layout()
        plt.savefig("figures/%s.png" % model_id)
        # plt.savefig("figures/intervals/%s.png" % model_id) # TODO: Change back after generate interval plots

    elif plot_type == "distributions":

        # Format input data
        x = timestamps[:num_to_print]
        y = []
        labels = []
        colors = []
        
        for i in range(len(cluster_probs[0])):
            y.append([v[i] for v in cluster_probs[:num_to_print]])
            labels.append(number_labels[i])
            colors.append(cmap_colors[i])
        # TODO: Original code
        # y = [
        #     [v[0] for v in cluster_probs[:num_to_print]],
        #     [v[1] for v in cluster_probs[:num_to_print]],
        #     [v[2] for v in cluster_probs[:num_to_print]],
        # ]

        # Count number of timestamps with discrete clustering
        max_vals = [max(cluster_prob) for cluster_prob in cluster_probs]
        limits = [0.6, 0.7, 0.8, 0.9]
        for limit in limits:
            num_above_limit = sum(max_val > limit for max_val in max_vals)
            print(f"{num_above_limit} out of {len(cluster_probs)} above {limit} ({num_above_limit/len(cluster_probs)})")

        # Plot cluster distributions with sliding functionality
        range_to_show = 100
        slider = False
        if not slider:
            plt.figure(figsize=(12,2))

            plt.stackplot(x, y, labels=labels, colors=colors)
            # TODO: Original code
            # plt.stackplot(x, y, labels=['1','2','3'], colors=["yellow", "green", "blue"])

            plt.title("Learned Cluster Distributions")
            plt.xlabel("Timestamp")
        else:
            Plot, Axis = plt.subplots()
            plt.subplots_adjust(bottom=0.25)
            plt.stackplot(x, y, labels=['1','2','3'], colors=["yellow", "green", "blue"])
            plt.title("Learned Cluster Distributions")
            plt.xlabel("Timestamp")
            slider_color = 'White'
            axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                                    facecolor = slider_color)
            slider_position = Slider(axis_position,
                                    'Pos', 0.1, num_to_print)
            def update(val):
                pos = slider_position.val
                Axis.axis([pos, pos+range_to_show, 0, 1])
                Plot.canvas.draw_idle()
            slider_position.on_changed(update)
        plt.savefig("figures/%sd.png" % model_id)

    elif plot_type == "mode_average":

        # Split up predictions into each mode
        modes_dict = {}
        for i in range(len(sample_modes)):
            curr_mode = sample_modes[i]
            curr_probs = cluster_probs[i]
            if curr_mode not in modes_dict.keys():
                modes_dict[curr_mode] = [curr_probs]
            else:
                modes_dict[curr_mode].append(curr_probs)

        # Calculate average over all trials across each mode
        trial_range = 100
        for k,v in modes_dict.items():
            num_items = len(v)
            modes_dict[k] = torch.reshape(torch.Tensor(v), (num_items//trial_range, trial_range, dataset.num_modes))
            modes_dict[k] = torch.mean(modes_dict[k], dim=0)

        # Plot averaged cluster predictions for each mode
        for k,v in modes_dict.items():
            x = timestamps[:trial_range]
            y = torch.transpose(v, 0, 1)

            # Plot cluster distributions
            plt.title("Learned Cluster Distributions for Mode %s" % k)
            plt.stackplot(x, y, labels=number_labels[:config.d], colors=cmap_colors[:config.d])
            # plt.show()
            plt.savefig("figures/%s_%s_avg.png" % (model_id, k))
            # plt.savefig("slide12_revised_b10_%s.pdf" % k)
            # plt.savefig("slide12_revised_b1_%s.pdf" % k)
            # plt.savefig("figures/intervals/%s_%s_avg.png" % (model_id, k)) # TODO: Change back after generate interval plots
            # plt.show()
    
    elif plot_type == "confusion_matrix":
        # Print confusion matrix
        print("Printing confusion matrix, C[i,j] is number of observations in group 'i' but predicted to be 'j'")
        print(confusion_matrix(ids, cluster_ids))

    elif plot_type == "discrete_state_overlap":

        # Populate overlap dict
        overlap_dict = {}
        num_modes = len(cluster_probs[0])
        for i in range(num_modes):
            curr_dict = {}
            for j in range(num_modes):
                curr_dict[j] = 0
            overlap_dict[i] = curr_dict
        
        # Calculate discrete state overlap between ground truth and predicted labels
        # Outside keys are ground truth labels (i.e. manual labels)
        # If using cage dataset
        if dataset.label_type == "set2":
            ids = [val-1 for val in ids]
            cluster_ids = [val-1 for val in cluster_ids]
        # If using mouse dataset
        elif dataset.label_type == "mouse":
            ids = [val[2] for val in dataset.train_dataset]
            cluster_ids = [val-1 for val in cluster_ids]
        for i in range(len(ids)):
            curr_manual_label = ids[i]
            curr_pred_label = cluster_ids[i]
            overlap_dict[curr_manual_label][curr_pred_label] += 1

        # Convert dictionary to nested list for input into heatmap
        overlap_list = []
        for outer_key in overlap_dict.keys():
            curr_list = []
            for inner_key in overlap_dict[outer_key].keys():
                curr_list.append(overlap_dict[outer_key][inner_key])
            overlap_list.append(curr_list)
        overlap_array = np.transpose(np.array(overlap_list))

        # Normalize over each manually annotated states
        normalize = True
        if normalize:
            normalized_array = []
            for i in range(len(overlap_array)):
                curr_list = []
                curr_column_sum = sum(overlap_array[:,i])
                for j in range(len(overlap_array)):
                    curr_list.append(overlap_array[j][i]/curr_column_sum)
                normalized_array.append(curr_list)
            normalized_array = np.transpose(np.array(normalized_array))
            overlap_array = normalized_array

        # Create heatmap of overlap values
        fig, ax = plt.subplots()
        heatmap = ax.imshow(overlap_array, cmap='Greys')
        fig.colorbar(heatmap)
        # If working with cage data
        if num_modes == 3:
            row_labels = ["1", "2", "3"]
            column_labels = ["crawl", "precision", "power"]
            title_str = "Discrete State Overlap (Cage Dataset)"
        # If working with mouse data
        elif num_modes == 11:
            row_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            column_labels = ["unlabeled", 'climbdown', 'climbup', 'eating', 'grooming',
                             'jumpdown', 'jumping', 'rearing', 'still', 'walkflat', 'walkgrid']
            title_str = "Discrete State Overlap (Mouse Dataset)"
        ax.set_xticks(np.arange(overlap_array.shape[1]))
        ax.set_yticks(np.arange(overlap_array.shape[0]))  
        ax.set_xticklabels(column_labels, rotation=45)
        ax.set_yticklabels(row_labels)
        ax.set_xlabel('Manually annotated states')
        ax.set_ylabel('Inferred discrete states')
        ax.set_title(title_str)
        fig.tight_layout()
        # plt.show()
        if normalize:
            plt.savefig("figures/%s_heatmap_norm.png" % model_id)
            # plt.savefig("figures/%s_heatmap_norm.pdf" % model_id)
        else:
            plt.savefig("figures/%s_heatmap.png" % model_id)
    # plt.show()


# Outdated and wrong
def full_R2(dataset, config, verbose):
    """
    Calculates full R^2 value over the three separately trained linear decoders on Set2 labels.
    """

    if not verbose:
        return
    
    # If using kmeans split data
    if len(dataset) > 1:
        dataset_list = dataset
        dataset = dataset_list[0]

    # Load in trained models for each behavioral label
    # model0 can be for crawl or cluster #0
    model0 = DecoderModel(dataset.N, dataset.M, 1)
    # checkpoint = torch.load("checkpoints/checkpoint96_epoch=499.ckpt")
    # checkpoint = torch.load("checkpoints/checkpoint156_epoch=499.ckpt")
    checkpoint = torch.load("checkpoints/checkpoint166_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model0 = TrainingModule(model=model0,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
    model0.load_state_dict(state_dict)
    
    # model1 can be for precision or cluster #1
    model1 = DecoderModel(dataset.N, dataset.M, 1)
    # checkpoint = torch.load("checkpoints/checkpoint97_epoch=499.ckpt")
    # checkpoint = torch.load("checkpoints/checkpoint157_epoch=499.ckpt")
    checkpoint = torch.load("checkpoints/checkpoint167_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model1 = TrainingModule(model=model1,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
    model1.load_state_dict(state_dict)

    # model2 can be for power or cluster #2
    model2 = DecoderModel(dataset.N, dataset.M, 1)
    # checkpoint = torch.load("checkpoints/checkpoint98_epoch=499.ckpt")
    # checkpoint = torch.load("checkpoints/checkpoint158_epoch=499.ckpt")
    checkpoint = torch.load("checkpoints/checkpoint168_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model2 = TrainingModule(model=model2,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
    model2.load_state_dict(state_dict)

    # TODO: Adding more models for k=6
    model3 = DecoderModel(dataset.N, dataset.M, 1)
    checkpoint = torch.load("checkpoints/checkpoint169_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model3 = TrainingModule(model=model3,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
    model3.load_state_dict(state_dict)

    model4 = DecoderModel(dataset.N, dataset.M, 1)
    # checkpoint = torch.load("checkpoints/checkpoint98_epoch=499.ckpt")
    checkpoint = torch.load("checkpoints/checkpoint170_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model4 = TrainingModule(model=model4,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
    model4.load_state_dict(state_dict)

    model5 = DecoderModel(dataset.N, dataset.M, 1)
    # checkpoint = torch.load("checkpoints/checkpoint98_epoch=499.ckpt")
    checkpoint = torch.load("checkpoints/checkpoint174_epoch=499.ckpt")
    state_dict = checkpoint["state_dict"]
    model5 = TrainingModule(model=model5,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
    model5.load_state_dict(state_dict)

    


    # Generate predicted value for each input training sample
    r2_list = []
    splits = ["train", "val"]
    out_str = "Full R2 values:\n\n"

    # If using kmeans split data
    if len(dataset_list) > 1:
        # models = [model0, model1, model2]
        models = [model0, model1, model2, model3, model4, model5]
        train_emgs = []
        train_preds = []
        val_emgs = []
        val_preds = []
        # Generate preds for each dataset/model pair
        for i in range(len(models)):
            curr_dataset = dataset_list[i]
            curr_model = models[i]
            train_dataset = curr_dataset.train_dataset
            val_dataset = curr_dataset.val_dataset
            # Generate train preds
            for val in train_dataset:
                curr_m1 = val[0].unsqueeze(0)
                curr_emg = val[1]
                curr_behavioral = val[2]
                train_emgs.append(curr_emg)
                train_preds.append(curr_model.forward(curr_m1).squeeze())
            # Generate val preds
            for val in val_dataset:
                curr_m1 = val[0].unsqueeze(0)
                curr_emg = val[1]
                curr_behavioral = val[2]
                val_emgs.append(curr_emg)
                val_preds.append(curr_model.forward(curr_m1).squeeze())
            
        # Calculate final R^2 value
        train_emgs = torch.stack(train_emgs)
        train_preds = torch.stack(train_preds).detach()
        train_r2 = r2_score(train_emgs, train_preds)
        r2_list.append(train_r2)
        val_emgs = torch.stack(val_emgs)
        val_preds = torch.stack(val_preds).detach()
        val_r2 = r2_score(val_emgs, val_preds)
        r2_list.append(val_r2)

        # Format output string
        out_str += "train\n%s\n\nval\n%s\n\n" % (train_r2, val_r2)
    
        print(out_str)

    else:
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
                    preds.append(model0.forward(curr_m1).squeeze())
                elif curr_behavioral == "precision":
                    preds.append(model1.forward(curr_m1).squeeze())
                elif curr_behavioral == "power":
                    preds.append(model2.forward(curr_m1).squeeze())
            
            # Calculate final R^2 value
            emgs = torch.stack(emgs)
            preds = torch.stack(preds).detach()
            r2 = r2_score(emgs, preds)
            r2_list.append(r2)

            # Format output string
            out_str += "%s\n%s\n\n" % (split, r2)
    
        print(out_str)

    return r2_list


def full_R2_reg(datasets, verbose):
    """
    Fits linear regressions for each dataset and calculates full R^2 value
    """

    if not verbose:
        return
    
    train_emgs = []
    train_preds = []
    val_emgs = []
    val_preds = []

    # Generate preds for each dataset/model pair
    total_train_samples = 0
    total_val_samples = 0
    for i in range(len(datasets)):
        curr_dataset = datasets[i]
        train_dataset = curr_dataset.train_dataset
        val_dataset = curr_dataset.val_dataset
        total_train_samples += len(train_dataset)
        total_val_samples += len(val_dataset)
        # Note: group can mean different modes or different clusters
        print("\nGroup %s: train (%s), val (%s)" % (i, len(train_dataset), len(val_dataset)))

        # Calculate linear regression model for each cluster
        curr_m1_train = np.array([val[0] for val in train_dataset])
        curr_emg_train = np.array([val[1] for val in train_dataset])
        curr_model = LinearRegression().fit(curr_m1_train, curr_emg_train)
        # Fit with Ridge regression
        curr_model = Ridge(alpha=15.0).fit(curr_m1_train, curr_emg_train)
        # Generate train preds and append to full list
        train_emgs.append(torch.Tensor(curr_emg_train))
        train_preds.append(torch.Tensor(curr_model.predict(curr_m1_train)))
        # Calculate train R2 for current cluster
        curr_train_r2 = r2_score(torch.Tensor(curr_emg_train), torch.Tensor(curr_model.predict(curr_m1_train)))
        print("Train R2: %s" % curr_train_r2)

        # Check if there are samples in validation set
        if len(val_dataset) != 0:

            # Calculate linear regression model for each cluster
            curr_m1_val = np.array([val[0] for val in val_dataset])
            curr_emg_val = np.array([val[1] for val in val_dataset])
            # Generate val preds and append to full list
            val_emgs.append(torch.Tensor(curr_emg_val))
            val_preds.append(torch.Tensor(curr_model.predict(curr_m1_val)))
            # Calculate val R2 for current cluster
            curr_val_r2 = r2_score(torch.Tensor(curr_emg_val), torch.Tensor(curr_model.predict(curr_m1_val)))
            print("Val R2: %s" % curr_val_r2)
        
        else:
            print("Val R2: None")

    # Calculate final R^2 value
    train_emgs = torch.cat(train_emgs)
    train_preds = torch.cat(train_preds)
    train_r2 = r2_score(train_emgs, train_preds)
    val_emgs = torch.cat(val_emgs)
    val_preds = torch.cat(val_preds)
    val_r2 = r2_score(val_emgs, val_preds)

    # Format output string
    print("\nFull Dataset %s: train (%s), val (%s)" % (i, total_train_samples, total_val_samples))
    print("Train R2: %s" % train_r2)
    print("Val R2: %s\n" % val_r2)

    return [train_r2, val_r2]


def model_sep_R2(dataset, model_path, config, verbose):
    """
    Calculates separate R^2 values for each of the individual behavioral labels in Set2 for our model.
    """

    if not verbose:
        return

    # Load trained model
    eval_mode = False
    model = CombinedModel(input_dim=dataset.N,
                          hidden_dim=config.hidden_dim,
                              output_dim=dataset.M,
                              num_modes=config.d, 
                              temperature=config.temperature,
                              ev=eval_mode,
                              cluster_model_type=config.cluster_model_type,
                              decoder_model_type=config.decoder_model_type)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    model = TrainingModule(model=model,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type)
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
        print(out_str)

        r2_list.append(r2_values)

    return r2_list


def sep_R2_reg(dataset, verbose):
    """
    Fits linear regression to entire dataset, and calculates separate R^2 values for each of the individual behavioral labels in Set2
    """

    if not verbose:
        return
    
    # Calculate linear regression model
    use_ffnn = True
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset
    m1_train = np.array([val[0] for val in train_dataset])
    emg_train = np.array([val[1] for val in train_dataset])
    m1_val = np.array([val[0] for val in val_dataset])
    emg_val = np.array([val[1] for val in val_dataset])
    # model = LinearRegression().fit(m1_train, emg_train)
    # Fit with Ridge regression
    if not use_ffnn:
        model = Ridge(alpha=15.0).fit(m1_train, emg_train)
    # Fit with neural network
    elif use_ffnn:
        print("Fitting MLPRegressor...")
        num_epochs = 300
        model = MLPRegressor(random_state=1, max_iter=num_epochs, verbose=False).fit(m1_train, emg_train)

    # Generate train and val preds
    train_preds = model.predict(m1_train)
    val_preds = model.predict(m1_val)

    # Calculate train/val R2 for current cluster
    train_r2 = r2_score(emg_train, train_preds)
    val_r2 = r2_score(emg_val, val_preds)
    print("\nFull Dataset: train (%s), val (%s)" % (len(train_dataset), len(val_dataset)))
    print("Train R2: %s" % train_r2)
    print("Val R2: %s\n" % val_r2)

    if use_ffnn:
        return []

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
                preds_dict[curr_behavioral] = [torch.Tensor(model.predict(curr_m1))]
            else:
                emg_dict[curr_behavioral].append(curr_emg)
                preds_dict[curr_behavioral].append(torch.Tensor(model.predict(curr_m1)))

        # Calculate separate R^2 values
        r2_values = []
        out_str = ("Separate R2 values:\n\n%s\n" % split)
        for key in emg_dict.keys():
            emg_dict[key] = torch.stack(emg_dict[key])
            preds_dict[key] = torch.stack(preds_dict[key]).squeeze()
            curr_r2 = r2_score(emg_dict[key], preds_dict[key])
            r2_values.append(curr_r2)
            out_str += ("%s: %s\n" % (key, curr_r2))
        out_str += "\n"
        print(out_str)

        r2_list.append(r2_values)

    return r2_list


def run_kmeans(dataset, config, verbose):
    """
    Following tutorial on kmeans: https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
    """

    if not verbose:
        return

    # Read in data
    m1 = torch.stack([v[0] for v in dataset.train_dataset] + [v[0] for v in dataset.val_dataset])
    emg = torch.stack([v[1] for v in dataset.train_dataset] + [v[1] for v in dataset.val_dataset])
    labels = [v[2] for v in dataset.train_dataset] + [v[2] for v in dataset.val_dataset]

    # TODO: Scale data
    # m1_scaled = StandardScaler().fit_transform(m1)

    # Perform kmeans on M1 and EMG
    data_name = ["M1", "EMG"]
    datasets = [m1, emg]
    # mode = "labels"
    # mode = "preds"
    mode = "clusters"
    for i in range(len(datasets)):
        data = datasets[i]

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        pca1 = pca_result[:,0]
        pca2 = pca_result[:,1]
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)

        # Apply kmeans
        kmeans = KMeans(n_clusters=3, n_init="auto")
        # kmeans.fit(pca_result)
        kmeans.fit(data)
        preds = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Calculate mode colors
        labels_color_dict = {"crawl": "green", "precision": "blue", "power": "red"}
        preds_color_dict = {0: "green", 1: "blue", 2: "red"}
        colors_labels = [labels_color_dict[v] for v in labels]
        colors_preds = [preds_color_dict[v] for v in preds]

        # Plot results
        if mode == "labels":
            plt.figure(figsize=(12,7))
            plt.scatter(pca1, pca2, s=8, c=colors_labels)
            plt.title("PCA Results (%s)" % data_name[i])
            plt.xlabel("PCA1")
            plt.ylabel("PCA2")
            green_patch = mpatches.Patch(color='green', label='crawl')
            blue_patch = mpatches.Patch(color='blue', label='precision')
            red_patch = mpatches.Patch(color='red', label='power')
            plt.legend(handles=[green_patch, blue_patch, red_patch])
            # plt.show()
            # plt.savefig("figures/pca_%s" % data_name[i])
        elif mode == "preds": # TODO: wrong
            plt.figure(figsize=(12,7))
            plt.scatter(pca1, pca2, s=8, c=colors_preds)
            plt.title("PCA Results (%s)" % data_name[i])
            plt.xlabel("PCA1")
            plt.ylabel("PCA2")
            green_patch = mpatches.Patch(color='green', label='0')
            blue_patch = mpatches.Patch(color='blue', label='1')
            red_patch = mpatches.Patch(color='red', label='2')
            plt.legend(handles=[green_patch, blue_patch, red_patch])
            # plt.show()
            # plt.savefig("figures/kmeans_%s" % data_name[i])
        elif mode == "clusters":
            # Same code from "clustering_check" function
            train = dataset.train_dataset
            train_emg_1 = torch.stack([val[1][0] for val in train]).tolist()
            train_behavioral_labels = [val[2] for val in train]
            id_dict = {}
            curr_id = 1
            for label in train_behavioral_labels:
                if label not in id_dict:
                    id_dict[label] = curr_id
                    curr_id +=1
            timestamps = range(0, len(train_behavioral_labels))
            ids = [id_dict[val] for val in train_behavioral_labels]
            ids_array = np.array(ids)
            num_timestamps = len(ids_array)
            cmap_colors = ["yellow","green","blue","red","purple","orange","pink"]
            number_labels = ["1", "2", "3", "4", "5", "6", "7"]
            fig, (ax1, ax2) = plt.subplots(2, figsize=(10,3))
            ax1.plot(timestamps, train_emg_1, color="black")
            cmap = ListedColormap(["yellow","green","blue"], name='from_list', N=None)
            cmap_preds = ListedColormap(cmap_colors[:config.d], name='from_list', N=None)
            ax1.imshow(np.expand_dims(ids_array, 0),
                    cmap=cmap,
                    alpha=1.0,
                    extent=[0, num_timestamps, -200, 200])
            ax1.title.set_text("Behavioral Labels")
            ax2.plot(timestamps, train_emg_1, color="black")
            ax2.imshow(np.expand_dims(preds, 0),
                    cmap=cmap_preds,
                    alpha=1.0,
                    extent=[0, num_timestamps, -200, 200])
            ax2.title.set_text("Learned Cluster Labels")
            # plt.show()
            plt.savefig("figures/kmeans_%s.png" % data_name[i])
        

def run_kmeans_M1(dataset, config):
    # Read in data
    m1 = torch.stack([v[0] for v in dataset.train_dataset] + [v[0] for v in dataset.val_dataset])
    emg = torch.stack([v[1] for v in dataset.train_dataset] + [v[1] for v in dataset.val_dataset])
    labels = [v[2] for v in dataset.train_dataset] + [v[2] for v in dataset.val_dataset]

    # Apply kmeans
    kmeans = KMeans(n_clusters=3, n_init="auto")
    kmeans.fit(m1)
    preds = kmeans.labels_
    centroids = kmeans.cluster_centers_
    import pdb; pdb.set_trace()
    pass


def sep_decoders_R2(model_path, dataset, config, plot_type, model_id, verbose):

    save_fig = True

    if not verbose:
        return

    # Load trained model
    eval_mode = True
    model = CombinedModel(input_dim=dataset.N,
                          hidden_dim=config.hidden_dim,
                              output_dim=dataset.M,
                              num_modes=config.d,
                              ev=eval_mode,
                              cluster_model_type=config.cluster_model_type,
                              decoder_model_type=config.decoder_model_type,
                              combined_model_type=config.combined_model_type)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    model = TrainingModule(model=model,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type,
                           temperature=config.temperature,
                           anneal_temperature=config.anneal_temperature,
                           num_epochs=config.epochs,
                           end_temperature=config.end_temperature,
                           lambda_val=config.lambda_val,
                           l1_lambda_val=config.l1_lambda_val)
    model.load_state_dict(state_dict)

    # Generate cm and dm weights
    train = dataset.train_dataset
    train_emg = torch.stack([val[1] for val in train])
    # train_behavioral_labels = [val[2] for val in train]
    cluster_probs = []
    decoder_outputs = []
    sample_behaviors = []
    for sample in train:
        x = sample[0].unsqueeze(0)
        curr_output = model.forward(x)
        curr_cluster_probs = curr_output[0]
        curr_decoder_outputs = curr_output[1]
        curr_behavior = sample[2]
        cluster_probs.append(curr_cluster_probs.squeeze())
        decoder_outputs.append(curr_decoder_outputs.squeeze())
        sample_behaviors.append(curr_behavior)
    cluster_probs = torch.stack(cluster_probs)

    # TODO: Calculate discreteness metric
    # import pdb; pdb.set_trace()

    decoder_outputs = torch.stack(decoder_outputs)
    torch.set_printoptions(sci_mode=False)
    # cluster_ids = [val.index(max(val))+1 for val in cluster_probs]

    # Calculate PCs of the 16-dim EMG
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(train_emg)
    pca1 = pca_result[:,0]
    pca2 = pca_result[:,1]
    loadings = pca.components_.T
    # loadings_pca1 = torch.Tensor(loadings[:,0])
    # loadings_pca2 = torch.Tensor(loadings[:,1])
    train_emg_mean = torch.mean(train_emg, dim=0)

    # Apply PC loading matrix to dm outputs
    decoder_outputs_dict = {} # key: mode id, value: mode output mapped to 2d space
    decoder_outputs_muscle_dict = {}
    for i in range(config.d):
        curr_behavior = i
        decoder_outputs_dict[curr_behavior] = torch.matmul(decoder_outputs[:,:,curr_behavior]-train_emg_mean, torch.Tensor(loadings))
        decoder_outputs_muscle_dict[curr_behavior] = decoder_outputs[:,:,curr_behavior]
    
    # TODO: Evaluate if falls into degeneracy
    # import pdb; pdb.set_trace()

    # Plot train EMG figure
    if plot_type == "baseline":

        plt.figure(figsize=(15, 2))
        plt.plot(pca1, label="PCA1", color="red")
        plt.plot(pca2, label="PCA2", color="blue")
        plt.title("Train EMG")
        plt.xlabel("Timstamp")
        plt.ylabel("EMG value")
        ax = plt.gca()
        ax.set_ylim([-200, 600])
        plt.legend()
        if save_fig:
            plt.savefig("figures/decoder_outputs/%s_emg.png" % model_id)
        else:
            plt.show()

        # Plot mode outputs figure
        for i in range(config.d):
            curr_behavior = i
            plt.figure(figsize=(15, 2))
            plt.plot(decoder_outputs_dict[curr_behavior][:,0].detach(), label="PCA1", color="red")
            plt.plot(decoder_outputs_dict[curr_behavior][:,1].detach(), label="PCA2", color="blue")
            plt.title("Mode #%s Decoder Output" % curr_behavior)
            plt.ylabel("EMG value")
            ax = plt.gca()
            ax.set_ylim([-200, 600])
            plt.legend()
            if save_fig:
                plt.savefig("figures/decoder_outputs/%s_decoder_mode%s.png" % (model_id, curr_behavior))
            else:
                plt.show()

        # Plot clustering probabilities figure
        for i in range(config.d):
            curr_behavior = i
            plt.figure(figsize=(15, 2)) 
            plt.plot(cluster_probs[:,curr_behavior].detach(), color="black")
            plt.title("Mode #%s Cluster Probabilities" % curr_behavior)
            plt.ylabel("Probability")
            ax = plt.gca()
            ax.set_ylim([0, 1.0])
            if save_fig:
                plt.savefig("figures/decoder_outputs/%s_clustering_mode%s.png" % (model_id, curr_behavior))
            else:
                plt.show()
    
    elif plot_type in ["behavior_average", "behavior_average_unweighted", "behavior_average_muscle"]:

        # Split up predictions into each behavior
        probs_behaviors_dict = {}
        outputs_behaviors_dict = {}
        outputs_behaviors_muscle_dict = {}
        emgs_behaviors_dict = {}
        emgs_behaviors_muscle_dict = {}
        for i in range(len(sample_behaviors)):
            curr_behavior = sample_behaviors[i]
            curr_probs = cluster_probs[i]
            curr_outputs = [decoder_outputs_dict[key][i].detach() for key in decoder_outputs_dict.keys()]
            curr_outputs_muscle = [decoder_outputs_muscle_dict[key][i].detach() for key in decoder_outputs_muscle_dict.keys()]
            curr_emgs = [pca1[i], pca2[i]]
            if curr_behavior not in probs_behaviors_dict.keys():
                probs_behaviors_dict[curr_behavior] = [curr_probs.detach()]
                outputs_behaviors_dict[curr_behavior] = [torch.stack(curr_outputs)]
                outputs_behaviors_muscle_dict[curr_behavior] = [torch.stack(curr_outputs_muscle)]
                emgs_behaviors_dict[curr_behavior] = [torch.Tensor(curr_emgs)]
                emgs_behaviors_muscle_dict[curr_behavior] = [torch.Tensor(train_emg[i])]
            else:
                probs_behaviors_dict[curr_behavior].append(curr_probs.detach())
                outputs_behaviors_dict[curr_behavior].append(torch.stack(curr_outputs))
                outputs_behaviors_muscle_dict[curr_behavior].append(torch.stack(curr_outputs_muscle))
                emgs_behaviors_dict[curr_behavior].append(torch.Tensor(curr_emgs))
                emgs_behaviors_muscle_dict[curr_behavior].append(torch.Tensor(train_emg[i]))
        
        # Calculate average over all trials across each behavior
        trial_range = 100
        for k,v in probs_behaviors_dict.items():
            num_items = len(v)
            probs_behaviors_dict[k] = torch.reshape(torch.stack(v), (num_items//trial_range, trial_range, dataset.num_modes))
            probs_behaviors_dict[k] = torch.mean(probs_behaviors_dict[k], dim=0)
        for k,v in outputs_behaviors_dict.items():
            num_items = len(v)
            outputs_behaviors_dict[k] = torch.reshape(torch.stack(v), (num_items//trial_range, trial_range, dataset.num_modes, 2))
            outputs_behaviors_dict[k] = torch.mean(outputs_behaviors_dict[k], dim=0)
        for k,v in outputs_behaviors_muscle_dict.items():
            num_items = len(v)
            outputs_behaviors_muscle_dict[k] = torch.reshape(torch.stack(v), (num_items//trial_range, trial_range, dataset.num_modes, dataset.M))
            outputs_behaviors_muscle_dict[k] = torch.mean(outputs_behaviors_muscle_dict[k], dim=0)
        for k,v in emgs_behaviors_dict.items():
            num_items = len(v)
            emgs_behaviors_dict[k] = torch.reshape(torch.stack(v), (num_items//trial_range, trial_range, 2))
            emgs_behaviors_dict[k] = torch.mean(emgs_behaviors_dict[k], dim=0)
        for k,v in emgs_behaviors_muscle_dict.items():
            num_items = len(v)
            emgs_behaviors_muscle_dict[k] = torch.reshape(torch.stack(v), (num_items//trial_range, trial_range, dataset.M))
            emgs_behaviors_muscle_dict[k] = torch.mean(emgs_behaviors_muscle_dict[k], dim=0)
        
        # Create color map and label options for multiple modes
        cmap_colors = ["yellow","green","blue","red","purple","orange","pink"]
        number_labels = ["1", "2", "3", "4", "5", "6", "7"]

        # Plot averaged decoding outputs for each behavior
        equal_scale = True
        fig, ax = plt.subplots(3,4, figsize=(14,7))
        ax2 = ax[0,0].twinx()
        ax2 = [[ax_inner.twinx() for ax_inner in ax_outer] for ax_outer in ax]
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        ax_pos = 0
        for k,v in outputs_behaviors_dict.items():
            x = range(0, trial_range)
            # Clustering probabilities
            y = torch.transpose(probs_behaviors_dict[k], 0, 1)
            # PCA1 and PCA2 for outputs
            y1 = torch.transpose(v[:,:,0], 0, 1)
            y2 = torch.transpose(v[:,:,1], 0, 1)
            # Ground truth EMG PCA
            y_1 = emgs_behaviors_dict[k][:,0]
            y_2 = emgs_behaviors_dict[k][:,1]
            # Final output PCA values
            y1_final = y*y1
            y2_final = y*y2

            # Ground truth EMG, just plotting first 2 EMG muscles
            y_1_muscle = emgs_behaviors_muscle_dict[k][:,0]
            y_2_muscle = emgs_behaviors_muscle_dict[k][:,1]
            # Predicted EMG, just plotting first 2 EMG muscle predictions
            y1_muscle = torch.transpose(outputs_behaviors_muscle_dict[k][:,:,0], 0, 1)
            y2_muscle = torch.transpose(outputs_behaviors_muscle_dict[k][:,:,1], 0, 1)

            # Plot output distributions
            for i in range(len(y1)+1):

                # Plot ground truth EMG values
                if i == 0: 
                    ax[ax_pos, i].set_title("Behavior %s, EMG" % (k), fontsize=10)
                    if plot_type == "behavior_average":
                        ax[ax_pos, i].plot(x, y_1, label="PCA1", color="red")
                        ax[ax_pos, i].plot(x, y_2, label="PCA2", color="blue")
                    elif plot_type == "behavior_average_muscle":
                        ax[ax_pos, i].plot(x, y_1_muscle, label="PCA1", color="purple")
                        ax[ax_pos, i].plot(x, y_2_muscle, label="PCA2", color="orange")
                    if equal_scale:
                        # ax[ax_pos][i].set_ylim([-250,550]) # Scale y-axis for equal comparison
                        # ax[ax_pos][i].set_ylim([-60,70]) # Scale y-axis for equal comparison
                        # ax[ax_pos][i].set_ylim([-0.7,1.3]) # Scale y-axis for equal comparison
                        # ax[ax_pos][i].set_ylim([-1.25,2.25]) # Scale y-axis for equal comparison

                        # TODO: Manual scaling for monkey dataset, weighted plot, cosyne submission
                        if plot_type in ["behavior_average_unweighted", "behavior_average"]:
                            if ax_pos == 0:
                                # ax[ax_pos][i].set_ylim([-35,70])
                                ax[ax_pos][i].set_ylim([-65,70]) # TODO: expand y-axis range to view full plots with global bias
                            elif ax_pos == 1:
                                # ax[ax_pos][i].set_ylim([-60,25])
                                ax[ax_pos][i].set_ylim([-80,35])
                            elif ax_pos == 2:
                                # ax[ax_pos][i].set_ylim([-70,70])
                                ax[ax_pos][i].set_ylim([-90,70])
                        elif plot_type == "behavior_average_muscle":
                            if ax_pos == 0:
                                ax[ax_pos][i].set_ylim([-5,150])
                            elif ax_pos == 1:
                                ax[ax_pos][i].set_ylim([-5,150])
                            elif ax_pos == 2:
                                ax[ax_pos][i].set_ylim([-5,150])
                    
                # Plot mode values
                else:
                    ax[ax_pos, i].set_title("Behavior %s, Mode %s" % (k, i-1), fontsize=10)
                    if plot_type == "behavior_average_unweighted":
                        ax[ax_pos, i].plot(x, y1[i-1], label="PCA1", color="red")
                        ax[ax_pos, i].plot(x, y2[i-1], label="PCA2", color="blue")
                    elif plot_type == "behavior_average":
                        ax[ax_pos, i].plot(x, y1_final[i-1], label="PCA1_final", color="red", linestyle='dashed')
                        ax[ax_pos, i].plot(x, y2_final[i-1], label="PCA2_final", color="blue", linestyle='dashed')
                    elif plot_type == "behavior_average_muscle":
                        ax[ax_pos, i].plot(x, y1_muscle[i-1], label="PCA1_final", color="purple", linestyle='dashed')
                        ax[ax_pos, i].plot(x, y2_muscle[i-1], label="PCA2_final", color="orange", linestyle='dashed')
                    ax2[ax_pos][i].plot(x, y[i-1], label="prob", color="black")
                    ax2[ax_pos][i].set_ylim([0,1])
                    if equal_scale:
                        # ax[ax_pos][i].set_ylim([-250,550]) # Scale y-axis for equal comparison
                        # ax[ax_pos][i].set_ylim([-60,70]) # Scale y-axis for equal comparison
                        # ax[ax_pos][i].set_ylim([-3.5,2.2]) # Scale y-axis for equal comparison
                        # ax[ax_pos][i].set_ylim([-1.25,1.5]) # Scale y-axis for equal comparison
                        
                        # TODO: Manual scaling for monkey dataset, weighted plot, cosyne submission
                        if plot_type in ["behavior_average_unweighted", "behavior_average"]:
                            if ax_pos == 0:
                                # ax[ax_pos][i].set_ylim([-35,70])
                                ax[ax_pos][i].set_ylim([-65,70]) # TODO: expand y-axis range to view full plots with global bias
                            elif ax_pos == 1:
                                # ax[ax_pos][i].set_ylim([-60,25])
                                ax[ax_pos][i].set_ylim([-80,35])
                            elif ax_pos == 2:
                                # ax[ax_pos][i].set_ylim([-70,70])
                                ax[ax_pos][i].set_ylim([-90,70])
                        elif plot_type == "behavior_average_muscle":
                            if ax_pos == 0:
                                ax[ax_pos][i].set_ylim([-5,150])
                            elif ax_pos == 1:
                                ax[ax_pos][i].set_ylim([-5,150])
                            elif ax_pos == 2:
                                ax[ax_pos][i].set_ylim([-5,150])

            ax_pos += 1
        
        if save_fig:
            if equal_scale:
                plt.savefig("figures/decoder_outputs/%s_%s_scaled.png" % (model_id, plot_type))
                # plt.savefig("figures/decoder_outputs/%s_%s_scaled.pdf" % (model_id, plot_type))
            else:
                plt.savefig("figures/decoder_outputs/%s_%s.png" % (model_id, plot_type))
        else:
            plt.show()

        

if __name__ == "__main__":

    # Read in configs and dataset
    config = load_config()
    if config.label_type == "mouse":
        dataset = Mouse_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type, seed=config.seed,
                           kmeans_cluster=config.kmeans_cluster, label_type=config.label_type,
                           remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
    else:
        dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                            behavioral_path=config.behavioral_path, num_modes=config.d, 
                            batch_size=config.b, dataset_type=config.type, seed=config.seed,
                            kmeans_cluster=config.kmeans_cluster, label_type=config.label_type,
                            remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
    
    # Print dataset statistics
    dataset_statistics(dataset=dataset, verbose=False)

    # Perform kmeans on input dataset
    run_kmeans(dataset=dataset, config=config, verbose=False)

    # Evaluate model clustering 
    # model_ids = [120]
    # model_ids = [426]
    # model_ids = [442, 443]
    # model_ids = [449]
    # model_ids = [120]
    # model_ids = [443]

    # model_ids = [484]
    # model_ids = [483]
    # model_ids = [485]
    # model_ids = [522, 523]
    model_ids = [524]

    for model_id in model_ids:
        # model_path = "checkpoints_intervals/%s.ckpt" % model_id
        if model_id in [449]:
            model_path = "checkpoints/checkpoint%s_epoch=749.ckpt" % model_id
        else:
            model_path = "checkpoints/checkpoint%s_epoch=499.ckpt" % model_id

            # model_path = "checkpoints/checkpoint%s_epoch=449.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=474.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=479.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=484.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=479.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=494.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=499.ckpt" % model_id
        num_to_print = 7800
        # num_to_print = 10000
        plot_type = "distributions"
        # plot_type = "majority"
        # plot_type = "mode_average"
        # plot_type = "confusion_matrix"
        # plot_type = "discrete_state_overlap"
        # Check clustering
        check_clustering(dataset=dataset,
                        model_path=model_path,
                        num_to_print=num_to_print,
                        config=config,
                        plot_type=plot_type,
                        model_id=model_id,
                        verbose=False)

    # Evaluate model decoding
    # model_id = 120
    # model_id = 115
    # model_id = 254
    # model_id = 259
    # model_id = 259
    # model_id = 273
    # model_id = 277
    # model_id = 260
    model_id = 423
    model_path = "checkpoints/checkpoint%s_epoch=499.ckpt" % model_id
    # plot_type = "baseline"
    # plot_type = "behavior_average"
    # plot_types = ["behavior_average", "behavior_average_unweighted"]
    # plot_types = ["behavior_average_unweighted"]
    # plot_types = ["behavior_average"]
    plot_types = ["behavior_average_muscle"]
    # Check decoding
    # model_ids = [423, 424, 425, 426, 427]
    # model_ids = [436, 437, 438, 439, 440]
    # model_ids = [447, 448, 449]
    # model_ids = [449]
    # model_ids = [488]
    # model_ids = [522, 523]
    # model_ids = [524]
    # model_ids = [536]
    # model_ids = [611, 616, 619]
    # model_ids = [621, 622, 628]
    # model_ids = [627]
    # model_ids = [639, 640, 641, 642, 643, 644, 645, 646, 647, 648]
    # model_ids = [649, 650, 651]
    # model_ids = [652, 653, 659, 660, 655, 656, 663, 664, 665, 666, 670]
    # model_ids = [663, 664, 665, 666, 669, 670]
    model_ids = [666, 669, 670]

    # model_ids = [671, 672, 673]
    # model_ids = [672]
    # model_ids = [677, 678, 679]
         
    for model_id in model_ids:
        if model_id in [449]:
            model_path = "checkpoints/checkpoint%s_epoch=749.ckpt" % model_id
        elif model_id in [448]:
            model_path = "checkpoints/checkpoint%s_epoch=599.ckpt" % model_id
        else:
            model_path = "checkpoints/checkpoint%s_epoch=499.ckpt" % model_id

            # model_path = "checkpoints/checkpoint%s_epoch=449.ckpt" % model_id
            # model_path = "checkpoints/checkpoint%s_epoch=459.ckpt" % model_id

        for plot_type in plot_types:
            sep_decoders_R2(dataset=dataset,
                            model_path=model_path,
                            config=config,
                            plot_type=plot_type,
                            model_id=model_id,
                            verbose=True)

    # Calculate full R^2 over separate models
    # If using kmeans split data, format separate datasets
    datasets = []
    if "kmeans_split" in config.m1_path:
        # If using k=6 or k=3
        if "k6" in config.m1_path:
            k = 6
        elif "k3" in config.m1_path:
            k = 3
        elif "k11" in config.m1_path:
            k = 11
        # Load datasets for each cluster
        if dataset.label_type == "mouse":
            for k in range(k):
                curr_dataset = Mouse_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                            behavioral_path=config.behavioral_path, num_modes=config.d, 
                            batch_size=config.b, dataset_type=config.type, seed=config.seed, 
                            kmeans_cluster=k, label_type=config.label_type,
                            remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
                datasets.append(curr_dataset)
        else:
            for k in range(k):
                curr_dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                            behavioral_path=config.behavioral_path, num_modes=config.d, 
                            batch_size=config.b, dataset_type=config.type, seed=config.seed, 
                            kmeans_cluster=k, label_type=config.label_type,
                            remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
                datasets.append(curr_dataset)
    # If using mode data, format separate datasets
    else:
        # If using mouse data
        if dataset.label_type == "mouse":
            for mode in ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]:
                curr_label_type = "%s_%s" % (config.label_type, mode)
                curr_dataset = Mouse_Dataset(m1_path="", emg_path="", 
                                behavioral_path="", num_modes=config.d, 
                                batch_size=config.b, dataset_type=config.type, seed=config.seed,
                                kmeans_cluster=config.kmeans_cluster, label_type=curr_label_type,
                                remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
                datasets.append(curr_dataset)
        # If using set1 data
        elif dataset.label_type == "set1":
            for mode in ["crawling", "pg", "sitting_still"]:
                curr_label_type = "%s_%s" % (config.label_type, mode)
                curr_dataset = Cage_Dataset(m1_path="", emg_path="", 
                                behavioral_path="", num_modes=config.d, 
                                batch_size=config.b, dataset_type=config.type, seed=config.seed,
                                kmeans_cluster=config.kmeans_cluster, label_type=curr_label_type,
                                remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
                datasets.append(curr_dataset)
        # If using set2 data
        else:
            for mode in ["crawl", "precision", "power"]:
                # If using B=10 dataset
                if "b10" in config.m1_path:
                    curr_m1_path = "data/set2_data/sep_modes_b10/m1_set2_t100_b10_%s.npy" % mode
                    curr_emg_path = "data/set2_data/sep_modes_b10/emg_set2_t100_b10_%s.npy" % mode
                    curr_behavioral_path = "data/set2_data/sep_modes_b10/behavioral_set2_t100_b10_%s.npy" % mode
                # If using B=1 dataset
                else:
                    curr_m1_path = "data/set2_data/m1_set2_t100_%s.npy" % mode
                    curr_emg_path = "data/set2_data/emg_set2_t100_%s.npy" % mode
                    curr_behavioral_path = "data/set2_data/behavioral_set2_t100_%s.npy" % mode
                curr_dataset = Cage_Dataset(m1_path=curr_m1_path, emg_path=curr_emg_path, 
                            behavioral_path=curr_behavioral_path, num_modes=config.d, 
                            batch_size=config.b, dataset_type=config.type, seed=config.seed,
                            kmeans_cluster=config.kmeans_cluster, label_type=config.label_type,
                            remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs)
                datasets.append(curr_dataset)

    # Calculate separate R^2 for each behavioral label in our model
    model_sep_r2_list = model_sep_R2(dataset=dataset, model_path=model_path,
                                     config=config, verbose=False)

    # Calculate full R2 value
    dataset_lengths = [len(dataset) for dataset in datasets]
    full_r2_list = full_R2_reg(datasets=datasets, verbose=False)

    # Calculate separate R^2 for each behavioral label in our model
    sep_r2_list = sep_R2_reg(dataset=dataset, verbose=False)

    # Run kmeans on points to get learned clusters
    # m1, preds = run_kmeans_M1(dataset, config)




