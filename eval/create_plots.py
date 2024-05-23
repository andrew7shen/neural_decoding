
# Script to create plots for lab presentation

# Import packages
import matplotlib.pyplot as plt
import numpy as np

# Create bar graphs
# train and val R^2 values for each model
our_model = [0.5740, 0.5449]
sep_modes = [0.5788, 0.5246]
single_decoder = [0.4731, 0.4282]
sep_clusters = [0.6003, 0.5258]
our_model_b10 = [0.7593, 0.6402]

our_model_nonlinear = [0.6868, 0.5773]
our_model_b10_nonlinear = [0.8758, 0.6042]
our_model_b10_nonlinear_reg = [0.8093, 0.6622]

# Choose what results to create bar graph with
# models = [our_model, sep_modes, single_decoder, sep_clusters, our_model_b10]
# model_names = ['Model','Separate\nModes','Single\nDecoder','Separate\nClusters', 'Model\n(B=10)']
# title = "Decoding Performance of Different Models"
models = [our_model, our_model_nonlinear, our_model_b10, our_model_b10_nonlinear, our_model_b10_nonlinear_reg]
model_names = ['Model','Model\nNonlinear','Model\n(B=10)','Model\nNonlinear\n(B=10)', 'Model\nNonlinear\nRegularized\n(B=10)']
title = "Decoding Performance of Nonlinear Models"

# Plot graph
train_r2s = [model[0] for model in models]
val_r2s = [model[1] for model in models]
n = len(models)
x_axis = np.arange(n)
width = 1
# plt.figure(figsize=(7,6))
plt.figure(figsize=(11,11))
# plt.bar(x_axis, train_r2s, width=width, edgecolor = 'black', label="Train")
# plt.bar(x_axis+width, val_r2s, width=width, edgecolor = 'black', label="Val")
plt.bar(x_axis, val_r2s, width=width, edgecolor = 'black')
# plt.xticks(x_axis + width/2, model_names)
plt.ylim(0.2, 0.7)
plt.xticks(x_axis, model_names, fontsize=14)
plt.yticks(fontsize=14)
# plt.xlabel("Model Type", fontsize=12)
# plt.ylabel("Valdation R^2 Value", fontsize=12)
# plt.title(title, fontsize=15)
plt.xlabel("Model Type", fontsize=16)
plt.ylabel("Valdation R^2 Value", fontsize=16)
plt.title(title, fontsize=21)
# plt.legend(loc="upper center")
plt.show()
