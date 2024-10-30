# Script for Cosyne Abstract submission

# Import packages
import matplotlib.pyplot as plt
import numpy as np

### Monkey validation performance figure

# val R^2 values for each model
our_model_discrete_k3 = [0.5348]
our_model_discrete_k6 = [0.5657]
sep_modes = [0.5246]
single_decoder = [0.4282]
sep_clusters = [0.5258]
neural_network = [0.5852]

# Choose what results to create bar graph with
models = [our_model_discrete_k3, our_model_discrete_k6, sep_modes, single_decoder, sep_clusters, neural_network]
model_names = ['Our Model\nK=3', 'Our Model\nK=6','Manually Annotated\nModes','Single\nDecoder','Kmeans plus\nRegression', 'Neural Network']
title = "Monkey Dataset Decoding Performance"

# Plot graph
verbose = False
val_r2s = [model[0] for model in models]
n = len(models)
x_axis = np.arange(n)
width = 0.75
# plt.figure(figsize=(7,6))
bar_plot = plt.bar(x_axis, val_r2s, width=width)
bar_plot[0].set_color('r')
bar_plot[1].set_color('r')
plt.ylim(0.2, 0.7)
plt.xticks(x_axis, model_names, fontsize=10, rotation=45)
plt.xlabel("Model Type", fontsize=12)
plt.ylabel("Valdation R^2 Value", fontsize=12)
plt.title(title, fontsize=15)
plt.tight_layout()
if verbose:
    plt.show()
else:
    # plt.savefig("figures/cosyne_figures/monkey_results.png")
    plt.savefig("figures/cosyne_figures/monkey_results.pdf")
plt.clf()





### Mouse validation performance figure

# val R^2 values for each model
our_model_discrete_k8 = [0.45572]
our_model_discrete_k11 = [0.4713]
sep_modes = [0.4356]
single_decoder = [0.3139]
sep_clusters = [0.3807]
neural_network = [0.4903]

# Choose what results to create bar graph with
models = [our_model_discrete_k8, our_model_discrete_k11, sep_modes, single_decoder, sep_clusters, neural_network]
model_names = ['Our Model\nK=8', 'Our Model\nK=11', 'Manually Annotated\nModes','Single\nDecoder','Kmeans plus\nRegression', 'Neural Network']

title = "Mouse Dataset Decoding Performance"

# Plot graph
verbose = False
val_r2s = [model[0] for model in models]
n = len(models)
x_axis = np.arange(n)
width = 0.75
# plt.figure(figsize=(7,6))
bar_plot = plt.bar(x_axis, val_r2s, width=width)
bar_plot[0].set_color('r')
bar_plot[1].set_color('r')
plt.ylim(0.2, 0.6)
plt.xticks(x_axis, model_names, fontsize=10, rotation=45)
plt.xlabel("Model Type", fontsize=12)
plt.ylabel("Valdation R^2 Value", fontsize=12)
plt.title(title, fontsize=15)
plt.tight_layout()
if verbose:
    plt.show()
else:
    # plt.savefig("figures/cosyne_figures/mouse_results.png")
    plt.savefig("figures/cosyne_figures/mouse_results.pdf")
