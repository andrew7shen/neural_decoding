# Script to perform EDA on data

# Import packages
import torch
import numpy as np
import matplotlib.pyplot as plt

# Plot range of values in M1 and EMG
m1_path = "data/set2_data/m1_set2_t100.npy"
emg_path = "data/set2_data/emg_set2_t100.npy"
m1 = torch.Tensor(np.load(m1_path))
emg = torch.Tensor(np.load(emg_path))
m1_list = m1.view(m1.shape[0]*m1.shape[1]*m1.shape[2]).tolist()
emg_list = emg.view(emg.shape[0]*emg.shape[1]*emg.shape[2]).tolist()
plt.hist(m1_list)
plt.title("M1 values")
# plt.savefig("figures/m1_distribution.png")
plt.show()
plt.hist(emg_list)
plt.title("EMG values")
# plt.savefig("figures/emg_distribution.png")
plt.show()
