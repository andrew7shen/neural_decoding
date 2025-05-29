import numpy as np
import matplotlib.pyplot as plt

# Parameters to adjust
t_single = np.linspace(0, 2 * np.pi, 300) # number of timepoints in each section
freq1 = 5
freq2 = 15
# TODO: Try different frequencies
# freq1 = 5
# freq2 = 30
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
    # n1 = 3 * mag * np.sin(freq1 * t_single) + mean # TODO: Try scaling magnitude by 3
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
# plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
# inputs
ax1.plot(t_all, n1_all, label="Neuron 1")
ax1.plot(t_all, n2_all, label="Neuron 2")
ax1.set_ylabel("Input Activity")
ax1.set_title("Input")
ax1.legend()
# output
ax2.plot(t_all, output_all, label="Weighted Output", color="purple")
ax2.set_xlabel("Time")
ax2.set_ylabel("Output Activity")
ax2.set_title("Output")
# Trial separators & weight annotations
for i, (w1, w2) in enumerate(w_pairs):
    trial_start = i * 2 * np.pi
    trial_mid = trial_start + np.pi
    if i > 0:
        ax1.axvline(x=trial_start, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=trial_start, color="gray", linestyle="--", alpha=0.5)
    # Annotate weights
    ax2.text(trial_mid, max(output_all)*0.8, f"w1={w1:.1f}, w2={w2:.1f}",
             ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.8))
plt.tight_layout()
plt.show()