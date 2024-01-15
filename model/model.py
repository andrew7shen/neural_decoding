# Script that holds all the model files for neural decoding

# Import packages
import torch.nn as nn

# Clustering model
class ClusterModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        # x = self.softmax(x)
        return x