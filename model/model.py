# Script that holds all the model files for neural decoding

# Import packages
import torch.nn as nn
import torch

# Clustering model
class ClusterModel(nn.Module):

    """
    input_dim: N
    num_modes: d
    """

    def __init__(self, input_dim, num_modes, temperature):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, 1) for i in range(num_modes)])
        self.softmax = nn.Softmax(dim=2)
        self.temperature = temperature

    def forward(self, x):
        x_d = []
        for linear in self.linears:
            x_d.append(linear(x))
        x = torch.stack(x_d, 2)
        x = x/self.temperature
        x = self.softmax(x) 
        return x
    
    
# Decoding model
class DecoderModel(nn.Module):

    """
    input_dim: N
    output_dim: M
    num_modes: d
    """
    
    def __init__(self, input_dim, output_dim, num_modes):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(num_modes)])
        # self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x_d = []
        for linear in self.linears:
            x_d.append(linear(x))
        x = torch.stack(x_d, 2)
        # x = self.leaky_relu(x)
        return x


class CombinedModel(nn.Module):

    def __init__(self, input_dim, output_dim, num_modes, temperature, ev):
        super(CombinedModel, self).__init__()
        self.cm = ClusterModel(input_dim, num_modes, temperature)
        self.dm = DecoderModel(input_dim, output_dim, num_modes)
        self.ev = ev

    def forward(self, x):
        x1 = self.cm(x)
        x2 = self.dm(x)
        output = torch.sum(x1 * x2, dim=-1)

        # Return softmax outputs if mode is "eval"
        if self.ev == True:
            return x1
        return output
    


"""
Example input/outputs for models

T = 5000
N = 10
M = 3
d = 2

ClusterModel:[T,N] --> *[N,1,d] --> [T,1,d]
            [5000,10] --> *[10,1,2] --> [5000,1,2]

DecoderModel:[T,N] --> *[N,M,d] --> [T,M,d]
            [5000,10] --> *[10,3,2] --> [5000,3,2]

CombinedModel: [T,1,d] dot [T,M,d] --> [T,M]
            [5000,1,2] dot [5000,3,2] --> [5000,3]

"""
