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

    def __init__(self, input_dim, hidden_dim, num_modes, cluster_model_type):
        super().__init__()

        self.cluster_model_type = cluster_model_type

        # METHOD #1: Original linear method
        if self.cluster_model_type == "method1":
            self.linears = nn.ModuleList([nn.Linear(input_dim, 1) for i in range(num_modes)])
        
        # METHOD #2: Explore non-linearities
        if self.cluster_model_type == "method2":
            self.ffnns = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                  nn.Tanh(),
                                                  nn.Linear(hidden_dim, 1))
                                                  for i in range(num_modes)])
            
        # METHOD #3: Explore non-linearity into linears
        if self.cluster_model_type == "method3":
            self.single_ffnn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, input_dim))
        
        # METHOD #2: Initialize all ffnns to same initial model weights
        # if cluster_model_type == "method2":
        #     ffnn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        #     state_dict = ffnn.state_dict()
        #     for net in self.ffnns:
        #         net.load_state_dict(state_dict)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, temperature):
        x_d = []
        
        # METHOD #1: Original linear method
        if self.cluster_model_type == "method1":
            for linear in self.linears:
                x_d.append(linear(x))

        # METHOD #2: Explore non-linearities
        if self.cluster_model_type == "method2":
            for ffnn in self.ffnns:
                x_d.append(ffnn(x))

        # METHOD #3: Explore non-linearity into linears
        if self.cluster_model_type == "method3":
            x = self.single_ffnn(x)
            for linear in self.linears:
                x_d.append(linear(x))

        x = torch.stack(x_d, 2)

        # Scale by temperature
        # import pdb; pdb.set_trace()
        x = x/temperature
        x = self.softmax(x) 
        return x
    
    
# Decoding model
class DecoderModel(nn.Module):

    """
    input_dim: N
    output_dim: M
    num_modes: d
    """
    
    def __init__(self, input_dim, output_dim, num_modes, decoder_model_type):
        super().__init__()
        self.decoder_model_type = decoder_model_type
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(num_modes)])
        # self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x_d = []
        for linear in self.linears:
            x_d.append(linear(x))
        x = torch.stack(x_d, 2)
        # x = self.leaky_relu(x)

        # Determine whether to perform output scaling experiment
        if self.decoder_model_type == "relu0.1":
            x = torch.tanh(x)
            x = nn.LeakyReLU(0.1)(x)

        elif self.decoder_model_type == "relu0.01":
            x = torch.tanh(x)
            x = nn.LeakyReLU(0.01)(x)

        elif self.decoder_model_type == "onlyrelu0.01":
            x = nn.LeakyReLU(0.01)(x)

        elif self.decoder_model_type == "onlytanh":
            x = torch.tanh(x)

        # Leaky Relu
        elif "leakyrelu" in self.decoder_model_type:
            slope_val = float(self.decoder_model_type.split("_")[-1])
            x = nn.LeakyReLU(slope_val)(x)
        
        else:
            scale_outputs = False
            if scale_outputs:
                x = torch.tanh(x)
                x = nn.LeakyReLU(0.01)(x)

        return x


class CombinedModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_modes, ev, cluster_model_type, decoder_model_type, combined_model_type):
        super(CombinedModel, self).__init__()
        self.cm = ClusterModel(input_dim, hidden_dim, num_modes, cluster_model_type)
        self.dm = DecoderModel(input_dim, output_dim, num_modes, decoder_model_type)
        self.ev = ev
        self.counter = 0
        self.combined_model_type = combined_model_type
        # Trainable scale vector
        # TODO: Try scaling vector with initialization
        if "init" in self.combined_model_type:
            # Initializing scalevector as max of each of the 16 muscle dimensions (*Note: this is unique to monkey dataset)
            init_vector = torch.Tensor([ 69.0413,  95.1343, 121.0038,  31.5815,  23.6285,  73.8865, 180.4965, 
                                        105.6594, 126.7641,  43.9303,  51.0514,  42.1889,  78.7334,  67.2236, 
                                        96.1045,  76.3258])
            self.scale_vector = nn.Parameter(init_vector)
        # TODO: Try scaling vector without initialization
        elif "scalevector" in self.combined_model_type:
            self.scale_vector = nn.Parameter(torch.randn(output_dim))
        # Trainable bias vector
        if "scalevector" in self.combined_model_type:
            self.bias_vector = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, temperature):
        x1 = self.cm(x, temperature)
        x2 = self.dm(x)
        output = torch.sum(x1 * x2, dim=-1)
        self.counter +=1

        # TODO: Try scaling vector
        if self.combined_model_type in ["scalevector", "scalevector_init"]:
            output = output*self.scale_vector

        # TODO: Try scaling vector with bias term
        if "bias" in self.combined_model_type:
            output = output*self.scale_vector + self.bias_vector

        # Return clustering and decoding outputs if mode is "eval"
        if self.ev == True:
            return [x1, x2]
        # return output
        

        # TODO: Return clustering probs and final output
        return [x1, output]
    


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
