import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class LSTMDecoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, learning_rate=1e-3):
        super(LSTMDecoder, self).__init__()
        self.save_hyperparameters()
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Loss function (assuming classification; change as needed)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        
        # Use the hidden state of the last time step
        out = out[:, -1, :]
        
        # Final output layer
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        print(f"Train loss: {loss}")
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     val_loss = self.criterion(y_hat, y)
    #     self.log("val_loss", val_loss, prog_bar=True)
    #     return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
# Dummy data
input_dim = 95    # Number of features in the input
seq_len = 5       # Sequence length
hidden_dim = 64   # Number of hidden units in LSTM
output_dim = 16    # Number of classes (for classification)
batch_size = 32

# Create synthetic dataset
x_data = torch.rand(1000, seq_len, input_dim)
y_data = torch.randint(0, output_dim, (1000,))
dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = LSTMDecoder(input_dim, hidden_dim, output_dim)

import pdb; pdb.set_trace()

# Train
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader)