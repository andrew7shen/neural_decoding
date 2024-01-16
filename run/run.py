# Script to train the end-to-end decoder

# Run script using command "python3 run/run.py" in home directory
import sys
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
sys.path.append('/Users/andrewshen/Desktop/neural_decoding')

from data.data import *
from model.model import *
from TrainingModule import *

if __name__ == "__main__":
    print("Running 'run.py'...")
    
    # Define parameters
    T = 5000
    N = 10
    batch_size = 16
    epochs = 250
    lr = 0.0001
    record = True

    # Load in dataset
    dataset = M1Dataset_Toy(num_samples=T, num_neurons=N, batch_size=batch_size)

    # Define model
    model = ClusterModel(N)

    # Define TrainingModule
    model = TrainingModule(model, lr, record)

    # Define trainer
    if record:
        wandb_logger = WandbLogger(project="neural_decoding")
        trainer = Trainer(max_epochs=epochs, logger=wandb_logger)
    else:
        trainer = Trainer(max_epochs=epochs)

    # Fit the model
    trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())

    wandb.finish()

    print("Finished 'run.py'!")
