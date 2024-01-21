# Script to train the end-to-end decoder

# Run script using command "python3 run/run.py" in home directory
import sys
import wandb
import pytorch_lightning as pl
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
    M = 3
    d = 2  # num_modes
    b = 16
    type = "emg"
    epochs = 100
    lr = 0.001
    record = False
    

    # Load in dataset
    dataset = M1_EMG_Dataset_Toy(num_samples=T, num_neurons=N, num_muscles=M, batch_size=b, dataset_type=type)

    # Define model
    model = CombinedModel(N, M, d)
    model = TrainingModule(model, lr, record, type)

    # Define trainer
    if record:
        wandb_logger = WandbLogger(project="neural_decoding")
        trainer = Trainer(max_epochs=epochs, logger=wandb_logger, callbacks=Callback(dataset))
    else:
        trainer = Trainer(max_epochs=epochs, callbacks=Callback(dataset))

    # Fit the model
    trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())

    wandb.finish()

    print("Finished 'run.py'!")
