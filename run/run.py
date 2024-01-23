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
    d = 3  # num_modes
    b = 8
    type = "emg"
    epochs = 500
    lr = 0.0001
    record = True
    m1_path = "data/set2_data/m1_set2.npy"
    emg_path = "data/set2_data/emg_set2.npy"
    behavioral_path = "data/set2_data/behavioral_set2.npy"

    # Load in dataset
    dataset = Cage_Dataset(m1_path=m1_path, emg_path=emg_path, behavioral_path=behavioral_path, num_modes=d, batch_size=b, dataset_type=type)
    # dataset = M1_EMG_Dataset_Toy(num_samples=T, num_neurons=N, num_muscles=M, num_modes=d, batch_size=b, dataset_type=type)

    # Define model
    # model = CombinedModel(N, M, d)
    model = CombinedModel(dataset.N, dataset.M, d)
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
