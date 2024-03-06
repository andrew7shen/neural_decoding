# Script to train the end-to-end decoder

# Run script using command "python3 run/run.py configs/configs_cage.yaml" in home directory
import sys
import wandb
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
cwd = os.getcwd()
sys.path.append(cwd)

from data.data import *
from model.model import *
from TrainingModule import *
from utils.constants import *

if __name__ == "__main__":
    print("Running 'run.py'...")

    # Load configs
    config = load_config()

    # Load in dataset
    dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type, seed=config.seed)
    # dataset = M1_EMG_Dataset_Toy(num_samples=T, num_neurons=N, num_muscles=M, num_modes=d, batch_size=b, dataset_type=type)

    # Define model
    if config.model == "decoder":
        model = DecoderModel(dataset.N, dataset.M, config.d)
    elif config.model == "combined":
        model = CombinedModel(input_dim=dataset.N,
                              output_dim=dataset.M,
                              num_modes=config.d, 
                              temperature=config.temperature,
                              ev=config.ev)
    model = TrainingModule(model, config.lr, config.record, config.type)

    # Define model checkpoints
    save_callback = ModelCheckpoint(dirpath = config.save_path, filename='checkpoint_{epoch}_%s' % config.d)

    # Define trainer
    if config.record:
        wandb_logger = WandbLogger(project="neural_decoding")
        trainer = Trainer(max_epochs=config.epochs, logger=wandb_logger, callbacks=[Callback(dataset), save_callback])
    else:
        trainer = Trainer(max_epochs=config.epochs, callbacks=[Callback(dataset), save_callback])

    # Fit the model
    trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())

    wandb.finish()

    print("Finished 'run.py'!")
