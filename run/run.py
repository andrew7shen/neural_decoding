# Script to train the end-to-end decoder

"""
Run script using command "python3 run/run.py configs/configs_cage.yaml" in home directory
New command for expanded trial ranges: "python3 run/run.py configs/t100_configs/configs_cage_t100.yaml"
"""
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
    if config.label_type == "mouse":
        dataset = Mouse_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                           behavioral_path=config.behavioral_path, num_modes=config.d, 
                           batch_size=config.b, dataset_type=config.type, seed=config.seed,
                           kmeans_cluster=config.kmeans_cluster, label_type=config.label_type,
                           remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs,
                           mean_centering=config.mean_centering)
    else:
        dataset = Cage_Dataset(m1_path=config.m1_path, emg_path=config.emg_path, 
                            behavioral_path=config.behavioral_path, num_modes=config.d, 
                            batch_size=config.b, dataset_type=config.type, seed=config.seed,
                            kmeans_cluster=config.kmeans_cluster, label_type=config.label_type,
                            remove_zeros=config.remove_zeros, scale_outputs=config.scale_outputs,
                           mean_centering=config.mean_centering) # TODO: Added for kmeans split runs, need to create flexible kwargs
        # dataset = M1_EMG_Dataset_Toy(num_samples=T, num_neurons=N, num_muscles=M, num_modes=d, batch_size=b, dataset_type=type)

    # Define model
    if config.model == "decoder":
        model = DecoderModel(dataset.N, dataset.M, config.d)
    elif config.model == "combined":
        model = CombinedModel(input_dim=dataset.N,
                              hidden_dim=config.hidden_dim,
                              output_dim=dataset.M,
                              num_modes=config.d, 
                              ev=config.ev,
                              cluster_model_type=config.cluster_model_type,
                              decoder_model_type=config.decoder_model_type,
                              combined_model_type=config.combined_model_type)
    
    # Define Training Module
    model = TrainingModule(model=model,
                           lr=config.lr,
                           weight_decay=config.weight_decay,
                           record=config.record,
                           type=config.type,
                           temperature=config.temperature,
                           anneal_temperature=config.anneal_temperature,
                           num_epochs=config.epochs,
                           end_temperature=config.end_temperature,
                           lambda_val=config.lambda_val,
                           l1_lambda_val=config.l1_lambda_val,
                           overlap_lambda_val=config.overlap_lambda_val)
    
    # Set initial decoder weights
    weights = []
    if config.remove_zeros:
        label_type = f"{dataset.label_type}_removezeros"
    else:
        label_type = dataset.label_type
    if config.set_decoder_weights != "none":
        if "b10" in config.m1_path:
            for i in range(config.d):
                if config.set_decoder_weights == "kmeans":
                    weights.append(nn.Parameter(torch.Tensor(np.load("data/%s_data/decoder_weights/k%s_b10/weights_%s.npy" % (label_type, config.d, i)))))
                elif config.set_decoder_weights == "random":
                    weights.append(nn.Parameter(torch.Tensor(np.load("data/%s_data/decoder_weights/k%s_b10_random/weights_%s.npy" % (label_type, config.d, i)))))
        else:
            for i in range(config.d):
                if config.set_decoder_weights == "kmeans":
                    weights.append(nn.Parameter(torch.Tensor(np.load("data/%s_data/decoder_weights/k%s/weights_%s.npy" % (label_type, config.d, i)))))
                elif config.set_decoder_weights == "random":
                    weights.append(nn.Parameter(torch.Tensor(np.load("data/%s_data/decoder_weights/k%s_random/weights_%s.npy" % (label_type, config.d, i)))))
        with torch.no_grad():
            for i in range(config.d):
                model.model.dm.linears[i].weight = weights[i]

    # Define model checkpoints
    if config.run_id == "none":
        callback_filename = 'checkpoint_{epoch}_%s' % config.d
    else:
        callback_filename = 'checkpoint%s_{epoch}' % config.run_id
    # Define when to save checkpoints
    if config.save_epochs == "none":
        save_callback = ModelCheckpoint(dirpath = config.save_path, filename=callback_filename)
    else:
        save_callback = ModelCheckpoint(dirpath = config.save_path, filename=callback_filename, save_top_k = -1, every_n_epochs = config.save_epochs)

    # Define trainer
    if config.record:
        wandb_logger = WandbLogger(project="neural_decoding")
        trainer = Trainer(max_epochs=config.epochs, logger=wandb_logger, callbacks=[Callback(dataset), save_callback])
    else:
        trainer = Trainer(max_epochs=config.epochs, callbacks=[Callback(dataset), save_callback])

    # Fit the model
    trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())
    # TODO: temp for continued training
    # trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader(), ckpt_path="/Users/andrewshen/Github_Repos/neural_decoding/checkpoints/checkpoint483_epoch=474.ckpt")

    wandb.finish()

    print("Finished 'run.py'!")
