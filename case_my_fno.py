# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# from lightning import Trainer
from modulus.models.mlp import FullyConnected
from modulus.models.fno import FNO
from modulus.models.afno import AFNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow
from omegaconf import DictConfig, OmegaConf

from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
import hydra
from pathlib import Path

from math import ceil
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from validator import GridValidator
from datasets.karman_street_dataset import KarmanStreetDataset, MixedReKarmanStreetDataset, SingleReKarmanStreetDataset

from torchvision.transforms import Normalize

import os
import datetime

#using 'velocity' to train the model.
@hydra.main(version_base="1.3", config_path="./conf", config_name="config.yaml")
def main(cfg: DictConfig):   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_id)

    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")


    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    result_folder = Path(cfg.output.dir).absolute()
    #print(OmegaConf.to_yaml(cfg))
    # initialize monitoring
    log = PythonLogger(name="LBM_fno")
    # initialize monitoring
    initialize_mlflow(
        experiment_name="LBM_FNO",
        experiment_desc="training an FNO model for the LBM problem for dataset with Re=200 and time duration of 2.6 seconds",
        run_name="LBM FNO training",
        run_desc="training FNO for LBM",
        user_name="hr",
        mode="online",
        tracking_location=str(result_folder.joinpath(f"mlflow_output_{dist.rank}")),
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,           # 2 for velocity
        out_channels=cfg.arch.decoder.out_features,     # 2 for velocity
        decoder_layers=cfg.arch.decoder.layers,         # 1 
        decoder_layer_size=cfg.arch.decoder.layer_size, # 128
        dimension=cfg.arch.fno.dimension,               # 2
        latent_channels=cfg.arch.fno.latent_channels,   # 32
        num_fno_layers=cfg.arch.fno.fno_layers,         # 5
        num_fno_modes=cfg.arch.fno.fno_modes,           # 40 for velocity and 100 for density
        padding=cfg.arch.fno.padding,                   # 9
    ).to(dist.device)

    #activation function used is "Gelu" for encoder and "Silu" for decoder.

    loss_fun = MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )

    transform = Normalize(mean=cfg.data.normalization_mean, std=cfg.data.normalization_std)
    dataset_train = SingleReKarmanStreetDataset(base_folder=cfg.data.base_folder, field_name=cfg.data.field_name, num_channels=cfg.data.num_channels, transform=transform, target_transform=transform)
    #dataset_train.__dict__.keys() = dict_keys(['data_x', 'data_y', 'num_elements', 'transform', 'target_transform'])
    dataset_train, dataset_val = random_split(
        dataset_train, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
    ) #len(dataset_val.__dict__['indices'])
    train_dataloader = DataLoader(dataset=dataset_train, batch_size = cfg.train.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=dataset_val, batch_size = cfg.train.training.batch_size, shuffle=True)
    
    ckpt_args = {
        #add the date and time to the results folder to create the path: 
        "path": str(result_folder.joinpath(f"{cfg.data.field_name}_{formatted_datetime}")),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    validator = GridValidator(out_dir=ckpt_args["path"] + "/validators", loss_fun=MSELoss(), num_channels=cfg.data.num_channels)
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(cfg.train.training.pseudo_epoch_sample_size / cfg.train.training.batch_size)
    #steps_per_pseudo_epoch = 2048/16 = 128. 128 times the training loop is executed for each pseudo epoch.

    validation_iters = ceil(cfg.train.validation.sample_size / cfg.train.training.batch_size) #validation_iters = 256/16 = 16.
    
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }

    if cfg.train.training.pseudo_epoch_sample_size % cfg.train.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch*cfg.train.training.batch_size}"
        )
    if cfg.train.validation.sample_size % cfg.train.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.train.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target): #invars.shape= [16,2,512,256]; target.shape= [16,2,512,256].
        pred = model(invars)
        loss = loss_fun(pred, target) #loss function is L1Loss (which is a scalar)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.train.training.max_pseudo_epochs + 1
    ):
        # Training loop
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for i, batch in zip(range(steps_per_pseudo_epoch), train_dataloader):
                loss = forward_train(batch[0].to(dist.device), batch[1].to(dist.device)) #batch[0].shape [16,2,512,256]; batch[1].shape [16,2,512,256]
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # save checkpoint
        if pseudo_epoch % cfg.train.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.train.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                #################  Note  #######################
                # validation_iters = 16 (maximum)
                # len(val_dataloader.__dict__['dataset'].__dict__['indices']) = 414, i.e. total of 414 timestep information is available for validation.
                # 414/16 = 25.875, so the loop will run for only "16" iterations and terminate.  "TO BE FIXED"
                for step, batch in zip(range(validation_iters), val_dataloader):
                    val_loss = validator.compare(
                        invar=batch[0].to(dist.device),
                        target=batch[1].to(dist.device),
                        prediction=forward_eval(batch[0].to(dist.device)),
                        step=pseudo_epoch,
                    )
                    #val_loss = validator.compute_only_loss(target=batch[1].to(dist.device)
                    #                                       ,prediction=forward_eval(batch[0].to(dist.device)))
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / step}) #should be divided by the final step number

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    save_checkpoint(**ckpt_args, epoch=cfg.train.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

if __name__ == "__main__":
    main()
