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
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow
from omegaconf import DictConfig

from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
import hydra
from pathlib import Path

from math import ceil
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from mlbm.validator import GridValidator, VTKvalidator, DensityVTKvalidator
from mlbm.datasets.karman_street_dataset import KarmanStreetDataset, MixedReKarmanStreetDataset, PorousDataset

from torchvision.transforms import Normalize

@hydra.main(version_base="1.3", config_path=".", config_name="config_porous_density.yaml")
def main(cfg: DictConfig):   
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere


    result_folder = Path(cfg.output.dir).absolute()

    # initialize monitoring
    log = PythonLogger(name="LBM_fno")
    # initialize monitoring
    initialize_mlflow(
        experiment_name="LBM_FNO",
        experiment_desc="training an FNO model for the LBM problem",
        run_name="LBM FNO training",
        run_desc="training FNO for LBM",
        user_name="winterjo",
        mode="offline",
        tracking_location=str(result_folder.joinpath(f"mlflow_output_{dist.rank}")),
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)

    loss_fun = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )

    resolution_list = [32]
    train_dataloaders = []
    val_dataloaders = []

    for resolution in resolution_list:
        # dataset = KarmanStreetDataset(cfg.data.file)
        dataset = PorousDataset(base_folder=cfg.data.base_folder, resolution=resolution, field_name=cfg.data.field_name, num_channels=cfg.data.num_channels, data_y_mean=cfg.data.normalization_mean, data_y_std=cfg.data.normalization_std)
        dataset_train, dataset_val = random_split(
            dataset, [0.75, 0.25], generator=torch.Generator().manual_seed(42)
        )
        train_dataloaders.append(DataLoader(dataset=dataset_train, batch_size = cfg.training.batch_size, shuffle=True))
        val_dataloaders.append(DataLoader(dataset=dataset_val, batch_size = cfg.training.batch_size, shuffle=True))
    
    ckpt_args = {
        "path": str(result_folder),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    if cfg.data.field_name == "density":
        validator = DensityVTKvalidator(loss_fun=MSELoss(), data_y_mean=dataset.data_y_mean, data_y_std=dataset.data_y_std, out_dir=ckpt_args["path"] + "/validators")
    else:
        validator = VTKvalidator(loss_fun=MSELoss(), data_y_mean=dataset.data_y_mean, data_y_std=dataset.data_y_std, out_dir=ckpt_args["path"] + "/validators")
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
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
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for idx, resolution in enumerate(resolution_list):
                for _, batch in zip(range(steps_per_pseudo_epoch), train_dataloaders[idx]):
                    loss = forward_train(batch[0].to(dist.device), batch[1].to(dist.device))
                    logger.log_minibatch({f"loss_{resolution}": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for idx, resolution in enumerate(resolution_list):
                    for _, batch in zip(range(validation_iters), val_dataloaders[idx]):
                        val_loss = validator.compare(
                            batch[0].to(dist.device),
                            batch[1].to(dist.device),
                            forward_eval(batch[0].to(dist.device)),
                            pseudo_epoch,
                            resolution,
                        )
                        total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / validation_iters})

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

if __name__ == "__main__":
    main()
