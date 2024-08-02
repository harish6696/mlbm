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

from math import ceil
from torch.utils.data import Dataset, DataLoader
import torch
from validator import GridValidator

import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.data = torch.load("/home/jwinter/Development/TorchLBM/cases/dataset_creation/velocity_karman.pt")
        self.data = self.data[100:]
        self.data_x = self.data[:-1]
        self.data_y = self.data[1:]

        self.num_elements = len(self.data_x)
        self.normalization_mean = self.data_x.mean()
        self.normalization_std = self.data_x.std()
        self.data_x = ( self.data_x - self.normalization_mean ) / self.normalization_std
        self.data_y = ( self.data_y - self.normalization_mean ) / self.normalization_std

        new_indices = torch.randperm(self.num_elements)
        self.data_x = self.data_x[new_indices]
        self.data_y = self.data_y[new_indices]


        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        input_tensor = self.data_x[idx,...].unsqueeze(0)
        output_tensor = self.data_y[idx,...].unsqueeze(0)
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            output_tensor = self.target_transform(output_tensor)
        return input_tensor, output_tensor

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):   
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="LBM_fno")
    # initialize monitoring
    initialize_mlflow(
        experiment_name="LBM_FNO",
        experiment_desc="training an FNO model for the LBM problem",
        run_name="LBM FNO training",
        run_desc="training FNO for LBM",
        user_name="Saydemir",
        mode="offline",
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

    dataset = CustomDataset()
    dataloader = DataLoader(dataset=dataset, batch_size = cfg.training.batch_size, shuffle=True)
    
    ckpt_args = {
        "path": f"./lbm_checkpoints",
        "models": model,
    }
    # validator = GridValidator(out_dir=ckpt_args["path"] + "/outputs/validators", loss_fun=MSELoss())
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    start_value = None

    for batch in dataloader:
        start_value = batch[0][0].unsqueeze(0)
        break


    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)
    print(start_value.shape)

    result = start_value
    for i in range(200):
        result = forward_eval(result.to(dist.device))
        plt.imshow(torch.transpose(result.squeeze(dim=[0,1]).clone(), 0, 1).cpu().numpy())
        time_string = "{:03d}".format(i)
        plt.savefig(f"test_{time_string}.png")
        plt.close()


    # # calculate steps per pseudo epoch
    # steps_per_pseudo_epoch = ceil(
    #     cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    # )
    # validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    # log_args = {
    #     "name_space": "train",
    #     "num_mini_batch": steps_per_pseudo_epoch,
    #     "epoch_alert_freq": 1,
    # }
    # if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
    #     log.warning(
    #         f"increased pseudo_epoch_sample_size to multiple of \
    #                   batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
    #     )
    # if cfg.validation.sample_size % cfg.training.batch_size != 0:
    #     log.warning(
    #         f"increased validation sample size to multiple of \
    #                   batch size: {validation_iters*cfg.training.batch_size}"
    #     )

    # if loaded_pseudo_epoch == 0:
    #     log.success("Training started...")
    # else:
    #     log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    # for pseudo_epoch in range(
    #     max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    # ):
    #     # Wrap epoch in launch logger for console / MLFlow logs
    #     with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
    #         for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
    #             loss = forward_train(batch[0].to(dist.device), batch[1].to(dist.device))
    #             logger.log_minibatch({"loss": loss.detach()})
    #         logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

    #     # save checkpoint
    #     if pseudo_epoch % cfg.training.rec_results_freq == 0:
    #         save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

    #     # validation step
    #     if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
    #         with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
    #             total_loss = 0.0
    #             for _, batch in zip(range(validation_iters), dataloader):
                   
    #                 val_loss = validator.compare(
    #                     batch[0].to(dist.device),
    #                     batch[1].to(dist.device),
    #                     forward_eval(batch[0].to(dist.device)),
    #                     pseudo_epoch,
    #                 )
    #                 total_loss += val_loss
    #             logger.log_epoch({"Validation error": total_loss / validation_iters})

    #     # update learning rate
    #     if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
    #         scheduler.step()

    # save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    # log.success("Training completed *yay*")

if __name__ == "__main__":
    main()
