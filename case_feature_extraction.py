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
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger
from modulus import Module
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
from custom_karman_street_dataset import CustomKarmanStreetDataset

from torchvision.transforms import Normalize

import os
import shutil
import wandb

@hydra.main(version_base="1.3", config_path="./conf", config_name="config.yaml")
def main(cfg: DictConfig):   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_id)

    formatted_datetime = cfg.output.timestamp

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="LBM_fno")

    LaunchLogger.initialize()  

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

    #transform = Normalize(mean=cfg.data.normalization_mean, std=cfg.data.normalization_std)

    case_name = 'raw_data'

    dataset_train = CustomKarmanStreetDataset(base_folder=cfg.data.base_folder, 
                                            Re_list=[200],
                                            field_name=cfg.data.field_name, 
                                            num_channels=cfg.data.num_channels, 
                                            case_name=case_name)
    
    #dataset_train.__dict__.keys() = dict_keys(['data_x', 'data_y', 'num_elements', 'transform', 'target_transform'])
    dataset_train, dataset_val = random_split(
        dataset_train, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
    ) #len(dataset_val.__dict__['indices'])
    
    train_dataloader = DataLoader(dataset=dataset_train, batch_size = cfg.train.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=dataset_val, batch_size = cfg.train.training.batch_size, shuffle=True)

    ckpt_args = {
        #add the date and time to the results folder to create the path: 
        #"path": str(result_folder.joinpath(f"{cfg.output.output_name}_{cfg.data.field_name}_{formatted_datetime}")),
        "path": os.path.dirname(os.getcwd()),
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

    #hardcoded (find a better way to do this)
    cfg.arch.decoder.out_features = cfg.data.num_channels
    cfg.arch.fno.in_channels = cfg.data.num_channels

    if cfg.output.logging.wandb:
        wandb_config = OmegaConf.to_container(cfg)
        wandb_config["num_model_params"] = sum(p.numel() for p in model.parameters())
        wandb_config["loaded_epoch"] = loaded_pseudo_epoch
        
        run_name = f"{cfg.output.output_name}_{cfg.data.field_name}_{formatted_datetime}"
        wandb_run = wandb.init(
                project=cfg.output.logging.wandb_project,
                entity=cfg.output.logging.wandb_entity,
                name=run_name,
                config=wandb_config,
                save_code=True,
            )
    
    
    ################# Training #######################
    best_validation_error = None
    
    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.train.training.max_pseudo_epochs + 1
    ):
        # Training loop
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            minibatch_losses = 0.0
            for step, batch in zip(range(steps_per_pseudo_epoch), train_dataloader):
                minibatch_loss = forward_train(batch[0].to(dist.device), batch[1].to(dist.device)) #batch[0].shape [16,2,512,256]; batch[1].shape [16,2,512,256]
                logger.log_minibatch({"loss": minibatch_loss.detach()}) #even if we pass minibatch_loss, the log_minibatch will accumulate and divideby the number of steps at the end of the epoch.
                minibatch_losses += minibatch_loss.detach().item()
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            if cfg.output.logging.wandb:
                wandb_run.log({"train/loss": minibatch_losses/(step+1)},pseudo_epoch)
                wandb_run.log({"train/Learning Rate": optimizer.param_groups[0]["lr"]},pseudo_epoch)
        
        # save checkpoint (to restart the training from the last saved checkpoint)
        if pseudo_epoch % cfg.train.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.train.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger: #after one epoch, he logging happens here.
                total_loss = 0.0
                #################  Note  #######################
                # validation_iters = 16 (maximum)
                # len(val_dataloader.__dict__['dataset'].__dict__['indices']) = 414, i.e. total of 414 timestep information is available for validation.
                # 414/16 = 25.875, so the loop will run for only "16" iterations and terminate. 
                for step, batch in zip(range(validation_iters), val_dataloader):
                #for step, batch in zip(range(len(val_dataloader)), val_dataloader):
                    # val_loss = validator.compare(
                    #     invar=batch[0].to(dist.device),
                    #     target=batch[1].to(dist.device),
                    #     prediction=forward_eval(batch[0].to(dist.device)),
                    #     step=pseudo_epoch,
                    # )
                    # batch[0].shape = [16,2,512,256]; batch[1].shape = [16,2,512,256]
                    val_loss = validator.compute_only_loss(target=batch[1].to(dist.device)
                                                          ,prediction=forward_eval(batch[0].to(dist.device)))
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / step}) #validation error per epoch
                
                #save the checkpoint with the best validation error inside the folder "best"
                if best_validation_error is None or best_validation_error > (total_loss / step):
                    best_validation_error = total_loss / step #new best validation error
                    print(f"best_validation_error so far: {best_validation_error}")
                    best_ckpt_args = {
                            "path": os.path.join(os.path.dirname(os.getcwd()),"best"),
                            "optimizer": optimizer,
                            "scheduler": scheduler,
                            "models": model,
                        }
                    #remove all the files inside the path
                    if os.path.exists(os.path.join(os.path.dirname(os.getcwd()),"best")):
                        shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),"best"))
                    
                    os.makedirs(os.path.join(os.path.dirname(os.getcwd()),"best"))
                    save_checkpoint(**best_ckpt_args, epoch=pseudo_epoch)       
                
                if cfg.output.logging.wandb:
                    wandb_run.log({"valid/Validation error": total_loss / step},pseudo_epoch)

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()
    
    save_checkpoint(**ckpt_args, epoch=cfg.train.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

    if cfg.output.logging.wandb:
        wandb.finish()

    ################# Inference #######################
    path = os.path.join(os.path.dirname(os.getcwd()),"best")
    model_inf = Module.from_checkpoint(path).to(dist.device)
    model_inf.eval()

    """
    with torch.inference_mode():
                #loading the data
                data_x = data_x_total[0]
                data_y = data_y_total[0]
                input = data_x.to("cuda")
                data_y = data_y.to("cuda")
 
                for t in range(prediction_timesteps):    
                    output = model_inf(input)
                    input = output.detach().clone()
    """


    with torch.inference_mode():
        pass
            
            





if __name__ == "__main__":
    main()