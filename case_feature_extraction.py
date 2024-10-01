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

from math import ceil
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split
from validator import GridValidator
from custom_karman_street_dataset import CustomDataset, DataTransform

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

    ##hardcoded for now (assuming only Re is the parameter)
    if cfg.data.param_names == []:
        cfg.data.num_channels = cfg.data.num_channels - 1
        #cfg.arch.fno.in_channels = eval(str(cfg.arch.fno.in_channels)) 
        #cfg.arch.decoder.out_features = eval(str(cfg.arch.decoder.out_features)) 

    #Re is not predicted (hardcoded)
    else:
        cfg.arch.decoder.out_features = eval(str(cfg.arch.decoder.out_features))-1 

    # define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=eval(str(cfg.arch.fno.in_channels)),           
        out_channels=eval(str(cfg.arch.decoder.out_features)),     
        decoder_layers=cfg.arch.decoder.layers,         
        decoder_layer_size=cfg.arch.decoder.layer_size, 
        dimension=cfg.arch.fno.dimension,              
        latent_channels=cfg.arch.fno.latent_channels,   
        num_fno_layers=cfg.arch.fno.fno_layers,         
        num_fno_modes=cfg.arch.fno.fno_modes,           
        padding=cfg.arch.fno.padding,                   
    ).to(dist.device)

    loss_fun = MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )

    log.log("############################################################################################################")
    log.log("Case Details:")
    log.log("case_name: " +str(cfg.data.case_name))

    log.log("------------------------------------------------------------------------------------------------------------")
    log.log("Model Details:")
    log.log("number of model parameters: "+str(sum(p.numel() for p in model.parameters())))
    log.log("in_channels: "+str(eval(str(cfg.arch.fno.in_channels))))
    log.log("out_features: "+str(eval(str(cfg.arch.decoder.out_features))))
    log.log("decoder_layers: "+str(cfg.arch.decoder.layers)) #decoder is for projection(downsampling) and encoder is for lifting(upsampling).
    log.log("decoder_layer_size: "+str(cfg.arch.decoder.layer_size)) #decoder_layer_size is the number of neurons in each layer of the decoder.
    log.log("dimension: "+str(cfg.arch.fno.dimension)) #for 2D-FNO the dimension of the tensor passed to the FNOmodel should be 4. (batch_size, channels, x_res, y_res)
    log.log("latent_channels (after lifting): "+str(cfg.arch.fno.latent_channels))
    log.log("num_fno_layers: "+str(cfg.arch.fno.fno_layers))
    log.log("num_fno_modes: "+str(cfg.arch.fno.fno_modes))
    log.log("padding: "+str(cfg.arch.fno.padding))

    dataset_train = CustomDataset(base_folder=cfg.data.base_folder,
                                            field_names=cfg.data.field_names, 
                                            param_names=cfg.data.param_names, 
                                            filter_frame=cfg.data.filter_frame,
                                            sequence_info=cfg.data.sequence_info,
                                            case_name=cfg.data.case_name,
                                            n_rollout_steps=cfg.data.n_rollout_steps)
    
    dataset_train.transform = DataTransform(mean_info=cfg.data.normalization_mean, 
                                            std_info=cfg.data.normalization_std,
                                            field_names=cfg.data.field_names,
                                            param_names=cfg.data.param_names,
                                            case_name=cfg.data.case_name)

    dataset_train, dataset_val = random_split(
        dataset_train, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
    ) 
    log.log("------------------------------------------------------------------------------------------------------------")
    log.log("Train Dataset Details:")
    log.log("base_folder: "+str(cfg.data.base_folder))
    log.log("field_names: "+str(cfg.data.field_names))
    log.log("param_names: "+str(cfg.data.param_names))
    #log.log("param_values: "+str(cfg.data.param_values))
    log.log("filter_frame (min and max frames used from the simulation): "+str(cfg.data.filter_frame))
    log.log("sequence_info (i/p+gt, sequnce_stride): "+str(cfg.data.sequence_info))
    log.log("mean_info: "+str(cfg.data.normalization_mean))
    log.log("std_info: "+str(cfg.data.normalization_std))
    log.log("No. of training batches (=len(train_loader)): "+str(ceil(len(dataset_train)/cfg.train.training.batch_size)))
    log.log("No. of validation batches (=len(val_loader)): "+str(ceil(len(dataset_val)/cfg.train.training.batch_size)))
    
    train_dataloader = DataLoader(dataset=dataset_train, batch_size = cfg.train.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=dataset_val, batch_size = cfg.train.training.batch_size, shuffle=True)

    ckpt_args = {
        "path": os.path.dirname(os.getcwd()),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    validator = GridValidator(out_dir=ckpt_args["path"] + "/validators", loss_fun=MSELoss(), num_channels=cfg.arch.decoder.out_features)
    
    # calculate the no. of times the training loop is executed for each pseudo epoch.
    steps_per_pseudo_epoch = ceil(cfg.train.training.pseudo_epoch_sample_size / cfg.train.training.batch_size)
    #steps_per_pseudo_epoch = 2048/16 = 128. 128 times the training loop is executed for each pseudo epoch.

    validation_iters = ceil(cfg.train.validation.sample_size / cfg.train.training.batch_size) #validation_iters = 256/16 = 16.
    
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }

    log.log("------------------------------------------------------------------------------------------------------------")
    log.log("Training Details:")
    log.log("Batch size: "+str(cfg.train.training.batch_size))
    log.log("Number of epochs: "+str(cfg.train.training.max_pseudo_epochs))
    log.log("Number of times the training loop is executed for each epoch: "+str(steps_per_pseudo_epoch))
    log.log("Actual number of times training loop is executed for each epoch: "+str(min(steps_per_pseudo_epoch,len(train_dataloader))))
    log.log("Number of times the validation loop is executed for each epoch: "+str(validation_iters))
    log.log("Actual number of times validation loop is executed for each epoch: "+str(min(validation_iters,len(val_dataloader))))
    log.log("Checkpoint save frequency (After these many epochs): "+str(cfg.train.training.rec_results_freq))
    log.log("Initial learning rate: "+str(cfg.scheduler.initial_lr))
    log.log("Decay rate: "+str(cfg.scheduler.decay_rate))
    log.log("Decay frequency (After these many epochs): "+str(cfg.scheduler.decay_pseudo_epochs))
    log.log("############################################################################################################")

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
        loss = loss_fun(pred, target) #MSE_loss(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)
    
    ##restart training from the last saved checkpoint
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)
    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    #modified below for logging in wandb
    cfg.arch.decoder.out_features = eval(str(cfg.arch.decoder.out_features))
    cfg.arch.fno.in_channels = eval(str(cfg.arch.fno.in_channels))

    if cfg.output.logging.wandb:
        wandb_config = OmegaConf.to_container(cfg)
        wandb_config["num_model_params"] = sum(p.numel() for p in model.parameters())
        wandb_config["loaded_epoch"] = loaded_pseudo_epoch
        
        run_name = f"{cfg.output.output_name}_{cfg.data.case_name}_{cfg.data.field_names}_{cfg.data.param_names}_{formatted_datetime}"
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
                logger.log_minibatch({"loss": minibatch_loss.detach()}) #even if we pass minibatch_loss, the log_minibatch will accumulate and divided by the number of steps at the end of the epoch.
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
                for step, batch in zip(range(validation_iters), val_dataloader):
                #for step, batch in zip(range(len(val_dataloader)), val_dataloader):
                    val_loss = validator.compare(
                        invar=batch[0].to(dist.device),
                        target=batch[1].to(dist.device),
                        prediction=forward_eval(batch[0].to(dist.device)),
                        step=pseudo_epoch,
                    )
                    # batch[0].shape = [16,2,512,256]; batch[1].shape = [16,2,512,256]
                    val_loss = validator.compute_only_loss(target=batch[1].to(dist.device)
                                                          ,prediction=forward_eval(batch[0].to(dist.device)))
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / step}) #validation error per epoch
                
                #save the checkpoint with the best validation error inside the folder "best" to be used in inference.
                if best_validation_error is None or best_validation_error > (total_loss / step):
                    best_validation_error = total_loss / step #new best validation error
                    log.success(f"best_validation_error so far: {best_validation_error}")
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


    """
    ################# Inference #######################
    path = os.path.join(os.path.dirname(os.getcwd()),"best")
    model_inf = Module.from_checkpoint(path).to(dist.device)
    model_inf.eval()

    
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


if __name__ == "__main__":
    main()