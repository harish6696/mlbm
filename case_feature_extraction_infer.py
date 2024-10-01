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
from hydra.utils import get_original_cwd

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
    log = PythonLogger(name="LBM_fno_infer")

    LaunchLogger.initialize()  

    #load the best model
    trained_model_path = os.path.join(get_original_cwd(), os.path.join(cfg.data.model_path,"best"))
    #load the file starting with FourrierNeuralOperator inside the trained_model_path
    trained_model_file = [f for f in os.listdir(trained_model_path) if f.startswith("FourierNeuralOperator")][0]
    #add to the path
    trained_model_file = os.path.join(trained_model_path, trained_model_file)
    # The parameters to instantitate the model will be loaded from the checkpoint
    model_inf = Module.from_checkpoint(trained_model_file).to("cuda")
    print("Model loaded successfully")
    # put the model in evaluation mode
    model_inf.eval()

    dataset_infer = CustomDataset(base_folder=cfg.data.base_folder,
                                            field_names=cfg.data.field_names, 
                                            param_names=cfg.data.param_names, 
                                            filter_frame=cfg.data.filter_frame,
                                            sequence_info=cfg.data.sequence_info,
                                            case_name=cfg.data.case_name,
                                            n_rollout_steps=cfg.data.n_rollout_steps,
                                            mode=cfg.data.mode)
    
    dataset_infer.transform = DataTransform(mean_info=cfg.data.normalization_mean,
                                            std_info=cfg.data.normalization_std,
                                            field_names=cfg.data.field_names,
                                            param_names=cfg.data.param_names,
                                            case_name=cfg.data.case_name)

    infer_dataloader = DataLoader(dataset=dataset_infer, batch_size = cfg.train.inference.batch_size, shuffle=True, drop_last=True)

    for batch in infer_dataloader:
        input_tensor=batch[0]
        target_tensor=batch[1]

        input_tensor = input_tensor.to("cuda")
        target_tensor = target_tensor.to("cuda") 

        #if parameter is not none, save the parameter channel as it needs to be appended to the input
        if cfg.data.param_names is not None:
            param_tensor = input_tensor[:,-len(cfg.data.param_names):,:,:]

        loss = []
        if cfg.data.case_name == "raw_in_raw_out":
            for t in range(cfg.data.n_rollout_steps):
                with torch.no_grad():
                    output = model_inf(input_tensor)
                    
                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #one number per batch (better to keep shuffle ON and only have one inference folder (say Re_100) inside infer)
                    loss.append(step_loss.item())
                    input_tensor = output.detach().clone()
                    
                    if cfg.data.param_names is not None:
                        input_tensor = torch.cat((input_tensor, param_tensor), dim=1)

            print("Loss list every batch: ", loss)

        elif cfg.data.case_name == "3_hist_raw_in_raw_out":
            input_tensor= input_tensor.reshape((cfg.train.inference.batch_size,3,-1,input_tensor.shape[2],input_tensor.shape[3]))
            for t in range(cfg.data.n_rollout_steps):
                with torch.no_grad():
                    #extracting the last two steps which will be concatenated with the output
                    two_steps_input = input_tensor[:,1:,:,:,:]

                    input_tensor = input_tensor.reshape((cfg.train.inference.batch_size,-1,input_tensor.shape[3],input_tensor.shape[4]))

                    output = model_inf(input_tensor)
                    
                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #one number per batch
                    loss.append(step_loss.item())
                    
                    if cfg.data.param_names is not None:
                        output = torch.cat((output.detach().clone(), param_tensor), dim=1)
                    
                    output = output[:,None,:,:,:]
                    input_tensor = torch.cat((two_steps_input, output), dim=1)

            print("Loss list every batch: ", loss)

        elif cfg.data.case_name == "raw_and_grad_rho_in_raw_out": 
            raise NotImplementedError()
        
        elif cfg.data.case_name == "acc_in_acc_out":
            #input_tensor.shape : [4,1024,256]; gt_tensor.shape : [10,3,1024,256]
            #both input, output and gt have differences of the field data. 
            for t in range(cfg.data.n_rollout_steps-1):
                with torch.no_grad():
                    output = model_inf(input_tensor)
                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #loss is computed over the differnces of the field data
                    loss.append(step_loss.item())
                    input_tensor = output.detach().clone()
                    
                    if cfg.data.param_names is not None:
                        input_tensor = torch.cat((input_tensor, param_tensor), dim=1)
            
            print("Loss list every batch: ", loss)

        elif cfg.data.case_name=="2_hist_acc_in_acc_out": 
            input_tensor= input_tensor.reshape((cfg.train.inference.batch_size,2,-1,input_tensor.shape[2],input_tensor.shape[3]))
            for t in range(cfg.data.n_rollout_steps-1):
                with torch.no_grad():
                    one_step_from_input = input_tensor[:,1:,:,:,:] #this will be concatenated with the output and used in the next iteration
                    input_tensor = input_tensor.reshape((cfg.train.inference.batch_size,-1,input_tensor.shape[3],input_tensor.shape[4]))

                    output = model_inf(input_tensor)
                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #loss is computed over the differnces of the field data
                    loss.append(step_loss.item())

                    if cfg.data.param_names is not None:
                        output = torch.cat((output.detach().clone(), param_tensor), dim=1)

                    output = output[:,None,:,:,:]
                    input_tensor = torch.cat((one_step_from_input, output), dim=1)

            print("Loss list every batch: ", loss)

        elif cfg.data.case_name=="acc_and_raw_in_raw_out":
            input_tensor= input_tensor.reshape((cfg.train.inference.batch_size,2,-1,input_tensor.shape[2],input_tensor.shape[3])) 
            for t in range(cfg.data.n_rollout_steps-1):
                with torch.no_grad():
                    one_step_from_input = input_tensor[:,1:,:,:,:] #this will be concatenated with the output and used in the next iteration
                    
                    input_tensor = input_tensor.reshape((cfg.train.inference.batch_size,-1,input_tensor.shape[3],input_tensor.shape[4]))
                    
                    output = model_inf(input_tensor)
                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #both output and target_tensor have raw field data
                    loss.append(step_loss.item())
                    
                    if cfg.data.param_names is not None:
                        output = torch.cat((output.detach().clone(), param_tensor), dim=1)
                    
                    output = output[:,None,:,:,:]
                    delta_op = output - one_step_from_input
                    input_tensor = torch.cat((delta_op, output), dim=1)
            
            print("Loss list every batch: ", loss)
        
        else:
            raise ValueError("Case name not recognized")


if __name__ == "__main__":
    main()

