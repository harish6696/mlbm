# from lightning import Trainer
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger
from modulus import Module
from omegaconf import DictConfig, OmegaConf

from torch.nn import MSELoss
import hydra
from torch import FloatTensor

from math import ceil
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split
from validator import GridValidator
from custom_karman_street_dataset import CustomDataset, DataTransform
from hydra.utils import get_original_cwd

from density_grad import get_density_grad

import os
import numpy as np

import h5py
import copy

def undo_acc_normalization(tensor, mean_info, std_info):
    #undo the normalization
    tensor[:,0,:,:] = tensor[:,0,:,:]*std_info['drho_dt'] + mean_info['drho_dt']
    tensor[:,1,:,:] = tensor[:,1,:,:]*std_info['du_dt'] + mean_info['du_dt']
    tensor[:,2,:,:] = tensor[:,2,:,:]*std_info['dv_dt'] + mean_info['dv_dt']
    return tensor


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

    #mask = torch.load(os.path.join(get_original_cwd(), "Dataset_KVS_200_0.2_re_100-500/obs_mask.pt"), weights_only=True)
    h5_file_path = os.path.join(get_original_cwd(), "VS_Re_100_to_500_uniform_skip_20_256x64/inverted_mask_256x64.h5")
    with h5py.File(h5_file_path, 'r') as h5_file:
        mask = h5_file['mask'][:]

    mask = torch.tensor(mask, dtype=torch.float32)
    mask = mask.to(dtype=torch.int32)
    mask = mask.to(dist.device) #this is a regular mask with 0s and 1s. Replace 0s with nans only inside the validator.

    dataset_infer = CustomDataset(base_folder=cfg.data.base_folder,
                                            field_names=cfg.data.field_names, 
                                            param_names=cfg.data.param_names, 
                                            filter_frame=cfg.data.filter_frame,
                                            sequence_info=cfg.data.sequence_info,
                                            case_name=cfg.data.case_name,
                                            n_rollout_steps=cfg.data.n_rollout_steps,
                                            mode=cfg.data.mode,
                                            obs_mask=mask)
    
    dataset_infer.transform = DataTransform(mean_info=cfg.data.normalization_mean,
                                            std_info=cfg.data.normalization_std,
                                            field_names=cfg.data.field_names,
                                            param_names=cfg.data.param_names,
                                            case_name=cfg.data.case_name,
                                            mode=cfg.data.mode)

    infer_dataloader = DataLoader(dataset=dataset_infer, batch_size = cfg.train.inference.batch_size, shuffle=True, drop_last=True)

    #increase the input channels by 1 if Re is present
    if cfg.data.param_names != []:
        cfg.arch.fno.in_channels = eval(str(cfg.arch.fno.in_channels))+1

    batch_loss_list = []
    for batch_idx, batch in enumerate(infer_dataloader):
        input_tensor=batch[0]
        target_tensor=batch[1]

        input_tensor = input_tensor.to("cuda")
        target_tensor = target_tensor.to("cuda") 

        #if parameter is not none, save the parameter channel as it needs to be appended to the input
        if cfg.data.param_names is not None:
            param_tensor = input_tensor[:,-len(cfg.data.param_names):,:,:]

        loss = []

        validator = GridValidator(out_dir=os.path.dirname(os.getcwd())+ f"/infer_plots_{batch_idx}", 
                                              loss_fun=MSELoss(), 
                                              num_output_channels=cfg.arch.decoder.out_features, 
                                              num_input_channels=cfg.arch.fno.in_channels, 
                                              case_name=cfg.data.case_name ,
                                              mean_info=cfg.data.normalization_mean, 
                                              std_info=cfg.data.normalization_std, 
                                              mode=cfg.data.mode,
                                              obs_mask=mask) 

        if cfg.data.case_name == "raw_in_raw_out":
            for t in range(cfg.data.n_rollout_steps):
                with torch.no_grad():
                    output = model_inf(input_tensor)
                    output = output*mask
                    target_tensor[:,t,:,:,:] = target_tensor[:,t,:,:,:]*mask

                    validator.compare(
                            invar=input_tensor,
                            target=target_tensor[:,t,:,:,:],
                            prediction=output,
                            step=t,
                        )

                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #one number per batch (better to keep shuffle ON and only have one dataset inside inference folder (say Re_100))
                    loss.append(step_loss.item())
                    input_tensor = output.detach().clone()
                    
                    if cfg.data.param_names is not None:
                        input_tensor = torch.cat((input_tensor, param_tensor), dim=1)
                    
            print(f"Loss list for batch_{batch_idx}: ", loss)
            batch_loss_list.append(loss)

        elif cfg.data.case_name == "3_hist_raw_in_raw_out":
            #input_tensor= input_tensor.reshape((cfg.train.inference.batch_size,10,input_tensor.shape[2],input_tensor.shape[3]))
            for t in range(cfg.data.n_rollout_steps):
                with torch.no_grad():
                    #extracting the last two steps which will be concatenated with the output
                    two_steps_input = input_tensor[:,3:9,:,:] #exclude the Re channel (the last channel)

                    output = model_inf(input_tensor)

                    two_steps_input = two_steps_input*mask
                    output = output*mask
                    target_tensor[:,t,:,:,:] = target_tensor[:,t,:,:,:]*mask
                    
                    validator.compare(
                            invar=input_tensor,
                            target=target_tensor[:,t,:,:,:],
                            prediction=output,
                            step=t,
                        )

                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #one number per batch
                    loss.append(step_loss.item())
                    
                    if cfg.data.param_names is not None:
                        output = torch.cat((output.detach().clone(), param_tensor), dim=1)
                    
                    input_tensor = torch.cat((two_steps_input, output), dim=1)

            print(f"Loss list for batch_{batch_idx}: ", loss)
            batch_loss_list.append(loss)

        elif cfg.data.case_name == "raw_and_grad_rho_in_raw_out": 
            for t in range(cfg.data.n_rollout_steps):
                with torch.no_grad():
                    output = model_inf(input_tensor)
                    
                    output = output*mask
                    target_tensor[:,t,:,:,:] = target_tensor[:,t,:,:,:]*mask

                    validator.compare(
                            invar=input_tensor,
                            target=target_tensor[:,t,:,:,:],
                            prediction=output,
                            step=t,
                        )

                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #one number per batch (better to keep shuffle ON and only have one inference folder (say Re_100) inside infer)
                    loss.append(step_loss.item())

                    #compute the spatial-gradient and append it to the input tensor.
                    density = output[:,0:1,:,:]

                    density_pu= density[:,0:1,:,:]*cfg.data.normalization_std['rho'] + cfg.data.normalization_mean['rho']
                    #convert density to physical units before obtaining the gradient (because dx is hardcoded in physical units inside get_density_grad function)
                    density_grad= get_density_grad(mask.repeat(density_pu.shape[0],1,1,1), density_pu.cpu())

                    #Z-normalize the density gradient before appending it to the input tensor
                    density_grad[:,0,:,:] = (density_grad[:,0,:,:] - cfg.data.normalization_mean["grad_rho_x/_rho"])/cfg.data.normalization_std["grad_rho_x/_rho"]
                    density_grad[:,1,:,:] = (density_grad[:,1,:,:] - cfg.data.normalization_mean["grad_rho_y/_rho"])/cfg.data.normalization_std["grad_rho_y/_rho"]

                    density_grad = density_grad.to(dist.device)
                    #density_grad = density_grad*mask #is this really required?
                    #append the density gradient to the input tensor
                    input_tensor = torch.cat((output, density_grad), dim=1)

                    input_tensor = input_tensor.detach().clone()

                    if cfg.data.param_names is not None:
                        input_tensor = torch.cat((input_tensor, param_tensor), dim=1)
                    
            print(f"Loss list for batch_{batch_idx}: ", loss)
            batch_loss_list.append(loss)                    
    
        elif cfg.data.case_name == "acc_in_acc_out":
            #input_tensor.shape : [4,1024,256]; gt_tensor.shape : [10,3,1024,256]
            #both input, output and gt have differences of the field data. 
            extra_field = batch[2].squeeze(dim=1).to("cuda") #has GT information
            for t in range(cfg.data.n_rollout_steps-1):
                with torch.no_grad():
                    output = model_inf(input_tensor)
                    
                    output_renormalized = undo_acc_normalization(copy.deepcopy(output), cfg.data.normalization_mean, cfg.data.normalization_std)
                    target_tensor_renormalized = undo_acc_normalization(copy.deepcopy(target_tensor[:,t,:,:,:]), cfg.data.normalization_mean, cfg.data.normalization_std)                    
                    
                    if(t==0):#converting from differences(accelerations) to actual field data
                        output_velocity = output_renormalized+extra_field 
                        target_tensor_velocity = target_tensor_renormalized+extra_field
                    else:
                        output_velocity = output_renormalized+output_velocity
                        target_tensor_velocity = target_tensor_renormalized+target_tensor_velocity
                    
                    output = output*mask
                    target_tensor[:,t,:,:,:] = target_tensor[:,t,:,:,:]*mask

                    output_velocity = output_velocity*mask
                    target_tensor_velocity = target_tensor_velocity*mask

                    validator.compare(
                            invar=input_tensor,
                            target=target_tensor_velocity, #these are already renormalized
                            prediction=output_velocity,
                            step=t,
                        )
                    
                    #step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #loss is computed over the differnces of the field data
                    step_loss = MSELoss()(output_velocity, target_tensor_velocity) 
                    
                    loss.append(step_loss.item())
                    input_tensor = output.detach().clone()
                    
                    if cfg.data.param_names is not None:
                        input_tensor = torch.cat((input_tensor, param_tensor), dim=1)
            
            print(f"Loss list for batch_{batch_idx}: ", loss)
            batch_loss_list.append(loss)

        elif cfg.data.case_name=="2_hist_acc_in_acc_out": 
            #input_tensor= input_tensor.reshape((cfg.train.inference.batch_size,2,-1,input_tensor.shape[2],input_tensor.shape[3]))
            extra_field = batch[2].squeeze(dim=1).to("cuda") #has GT information
            for t in range(cfg.data.n_rollout_steps-1):
                with torch.no_grad():
                    one_step_from_input = input_tensor[:,3:6,:,:] #this will be concatenated with the output and used in the next iteration
                    #input_tensor = input_tensor.reshape((cfg.train.inference.batch_size,-1,input_tensor.shape[3],input_tensor.shape[4]))

                    output = model_inf(input_tensor)

                    output_renormalized = undo_acc_normalization(copy.deepcopy(output), cfg.data.normalization_mean, cfg.data.normalization_std)
                    target_tensor_renormalized = undo_acc_normalization(copy.deepcopy(target_tensor[:,t,:,:,:]), cfg.data.normalization_mean, cfg.data.normalization_std)                    
                    
                    if(t==0):#converting from differences(accelerations) to actual field data
                        output_velocity = output_renormalized+extra_field 
                        target_tensor_velocity = target_tensor_renormalized+extra_field
                    else:
                        output_velocity = output_renormalized+output_velocity
                        target_tensor_velocity = target_tensor_renormalized+target_tensor_velocity                   

                    one_step_from_input = one_step_from_input*mask
                    output = output*mask
                    target_tensor[:,t,:,:,:] = target_tensor[:,t,:,:,:]*mask

                    validator.compare(
                            invar=input_tensor,
                            target=target_tensor_velocity, #these are already renormalized
                            prediction=output_velocity,
                            step=t,
                        )
                    step_loss = MSELoss()(output_velocity, target_tensor_velocity) 
                    #step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #loss is computed over the differnces of the field data
                    loss.append(step_loss.item())

                    if cfg.data.param_names is not None:
                        output = torch.cat((output.detach().clone(), param_tensor), dim=1)

                    #output = output[:,None,:,:,:]
                    input_tensor = torch.cat((one_step_from_input, output), dim=1)

            print(f"Loss list for batch_{batch_idx}: ", loss)
            batch_loss_list.append(loss)

        elif cfg.data.case_name=="acc_and_raw_in_raw_out":
            #input_tensor= input_tensor.reshape((cfg.train.inference.batch_size,2,-1,input_tensor.shape[2],input_tensor.shape[3])) 
            for t in range(cfg.data.n_rollout_steps):
                with torch.no_grad():
                    one_step_from_input = input_tensor[:,3:6,:,:] #this will be concatenated with the output and used in the next iteration
                    
                    output = model_inf(input_tensor)
                    
                    output = output*mask
                    target_tensor[:,t,:,:,:] = target_tensor[:,t,:,:,:]*mask
                    
                    validator.compare(
                            invar=input_tensor,
                            target=target_tensor[:,t,:,:,:],
                            prediction=output,
                            step=t,
                        )
                    
                    step_loss = MSELoss()(output, target_tensor[:,t,:,:,:]) #both output and target_tensor have raw field data
                    loss.append(step_loss.item())
                    
                    #output = output[:,None,:,:,:]
                    delta_op = output - one_step_from_input

                    if cfg.data.param_names is not None:
                        output = torch.cat((output.detach().clone(), param_tensor), dim=1)
                    
                    input_tensor = torch.cat((delta_op, output), dim=1)
            
            print(f"Loss list for batch_{batch_idx}: ", loss)
            batch_loss_list.append(loss)
        
        else:
            raise ValueError("Case name not recognized")
    
    Re_string= os.listdir(os.path.join(get_original_cwd(), cfg.data.base_folder))[0]
    batch_loss_list = np.array(batch_loss_list)
    #save the loss list
    file_path = os.path.join(os.getcwd(), f"{cfg.data.case_name}_{Re_string}_fno_no_pad.npy")
    np.save(file_path, batch_loss_list)

if __name__ == "__main__":
    main()

