import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
import os
from hydra.utils import get_original_cwd
##Datasets have arrived from the folder "final_dataset_for_train". Now build the CustomDataset

class CustomDataset(Dataset):
    def __init__(self, base_folder: str, 
                 field_names: List[str], 
                 param_names: List[str], 
                 case_name: str,
                 filter_frame:List[Tuple[int,int]]=[], 
                 sequence_info: List[Tuple[int,int]]=[],
                 mode:str='train',
                 n_rollout_steps: int=1): #while training n_rollout_steps=1
        
        self.case_name = case_name
        self.transform = None
        self.data_dir = base_folder 
        self.field_names = field_names  #velocity, density or both
        self.param_names = param_names  #Re, Ma or both
        self.min_frame = filter_frame[0][0]
        self.max_frame = filter_frame[0][1]
        self.seq_length= sequence_info[0][0] # includes n-1 historic info and 1 current info (GT)
        self.seq_stride = sequence_info[0][1] # stride between sequences
        self.mode = mode
        self.data_paths = []
        ######################################
        ### Stage 1 : Generate data paths list
        ######################################
        top_dir= os.path.join(get_original_cwd(), self.data_dir)
        top_dir_folders= os.listdir(os.path.join(get_original_cwd(), self.data_dir))
        top_dir_folders.sort()

        if self.mode == 'infer':
            self.original_seq_length = self.seq_length
        
        self.seq_length = self.seq_length + n_rollout_steps -1 #-1 is to remove the GT (as there is no GT in inference mode)

        for dir in top_dir_folders: #top_dir_folders has Re_200, Re_300 and Re_400.
            cwd=os.path.join(top_dir, dir) #need this because hydra changes the cwd
            for seq_start in range(self.min_frame, self.max_frame, self.seq_length*self.seq_stride):
                valid_seq = True
                for frame in range(seq_start, seq_start + (self.seq_length*self.seq_stride), self.seq_stride): 
                    # discard incomplete sequences at simulation end
                    if seq_start+self.seq_length*self.seq_stride > self.max_frame:
                        valid_seq = False
                        break

                    for field in self.field_names: #this loop is just to check if the file exists or not
                        current_field = os.path.join(cwd, "%s_%04d.pt" % (field, frame))
                        if not os.path.isfile(current_field):
                            raise FileNotFoundError("Could not load %s file: %s" % (field, current_field))

                # incomplete sequence means there are no more frames left
                if not valid_seq:
                    break
                #data_paths doesnt have density or velocity in the name, just the paths
                self.data_paths.append((cwd, seq_start, seq_start + self.seq_length*self.seq_stride, self.seq_stride))

        print("Dataset Length: %d\n" % len(self.data_paths))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        #only load the data for the current idx from the data_paths list. (Slow but memory efficient)
        #print(self.data_paths[idx])
        base_path, seq_start, seq_end, seq_stride = self.data_paths[idx] #basePath = '.../Re_200', seqStart = 661, seqEnd = 665, seqSkip = 2, say for idx 11

        loaded = {} #empty dictionary
        for field in self.field_names:
            loaded[field] = []

        for frame in range(seq_start, seq_end, seq_stride): #this loop executes 3 times as seqSkip = 2. 
            for field in self.field_names:
                loaded_arr = torch.load(os.path.join(base_path, "%s_%04d.pt" % (field, frame)), weights_only=True)
                loaded[field] += [loaded_arr.to(torch.float32)]       

        loaded_fields = []
        for field in self.field_names:
            loaded_fields += [torch.stack(loaded[field], dim=0)]  
            #loaded_fields[0].shape (3, 1, 1024, 256) i.e density, loaded_fields[1].shape (3, 2, 1024, 256) i.e. velocity,
        #loaded_fields is a list with each entry correspondin to a field- density, velocity etc.
        ###########################################################################
        ### Stage 2 : Feature Engineering (Modifiy the data before normalizing it)
        ###########################################################################
            
        if self.case_name=="raw_in_raw_out": #contains 2 (1+1 GT) timesteps of velocity or density or both. u^n --> u^{n+1}
            pass #nothing to be modified, just normalize the data in the next step

        elif self.case_name=="3_hist_raw_in_raw_out":  #contains 4 (3+1 GT) timesteps of velocity or density or both. u^{n-2}, u^{n-1}, u^n --> u^{n+1}
            pass #nothing to be modified, just normalize the data in the next step

        elif self.case_name=="raw_and_grad_rho_in_raw_out": #contains 2 (1+1 GT) timesteps of velocity or density or both. u^n, grad_rho^n --> u^{n+1}
            pass

        elif self.case_name=="acc_in_acc_out": #contains 3 (2+1 GT) timesteps of derivative of velocity or density or both. delta_u^n (u^n-u^{n-1}) --> delta_u^{n+1} (u^{n+1}-u^n)
            for i in range(len(self.field_names)):
                loaded_fields[i] = loaded_fields[i][1:] - loaded_fields[i][:-1] #delta_u^n (u^n-u^{n-1})
                # loaded_fields[0].shape (3-->2,1, 1024, 256) and loaded_fields[1].shape (3-->2,2, 1024, 256).--> final_shape after concatenation= (2, 3, 1024, 256) 
                        
        elif self.case_name=="2_hist_acc_in_acc_out":  # contains 4 (3+1 GT) timesteps of derivative of velocity or density or both.  delta_u^{n-1}, delta_u^n --> delta_u^{n+1} 
            for i in range(len(self.field_names)):
                loaded_fields[i] = loaded_fields[i][1:] - loaded_fields[i][:-1] #delta_u^n (u^n-u^{n-1})
                # loaded_fields[0].shape (4-->3,1, 1024, 256) and loaded_fields[1].shape (4-->3, 2, 1024, 256).--> final_shape after concatenation= (3, 3, 1024, 256)
                # here both input is two acceleraton and output is also accelerations hence dim_0=3
        
        elif self.case_name=="acc_and_raw_in_raw_out": #contains 3 (2+1 GT) timesteps. delta_u^n (=u^n-u^{n-1}), u^n --> u^{n+1}    
            for i in range(len(self.field_names)): #selecting i-th entry of the loaded_fields list which is a field array like denstiy, velocity etc.
                loaded_fields[i][0] = loaded_fields[i][1] - loaded_fields[i][0] #other entries of the loaded_fields[i] array are already in the form of u^n 
                
        #Condition it with the simulation parameter
        if 'Re' in self.param_names:
            Re = int(base_path.split('_')[-1]) #Extract the Re value from the path
            Re_tensor = torch.tensor(Re, dtype=torch.float32)
            Re_tensor_expanded = Re_tensor.expand(loaded_fields[0].shape[0], 1, loaded_fields[0].shape[2], loaded_fields[0].shape[3])
            loaded_fields += [Re_tensor_expanded] #loaded_fields[2].shape (3, 1, 1024, 256) (1 is fixed as Re is a scalar)
        ###########OTHER SIMULATION PARAMETERS like 'Ma' CAN BE ADDED HERE#############

        sample= torch.cat(loaded_fields, dim=1) #data.shape (3, 4, 1024, 256) i.e. density(1), velocity(2), Re(1) as an example. 3 are the frames and 4 are the fields and params

        ######### Z-Normalization after feature engineering #########
        if self.transform:
            sample = self.transform(sample)

        if self.mode == 'train': #one step prediction
            #split sample into input and target. sample for case 2 having 3 historic velocities and 1 GT velocity: sample.shape (4, 4, 1024, 256)
            #input_tensor.shape (3, 4, 1024, 256) and gt_tensor.shape (1, 4, 1024, 256), now reshape them to (12, 1024, 256) and (4, 1024, 256) respectively to feed into the model.
            input_tensor = sample[:-1]
            input_tensor_shape = (input_tensor.shape[0] * input_tensor.shape[1],) + input_tensor.shape[2:]
            input_tensor = input_tensor.view(input_tensor_shape)

            gt_tensor = sample[-1:]
            gt_tensor_shape = (gt_tensor.shape[0] * gt_tensor.shape[1],) + gt_tensor.shape[2:]
            gt_tensor = gt_tensor.view(gt_tensor_shape)
       
            #Remove the parameter tensor from the GT tensor if the param_names is not empty
            if self.param_names!=[]:
                gt_tensor = gt_tensor[:-1]

        else: #for inference
            if (self.case_name=="raw_in_raw_out" or self.case_name=="3_hist_raw_in_raw_out"):
                input_tensor = sample[0:self.original_seq_length-1]
                input_tensor_shape = (input_tensor.shape[0] * input_tensor.shape[1],) + input_tensor.shape[2:]
                input_tensor = input_tensor.view(input_tensor_shape)

                gt_tensor = sample[self.original_seq_length-1:]

            elif (self.case_name=="acc_in_acc_out" or self.case_name=="2_hist_acc_in_acc_out"):
                #sample itself has the difference of the fields
                input_tensor = sample[0:self.original_seq_length-2]
                input_tensor_shape = (input_tensor.shape[0] * input_tensor.shape[1],) + input_tensor.shape[2:]
                input_tensor = input_tensor.view(input_tensor_shape)

                gt_tensor = sample[self.original_seq_length-2:]

            elif self.case_name=="acc_and_raw_in_raw_out":
                input_tensor = sample[0:self.original_seq_length-1] #sample[0] has the difference_info and sample[1] has the raw data
                input_tensor_shape = (input_tensor.shape[0] * input_tensor.shape[1],) + input_tensor.shape[2:]
                input_tensor = input_tensor.view(input_tensor_shape)

                gt_tensor = sample[self.original_seq_length-1:] #only has raw data u^{n+1}

            else:
                raise ValueError("Case name not recognized")

            #Remove the parameter tensor from the GT tensor if the param_names is not empty
            if self.param_names!=[]:
                gt_tensor = gt_tensor[:,:-1,:,:]

        return input_tensor, gt_tensor

class DataTransform(object):

    def __init__(self, mean_info:Dict, std_info:Dict, field_names:str, param_names:str, case_name:str):
        self.mean = mean_info
        self.std = std_info
        self.field_names = field_names
        self.param_names = param_names
        self.case_name = case_name

        #TODO : Make the mean and std as a dictionary with field names as keys and mean and std as values
    def __call__(self, sample):
        """
        # normalization to std. normal distr. with zero mean and unit std via statistics from whole dataset
        # ORDER (fields): density, velocity_x, velocity_y, pressure, Re, Ma
        # filter_list=[]
        # if "density" in self.field_names:
        #     filter_list += [0]
        # if "velocity" in self.field_names:
        #     filter_list += [1,2]
        # if "pressure" in self.field_names:
        #     filter_list += [3]
        # if "Re" in self.param_names:
        #     filter_list += [4]
        # if "Ma" in self.param_names:
        #     filter_list += [5]

        #all_filter = torch.tensor(filter_list)
        #param_filter = all_filter[-len(self.param_names):]
        #write an assert to check of all the data is normalized i.e. all values of sample are between -1 and 1

        #mean_info =self.mean[all_filter].reshape((1, -1, 1, 1)) #mean_info.shape (1, 4, 1, 1) for velocity (2), density(1), Re(1)
        
        sample_copy = sample.clone()
        
        if self.case_name=="acc_and_raw_in_raw_out" and self.field_names==['density','velocity']:
            sample[0,0,:,:] = (sample[0,0,:,:] - self.mean[0]) / self.std[0]     #normalizing the density derivative
            sample[0,1,:,:] = (sample[0,1,:,:] - self.mean[1]) / self.std[1]     #normalizing the x-velocity derivative (acc_x)
            sample[0,2,:,:] = (sample[0,2,:,:] - self.mean[2]) / self.std[2]     #normalizing the y-velocity derivative (acc_y)
            sample[1:,0,:,:] = (sample[1:,0,:,:] - self.mean[3]) / self.std[3]   #normalizing density
            sample[1:,1,:,:] = (sample[1:,1,:,:] - self.mean[4]) / self.std[4]   #normalizing x-velocity
            sample[1:,2,:,:] = (sample[1:,2,:,:] - self.mean[5]) / self.std[5]   #normalizing y-velocity
            if self.param_names!=[]:
                sample[0:,3,:,:] = (sample[0:,3,:,:] - self.mean[6]) / self.std[6] #normalizing Re
        
        elif self.case_name=="acc_and_raw_in_raw_out" and self.field_names==['velocity']:
            sample[0,0,:,:] = (sample[0,0,:,:] - self.mean[0]) / self.std[0]    #normalizing the x-velocity derivative (acc_x)
            sample[0,1,:,:] = (sample[0,1,:,:] - self.mean[1]) / self.std[1]   #normalizing the y-velocity derivative (acc_y)
            sample[1:,0,:,:] = (sample[1:,0,:,:] - self.mean[2]) / self.std[2] #normalizing x-velocity (1:3 means 1 and 2 included--> normalizing x-vel of input and gt)
            sample[1:,1,:,:] = (sample[1:,1,:,:] - self.mean[3]) / self.std[3] #normalizing y-velocity
            if self.param_names!=[]:
                sample[0:,2,:,:] = (sample[0:,2,:,:] - self.mean[4]) / self.std[4]

        else:
            #if self.param_names is empty then (to be fixed if number of parameters are more than 1)
            if self.param_names==[]:
                mean=self.mean.reshape((1, -1, 1, 1))
                std = self.std.reshape((1, -1, 1, 1))
                sample = (sample - mean[:,:-1,:,:]) / std[:,:-1,:,:]
            else:
                mean=self.mean.reshape((1, -1, 1, 1)) #mean.shape (1, 4, 1, 1) for density(1), velocity(2) and Re(1)
                std = self.std.reshape((1, -1, 1, 1)) 
                sample = (sample - mean) / std
        """
        ##################################################################################################################################################
        #### Case 1 and Case 2 both only have raw data in the input and output
        if self.case_name=="raw_in_raw_out" or self.case_name=="3_hist_raw_in_raw_out" and self.field_names==['density','velocity']:
            sample[:,0,:,:] = (sample[:,0,:,:] - self.mean['rho']) / self.std['rho']
            sample[:,1,:,:] = (sample[:,1,:,:] - self.mean['u']) / self.std['u']
            sample[:,2,:,:] = (sample[:,2,:,:] - self.mean['v']) / self.std['v']
            if self.param_names!=[]:
                sample[:,3,:,:] = (sample[:,3,:,:] - self.mean['Re']) / self.std['Re']

        elif self.case_name=="raw_in_raw_out" or self.case_name=="3_hist_raw_in_raw_out" and self.field_names==['velocity']:
            sample[:,0,:,:] = (sample[:,0,:,:] - self.mean['u']) / self.std['u']
            sample[:,1,:,:] = (sample[:,1,:,:] - self.mean['v']) / self.std['v']
            if self.param_names!=[]:
                sample[:,2,:,:] = (sample[:,2,:,:] - self.mean['Re']) / self.std['Re']

        #### Case 3 has both raw data and gradient of density in the input and only raw data in the output
        elif self.case_name=="raw_and_grad_rho_in_raw_out" and self.field_names==['density','velocity']:
            #not implemented yet
            raise NotImplementedError()

        elif self.case_name=="raw_and_grad_rho_in_raw_out" and self.field_names==['velocity']:
            #not implemented yet
            raise NotImplementedError()

        #### Case 4 and Case 5 both have acceleration in the input and output
        elif self.case_name=="acc_in_acc_out" or self.case_name=="2_hist_acc_in_acc_out" and self.field_names==['density','velocity']:
            sample[:,0,:,:] = (sample[:,0,:,:] - self.mean['drho_dt']) / self.std['drho_dt']
            sample[:,1,:,:] = (sample[:,1,:,:] - self.mean['du_dt']) / self.std['du_dt']
            sample[:,2,:,:] = (sample[:,2,:,:] - self.mean['dv_dt']) / self.std['dv_dt']
            if self.param_names!=[]:
                sample[:,3,:,:] = (sample[:,3,:,:] - self.mean['Re']) / self.std['Re']

        elif self.case_name=="acc_in_acc_out" or self.case_name=="2_hist_acc_in_acc_out" and self.field_names==['velocity']:
            sample[:,0,:,:] = (sample[:,0,:,:] - self.mean['du_dt']) / self.std['du_dt']
            sample[:,1,:,:] = (sample[:,1,:,:] - self.mean['dv_dt']) / self.std['dv_dt']
            if self.param_names!=[]:
                sample[:,2,:,:] = (sample[:,2,:,:] - self.mean['Re']) / self.std['Re']
        
        #### Case 6 has both acceleration and raw data in the input and only raw data in the output
        elif self.case_name=="acc_and_raw_in_raw_out" and self.field_names==['density','velocity']:
            sample[0,0,:,:] = (sample[0,0,:,:] - self.mean['drho_dt']) / self.std['drho_dt'] #idx 0 is the acceleration of density and velocity.
            sample[0,1,:,:] = (sample[0,1,:,:] - self.mean['du_dt']) / self.std['du_dt']
            sample[0,2,:,:] = (sample[0,2,:,:] - self.mean['dv_dt']) / self.std['dv_dt']
            sample[1:,0,:,:] = (sample[1:,0,:,:] - self.mean['rho']) / self.std['rho'] #idx 1 onwards it is raw data
            sample[1:,1,:,:] = (sample[1:,1,:,:] - self.mean['u']) / self.std['u']
            sample[1:,2,:,:] = (sample[1:,2,:,:] - self.mean['v']) / self.std['v']
            if self.param_names!=[]:
                sample[0:,3,:,:] = (sample[0:,3,:,:] - self.mean['Re']) / self.std['Re']

        elif self.case_name=="acc_and_raw_in_raw_out" and self.field_names==['velocity']:
            sample[0,0,:,:] = (sample[0,0,:,:] - self.mean['du_dt']) / self.std['du_dt']
            sample[0,1,:,:] = (sample[0,1,:,:] - self.mean['dv_dt']) / self.std['dv_dt']
            sample[1:,0,:,:] = (sample[1:,0,:,:] - self.mean['u']) / self.std['u']
            sample[1:,1,:,:] = (sample[1:,1,:,:] - self.mean['v']) / self.std['v']
            if self.param_names!=[]:
                sample[0:,2,:,:] = (sample[0:,2,:,:] - self.mean['Re']) / self.std['Re']
        
        return sample
    