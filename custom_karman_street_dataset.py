import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import os
from hydra.utils import get_original_cwd
##Datasets have arrived from the folder "final_dataset_for_train". Now build the CustomDataset

class CustomDataset(Dataset):
    def __init__(self, base_folder: str, 
                 field_names: List[str], 
                 param_names: List[str], 
                 param_values: List,
                 case_name: str,
                 filter_frame:List[Tuple[int,int]]=[], 
                 sequence_info: List[Tuple[int,int]]=[]):
        
        self.case_name = case_name
        self.transform = None
        self.data_dir = base_folder 
        self.field_names = field_names  #velocity, density or both
        self.param_names = param_names  #Re, Ma or both
        self.param_values = param_values 
        self.min_frame = filter_frame[0][0]
        self.max_frame = filter_frame[0][1]
        self.seq_length= sequence_info[0][0] # includes n-1 historic info and 1 current info (GT)
        self.seq_stride = sequence_info[0][1] # stride between sequences

        self.data_paths = []
        ######################################
        ### Stage 1 : Generate data paths list
        ######################################
        top_dir= os.path.join(get_original_cwd(), self.data_dir)
        top_dir_folders= os.listdir(os.path.join(get_original_cwd(), self.data_dir))
        top_dir_folders.sort()

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
        base_path, seq_start, seq_end, seq_stride = self.data_paths[idx] #basePath = '.../Re_200', seqStart = 661, seqEnd = 665, seqSkip = 2, say for idx 11
        #seq_len = int((seq_end - seq_start) / seq_stride)

        loaded = {} #empty dictionary
        for field in self.field_names:
            loaded[field] = []

        for frame in range(seq_start, seq_end, seq_stride): #this loop executes 3 times as seqSkip = 2. 
            for field in self.field_names:
                loaded_arr = torch.load(os.path.join(base_path, "%s_%04d.pt" % (field, frame)))
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
            raise NotImplementedError()

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
                loaded_fields[i][0] = loaded_fields[i][1] - loaded_fields[i][0] #other entries of the loaded_fields[i] array are already in the form of delta_u^n 

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
        return sample

class DataTransform(object):

    def __init__(self, mean_info: List, std_info:List, field_names:str, param_names:str, case_name:str):
        self.mean = torch.tensor(mean_info)
        self.std = torch.tensor(std_info)
        self.field_names = field_names
        self.param_names = param_names
        self.case_name = case_name

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
        """
        if self.case_name=="acc_and_raw_in_raw_out":
            sample[0,0,:,:] = (sample[0,0,:,:] - self.mean[0]) / self.std[0]     #normalizing the density derivative
            sample[0,1,:,:] = (sample[0,1,:,:] - self.mean[1]) / self.std[1]     #normalizing the x-velocity derivative (acc_x)
            sample[0,2,:,:] = (sample[0,2,:,:] - self.mean[2]) / self.std[2]     #normalizing the y-velocity derivative (acc_y)
            sample[1:3,0,:,:] = (sample[1:3,0,:,:] - self.mean[3]) / self.std[3] #normalizing density
            sample[1:3,1,:,:] = (sample[1:3,1,:,:] - self.mean[4]) / self.std[4] #normalizing x-velocity
            sample[1:3,2,:,:] = (sample[1:3,2,:,:] - self.mean[5]) / self.std[5] #normalizing y-velocity
            sample[0:3,3,:,:] = (sample[0:3,3,:,:] - self.mean[6]) / self.std[6] #normalizing Re
            
        else:
            mean=self.mean.reshape((1, -1, 1, 1)) #mean_info.shape (1, 4, 1, 1) for density(1), velocity(2) and Re(1)
            std = self.std.reshape((1, -1, 1, 1)) 
            sample = (sample - mean) / std

        return sample
    