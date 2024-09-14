
import numpy as np
import torch
from pathlib import Path
from hydra.utils import to_absolute_path
#append the datasets starting with the word velocity froom  the folder "final_dataset_for_train"


field_name = "mixed"
num_channels = 3

base_folder = "final_dataset_for_train"
Re_list = [200, 300, 400]
data = torch.empty([0,num_channels,1024,256])

#base_folder = "old_datasets"
#Re_list = [100, 125, 150, 175, 200]
#data = torch.empty([0,num_channels,512,256])

for Re in Re_list:
    data_file = f"{field_name}_Re_{Re}.pt"
    #join the path with the base folder
    data_file_path = Path(base_folder).joinpath(data_file)
    loaded_data = torch.load(data_file_path)
    loaded_data = loaded_data[350:]
    data = torch.cat([data, loaded_data], dim=0)
    
    
print(data.shape)
data= data.numpy()
mean = np.mean(data, axis = (0,2,3))
std= np.std(data, axis = (0,2,3))
#print the field name and the mean value
print(f"{field_name}: {mean} and {std}")
#calculate the mean and std of the Re_list
Re_mean = np.mean(np.array(Re_list))
Re_std = np.std(np.array(Re_list))
print(f"Re: mean: {Re_mean} and std: {Re_std}")

"""
For Re from 200 to 400, the mean and std of the velocity field are for grid resolution 1024x256:
velocity: 
    mean:   [ 2.9988546e-02 -1.5189227e-07 (approx 0)]
    std:    [0.0137524  0.01020263] 

density:
    mean: [1.1303548] 
    std: [0.05823061]

Re: mean: 300.0
    std: 81.64965809277261
"""

"""
For Re from 100 to 500, the mean and std of the velocity field are for grid resolution 1024x256:
velocity: 
    mean:  [2.9989550e-02 2.9060382e-07 (approx 0)] 
    std:   [0.01839873 0.01116166]

density:
    mean:[1.1166793] 
    std: [0.05635419]

Re: mean: 300.0
    std: 141.4213562373095
"""