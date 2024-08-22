##################################################################################################
# This script merges the .pt files into one and also calculates the mean and vatiance of the data
##################################################################################################

path_to_files = "./VortexStreet_Re_200_2.6_sec/pytorch_output"
path_to_save = "./VortexStreet_Re_200_2.6_sec"
#combine all files which start with density into a single .pt file
import torch
import os
import numpy as np
import time

# Get the list of files
files = os.listdir(path_to_files)
density_files = [f for f in files if f.startswith('density')]
velocity_files = [g for g in files if g.startswith('velocity')]


#merge the files. data should have the shape (num_files, -1, 1)
density_data = torch.load(os.path.join(path_to_files, density_files[0]))
velocity_data = torch.load(os.path.join(path_to_files, velocity_files[0]))

#add a new dimension at the front to accumulate time series data
density_data = density_data.unsqueeze(0)
for f in density_files[1:]:
    density_data = torch.cat((density_data, torch.load(os.path.join(path_to_files, f)).unsqueeze(0)), dim=0)

velocity_data = velocity_data.unsqueeze(0)    
for g in velocity_files[1:]:     
    velocity_data = torch.cat((velocity_data, torch.load(os.path.join(path_to_files, g)).unsqueeze(0)), dim=0)

#permute the velocity data to have the shape (num_files,3,x_res, y_res)
#make the dimension (num_files, 2, x_res, y_res)

#velocity_data = velocity_data.permute(0, 2, 3, 1, 4)
velocity_data = velocity_data.squeeze(-1)
#remove the z component of velocity
velocity_data = velocity_data[:,:2,:,:]

#permute the density data to have the shape (num_files, 1, x_res, y_res)
density_data = density_data.permute(0, 3, 1, 2)

print("Shape of the density data: ", density_data.shape)
print("Shape of the velocity data: ", velocity_data.shape)

#calculate the mean and variance of the density and velocity

#save the data as .pt file
torch.save(density_data, os.path.join(path_to_save, 'density_karman_vortex_2.6.pt'))
torch.save(velocity_data, os.path.join(path_to_save, 'velocity_karman_vortex_2.6.pt'))

#load dataset
density_data = torch.load(os.path.join(path_to_save, 'density_karman_vortex_2.6.pt'))
velocity_data = torch.load(os.path.join(path_to_save, 'velocity_karman_vortex_2.6.pt'))

#merge datasets
mixed_data = torch.cat((density_data, velocity_data), dim=1)
print("Shape of the mixed data: ", mixed_data.shape)
torch.save(mixed_data, os.path.join(path_to_save, 'mixed_karman_vortex_2.6.pt'))
