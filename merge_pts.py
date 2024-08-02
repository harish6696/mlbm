##################################################################################################
# This script merges the .pt files into one and also calculates the mean and vatiance of the data
##################################################################################################

path_to_files = "/local/disk1/fno_modulus/mlbm/VortexStreet_Re_200_2.6_sec/pytorch_output"

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

#print shape of the data
print("Shape of the density data: ", density_data.shape)
print("Shape of the velocity data: ", velocity_data.shape)

#calculate the mean and variance of the density and velocity
density_mean = density_data.mean()
density_variance = density_data.var()
x_velocity_mean = velocity_data[:,0,:,:].mean()
x_velocity_variance = velocity_data[:,0,:,:].var()
y_velocity_mean = velocity_data[:,1,:,:].mean()
y_velocity_variance = velocity_data[:,1,:,:].var()
v_mag = torch.sqrt(velocity_data[:,0,:,:]**2 + velocity_data[:,1,:,:]**2)
velocity_magnitude_mean = v_mag.mean()
velocity_magnitude_variance = v_mag.var()

print("Density Mean: ", density_mean)
print("Density Variance: ", density_variance)
print("x-Velocity Mean: ", x_velocity_mean)
print("x-Velocity Variance: ", x_velocity_variance)
print("y-Velocity Mean: ", y_velocity_mean)
print("y-Velocity Variance: ", y_velocity_variance)
print("Velocity Magnitude Mean: ", velocity_magnitude_mean)
print("Velocity Magnitude Variance: ", velocity_magnitude_variance)
#find the velocity magnitude

#save the mean and variance as a .txt file
path_to_save = "/local/disk1/fno_modulus/mlbm/VortexStreet_Re_200_2.6_sec"
with open(os.path.join(path_to_save, 'mean_variance_2.6_sec.txt'), 'w') as f:
    f.write("Density Mean: " + str(density_mean.item()) + "\n")
    f.write("Density Variance: " + str(density_variance.item()) + "\n")
    f.write("x_Velocity Mean: " + str(x_velocity_mean.item()) + "\n")
    f.write("x_Velocity Variance: " + str(x_velocity_variance.item()) + "\n")
    f.write("y_Velocity Mean: " + str(y_velocity_mean.item()) + "\n")
    f.write("y_Velocity Variance: " + str(y_velocity_variance.item()) + "\n")
    f.write("Velocity Magnitude Mean: " + str(velocity_magnitude_mean.item()) + "\n")
    f.write("Velocity Magnitude Variance: " + str(velocity_magnitude_variance.item()) + "\n")

#save the data as .pt file
torch.save(density_data, os.path.join(path_to_save, 'density_karman_vortex_2.6.pt'))
torch.save(velocity_data, os.path.join(path_to_save, 'velocity_karman_vortex_2.6.pt'))

