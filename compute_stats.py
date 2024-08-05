import os
import torch

path_to_files = "./VortexStreet_Re_200_2.6_sec"
#load the density and velocity data
density_data = torch.load(os.path.join(path_to_files, 'density_karman_vortex_2.6.pt'))
velocity_data = torch.load(os.path.join(path_to_files, 'velocity_karman_vortex_2.6.pt'))

print("Shape of the density data: ", density_data.shape)
print("Shape of the velocity data: ", velocity_data.shape)

density_mean = density_data.mean()
density_std = density_data.std()
x_velocity_mean = velocity_data[:,0,:,:].mean()
x_velocity_std = velocity_data[:,0,:,:].std()
y_velocity_mean = velocity_data[:,1,:,:].mean()
y_velocity_std = velocity_data[:,1,:,:].std()
v_mag = torch.sqrt(velocity_data[:,0,:,:]**2 + velocity_data[:,1,:,:]**2)
velocity_magnitude_mean = v_mag.mean()
velocity_magnitude_std = v_mag.std()

print("Density Mean: ", density_mean)
print("Density Std: ", density_std)
print("x-Velocity Mean: ", x_velocity_mean)
print("x-Velocity Std: ", x_velocity_std)
print("y-Velocity Mean: ", y_velocity_mean)
print("y-Velocity Std: ", y_velocity_std)
print("Velocity Magnitude Mean: ", velocity_magnitude_mean)
print("Velocity Magnitude Std: ", velocity_magnitude_std)
#find the velocity magnitude

#save the mean and variance as a .txt file
path_to_save = "./VortexStreet_Re_200_2.6_sec"
with open(os.path.join(path_to_save, 'mean_variance_2.6_sec.txt'), 'w') as f:
    f.write("Resolution of the data: \n")
    f.write("Shape of the density data: " + str(density_data.shape) + "\n")
    f.write("Shape of the velocity data: " + str(velocity_data.shape) + "\n\n")
    f.write("Mean and Variance of the data: \n")
    f.write("Density Mean: " + str(density_mean.item()) + "\n")
    f.write("Density Std: " + str(density_std.item()) + "\n")
    f.write("x_Velocity Mean: " + str(x_velocity_mean.item()) + "\n")
    f.write("x_Velocity Std: " + str(x_velocity_std.item()) + "\n")
    f.write("y_Velocity Mean: " + str(y_velocity_mean.item()) + "\n")
    f.write("y_Velocity Std: " + str(y_velocity_std.item()) + "\n")
    f.write("Velocity Magnitude Mean: " + str(velocity_magnitude_mean.item()) + "\n")
    f.write("Velocity Magnitude Std: " + str(velocity_magnitude_std.item()) + "\n")
