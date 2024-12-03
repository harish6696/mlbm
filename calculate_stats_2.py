import numpy as np
import torch
from pathlib import Path
from hydra.utils import to_absolute_path
import os
import h5py
from scipy.ndimage import binary_dilation

base_folder="VS_Re_100_to_500_uniform_skip_20_256x64/train"
res_x= 256
res_y= 64
mask_file_location = "inverted_mask_256x64.h5"
#load the h5 file
mask_file = h5py.File(mask_file_location, 'r')
mask_256x64 = mask_file['mask'][:]

Re_list=[100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500]
#Re_list=[100,200,300,400,500]

density_mean_list=[]
density_var_list=[]

density_acc_mean_list=[]
density_acc_var_list=[]

grad_rho_x_mean_list=[]
grad_rho_x_var_list=[]

grad_rho_y_mean_list=[]
grad_rho_y_var_list=[]

x_velocity_mean_list=[]
x_velocity_var_list=[]

y_velocity_mean_list=[]
y_velocity_var_list=[]

x_acc_mean_list=[]
x_acc_var_list=[]

y_acc_mean_list=[]
y_acc_var_list=[]

def average_neighbor_field(field, boundary_mask): #function to fill the boundary cells of the 
    #cylinder with a 3x3 average of the surrounding cells. field.shape = (1, 256, 64)
    avg_field = np.zeros_like(field)
    # Get the indices of the boundary
    boundary_indices = np.argwhere(boundary_mask)
    for idx in boundary_indices:
        i, j = idx[1], idx[2]
        # Extract neighboring values (considering a 3x3 window)
        neighbors = field[0, max(i-1, 0):min(i+2, field.shape[1]),
                            max(j-1, 0):min(j+2, field.shape[2])]
        # Exclude zero values (obstacle cells)
        valid_neighbors = neighbors[neighbors != 0]
        # Calculate the average of valid neighbors
        if valid_neighbors.shape[0]> 0:
            avg_field[0, i, j] = np.mean(valid_neighbors)
        else:
            avg_field[0, i, j] = 0  # or some default value
    return avg_field

def compute_density_grad(density_data_masked, mask, grid_spacing):
    dilated_mask = binary_dilation(mask)
    # Identify the boundary by subtracting the obstacle mask from the dilated mask
    boundary_mask = dilated_mask & ~mask

    # Fill the boundary with the average density
    for i in range(density_data_masked.shape[0]): #loop over the timesteps
        average_density_array= average_neighbor_field(density_data_masked[i], boundary_mask[i])
        for idx in np.argwhere(boundary_mask[i]):
            j, k = idx[1], idx[2]
            density_data_masked[i, 0, j, k] = average_density_array[0, j, k]

    ##### Copy the density values into the walls
    new_density_array = np.zeros((density_data_masked.shape[0], density_data_masked.shape[1], density_data_masked.shape[2]+2, density_data_masked.shape[3]+2))

    # Copy the original array into the middle of the new array
    new_density_array[:, :, 1:-1, 1:-1] = density_data_masked

    # Copy the boundaries
    # Top and bottom rows
    new_density_array[:, :, 0, 1:-1] = density_data_masked[:, :, 0, :]
    new_density_array[:, :, -1, 1:-1] = density_data_masked[:, :, -1, :]

    # Left and right columns
    new_density_array[:, :, 1:-1, 0] = density_data_masked[:, :, :, 0]
    new_density_array[:, :, 1:-1, -1] = density_data_masked[:, :, :, -1]

    # Corners
    new_density_array[:, :, 0, 0] = density_data_masked[:, :, 0, 0]        # Top-left
    new_density_array[:, :, 0, -1] = density_data_masked[:, :, 0, -1]      # Top-right
    new_density_array[:, :, -1, 0] = density_data_masked[:, :, -1, 0]      # Bottom-left
    new_density_array[:, :, -1, -1] = density_data_masked[:, :, -1, -1]   # Bottom-right

    grad_height = np.gradient(new_density_array, grid_spacing, axis=2) #across the rows (height) i.e. along the height of the channel
    # Gradient along the width (axis 3)
    grad_length = np.gradient(new_density_array, grid_spacing, axis=3) #across the columns (length) i.e. along the length of the channel

    # Stack the gradients into a new array with shape [100, 2, 1026, 258]
    # The first channel (index 1) will be the height gradient, the second (index 2) will be the width gradient
    # We extract the first component from grad_height and grad_width since they return arrays of the same shape
    spatial_gradients = np.concatenate((grad_height, grad_length), axis=1)
    #print(spatial_gradients.shape)  # Should print (100, 2, 1026, 258)
    #discard the boundaries from the spatial gradient
    spatial_gradients = spatial_gradients[:, :, 1:-1, 1:-1]

    spatial_gradients_masked = spatial_gradients * mask

    non_zero_mask = mask[:, 0:1, :, :] != 0

    feature_y = np.zeros_like(spatial_gradients_masked[:, 1:2, :, :]) #length gradient
    feature_x = np.zeros_like(spatial_gradients_masked[:, 0:1, :, :]) #height gradient
    feature_y[non_zero_mask] = spatial_gradients_masked[:, 1:2, :, :][non_zero_mask] / (density_data_masked[:, 0:1, :, :])[non_zero_mask] # spatial gradient along the width
    feature_x[non_zero_mask] = spatial_gradients_masked[:, 0:1, :, :][non_zero_mask] / (density_data_masked[:, 0:1, :, :])[non_zero_mask]

    grad_rho_feature = np.concatenate((feature_x, feature_y), axis=1) #across the rows (height gradient) and across the columns(length gradient)
    #print(grad_rho_feature.shape)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(5, 4, figsize=(20, 10))
    # for idx, i in enumerate(range(350, 354)):
    #     im = axs[0, idx].imshow(density_data_masked[i, 0, :, :], cmap='viridis')
    #     im2 = axs[1, idx].imshow(spatial_gradients_masked[i, 1, :, :], cmap='viridis') # only gradient along the width
    #     im3 = axs[2, idx].imshow(spatial_gradients_masked[i, 0, :, :], cmap='viridis') # only gradient along the height
    #     im4 = axs[3, idx].imshow(feature_x[i, 0, :, :], cmap='viridis') # rho_gradient/rho along the channel length
    #     im5 = axs[4, idx].imshow(feature_y[i, 0, :, :], cmap='viridis') # rho_gradient/rho along the height of the channel
        
    #     # axs[i].axis('off')
    #     fig.colorbar(im, ax=axs[0, idx])
    #     fig.colorbar(im2, ax=axs[1, idx])
    #     fig.colorbar(im3, ax=axs[2, idx])
    #     fig.colorbar(im4, ax=axs[3, idx])
    #     fig.colorbar(im5, ax=axs[4, idx])
    # #save the figure
    # plt.savefig("density_grad.png")
    # plt.show()

    return grad_rho_feature

#####################
# Main Loop
#####################
grid_spacing = 0.1/64
for Re in Re_list:
    Re_folder_path=os.path.join(base_folder, f"Re_{Re}")
    density_files = [f for f in os.listdir(Re_folder_path) if f.startswith('density')]
    velocity_files = [f for f in os.listdir(Re_folder_path) if f.startswith('velocity')]
    density_files.sort(key=lambda x: float(x.split("_")[-1][:-3]))
    velocity_files.sort(key=lambda x: float(x.split("_")[-1][:-3]))

    density_data = np.zeros([len(density_files),1,res_x,res_y])
    velocity_data = np.zeros([len(velocity_files),2,res_x,res_y])
    
    for i,file in enumerate(density_files):
        with h5py.File(os.path.join(Re_folder_path, file), 'r+') as h5_file:
            density_data[i,:,:,:] = h5_file['density'][:]
    
    density_acc = density_data[1:] - density_data[:-1]
    
    mask = np.tile(mask_256x64, (density_data.shape[0], 1, 1, 1))
    density_data_masked = density_data*mask
    density_acc_masked = density_acc*mask[:-1] 
    #####start from here
    ###mask has 76 0s and density_data has only 52...Why?
    grad_rho_features=compute_density_grad(density_data_masked, mask, grid_spacing)

    #density stats 
    density_mean_list.append(np.mean(density_data_masked))
    density_var_list.append(np.var(density_data_masked))
    density_acc_mean_list.append(np.mean(density_acc_masked))
    density_acc_var_list.append(np.var(density_acc_masked))
    grad_rho_x_mean_list.append(np.mean(grad_rho_features[:, 0, :, :]))
    grad_rho_x_var_list.append(np.var(grad_rho_features[:, 0, :, :]))
    grad_rho_y_mean_list.append(np.mean(grad_rho_features[:, 1, :, :]))
    grad_rho_y_var_list.append(np.var(grad_rho_features[:, 1, :, :]))
    ################################################################################
    for i, file in enumerate(velocity_files):
        with h5py.File(os.path.join(Re_folder_path, file), 'r+') as h5_file:
            velocity_data[i,:,:,:] = h5_file['velocity'][:]

    acc = velocity_data[1:] - velocity_data[:-1]
    
    #velocity stats
    velocity_mean_per_Re = np.mean(velocity_data, axis=(0,2,3))
    x_velocity_mean_per_Re = velocity_mean_per_Re[0]
    y_velocity_mean_per_Re = velocity_mean_per_Re[1]

    velocity_var_per_Re = np.var(velocity_data,  axis=(0,2,3))
    x_velocity_var_per_Re = velocity_var_per_Re[0]
    y_velocity_var_per_Re = velocity_var_per_Re[1]

    x_velocity_mean_list.append(x_velocity_mean_per_Re)
    x_velocity_var_list.append(x_velocity_var_per_Re)
    y_velocity_mean_list.append(y_velocity_mean_per_Re)
    y_velocity_var_list.append(y_velocity_var_per_Re)

    #acceleration stats
    acc_mean_per_Re = np.mean(acc, axis=(0,2,3))
    x_acc_mean_per_Re = acc_mean_per_Re[0]
    y_acc_mean_per_Re = acc_mean_per_Re[1]

    acc_var_per_Re = np.var(acc,  axis=(0,2,3))
    x_acc_var_per_Re = acc_var_per_Re[0]
    y_acc_var_per_Re = acc_var_per_Re[1]

    x_acc_mean_list.append(x_acc_mean_per_Re)
    x_acc_var_list.append(x_acc_var_per_Re)
    y_acc_mean_list.append(y_acc_mean_per_Re)
    y_acc_var_list.append(y_acc_var_per_Re)


# #print all stats
print("Density Mean: ", density_mean_list)
print("Density Variance: ", density_var_list)
print("Density Acc Mean: ", density_acc_mean_list)
print("Density Acc Variance: ", density_acc_var_list)
print("Grad_rho_x Mean: ", grad_rho_x_mean_list)
print("Grad_rho_x Variance: ", grad_rho_x_var_list)
print("Grad_rho_y Mean: ", grad_rho_y_mean_list)
print("Grad_rho_y Variance: ", grad_rho_y_var_list)
print("\n")
print("X Velocity Mean: ", x_velocity_mean_list)
print("X Velocity Variance: ", x_velocity_var_list)
print("Y Velocity Mean: ", y_velocity_mean_list)
print("Y Velocity Variance: ", y_velocity_var_list)
print("X Acceleration Mean: ", x_acc_mean_list)
print("X Acceleration Variance: ", x_acc_var_list)
print("Y Acceleration Mean: ", y_acc_mean_list)
print("Y Acceleration Variance: ", y_acc_var_list)

##############################################
# Overall statistics across all Re values
##############################################
#compute the overall mean from the mean list
density_mean = np.mean(density_mean_list)
density_acc_mean = np.mean(density_acc_mean_list)

grad_rho_x_mean = np.mean(grad_rho_x_mean_list)
grad_rho_y_mean = np.mean(grad_rho_y_mean_list)

x_velocity_mean = np.mean(x_velocity_mean_list)
y_velocity_mean = np.mean(y_velocity_mean_list)

x_acc_mean = np.mean(x_acc_mean_list)
y_acc_mean = np.mean(y_acc_mean_list)

#compute the overall variance from the variance list
overall_density_var = 0
overall_density_acc_var = 0
overall_grad_rho_x_var = 0
overall_grad_rho_y_var = 0
overall_x_velocity_var = 0
overall_y_velocity_var = 0
overall_x_acc_var = 0
overall_y_acc_var = 0

for i in range(len(density_var_list)):
    term1 = density_var_list[i] + (density_mean_list[i] - density_mean)**2
    overall_density_var = overall_density_var + term1

    term2 = density_acc_var_list[i] + (density_acc_mean_list[i] - density_acc_mean)**2
    overall_density_acc_var = overall_density_acc_var + term2

    term3 = x_velocity_var_list[i] + (x_velocity_mean_list[i] - x_velocity_mean)**2
    overall_x_velocity_var = overall_x_velocity_var + term3

    term4 = y_velocity_var_list[i] + (y_velocity_mean_list[i] - y_velocity_mean)**2
    overall_y_velocity_var = overall_y_velocity_var + term4

    term5 = x_acc_var_list[i] + (x_acc_mean_list[i] - x_acc_mean)**2
    overall_x_acc_var = overall_x_acc_var + term5

    term6 = y_acc_var_list[i] + (y_acc_mean_list[i] - y_acc_mean)**2
    overall_y_acc_var = overall_y_acc_var + term6

    term7 = grad_rho_x_var_list[i] + (grad_rho_x_mean_list[i] - grad_rho_x_mean)**2
    overall_grad_rho_x_var = overall_grad_rho_x_var + term7

    term8 = grad_rho_y_var_list[i] + (grad_rho_y_mean_list[i] - grad_rho_y_mean)**2
    overall_grad_rho_y_var = overall_grad_rho_y_var + term8

#compute the standard deviation
overall_density_std = np.sqrt(overall_density_var/len(density_var_list))
overall_density_acc_std = np.sqrt(overall_density_acc_var/len(density_acc_var_list))
overall_grad_rho_x_std = np.sqrt(overall_grad_rho_x_var/len(grad_rho_x_var_list))
overall_grad_rho_y_std = np.sqrt(overall_grad_rho_y_var/len(grad_rho_y_var_list))
overall_x_velocity_std = np.sqrt(overall_x_velocity_var/len(x_velocity_var_list))
overall_y_velocity_std = np.sqrt(overall_y_velocity_var/len(y_velocity_var_list))
overall_x_acc_std = np.sqrt(overall_x_acc_var/len(x_acc_var_list))
overall_y_acc_std = np.sqrt(overall_y_acc_var/len(y_acc_var_list))

#print the overall mean and variance
print("\n rho_mean: ", density_mean)
print(" rho_std: ", overall_density_std)
print("\n drho_dt Mean: ", density_acc_mean)
print("drho_dt std: ", overall_density_acc_std)

print("\n grad_rho_by_rho_x_mean: ", grad_rho_x_mean)
print(" grad_rho_by_rho_x_std: ", overall_grad_rho_x_std)
print("\n grad_rho_by_rho_y_mean: ", grad_rho_y_mean)
print(" grad_rho_by_rho_y_std: ", overall_grad_rho_y_std)

print("\n u_mean: ", x_velocity_mean)
print(" u_std: ", overall_x_velocity_std)
print("\n v_mean: ", y_velocity_mean)
print("v_std: ", overall_y_velocity_std)


print("\n du_dt Mean: ", x_acc_mean)
print("du_dt std: ", overall_x_acc_std)
print("\n dv_dt Mean: ", y_acc_mean)
print("dv_dt std: ", overall_y_acc_std)

#compute the mean and std of Re values
Re_mean = np.mean(Re_list)
Re_std = np.std(Re_list)
print("\n Re mean: ", np.mean(Re_list))
print("Re std: ", np.std(Re_list))


means = {
    "rho": density_mean,
    "u": x_velocity_mean,
    "v": y_velocity_mean,
    "drho_dt": density_acc_mean,
    "du_dt": x_acc_mean,
    "dv_dt": y_acc_mean,
    "grad_rho_x/_rho": grad_rho_x_mean,
    "grad_rho_y/_rho": grad_rho_y_mean,
    "Re": Re_mean  # Replace `None` with the value for Re if available
}

# Standard deviations dictionary
stds = {
    "rho": overall_density_std,
    "u": overall_x_velocity_std,
    "v": overall_y_velocity_std,
    "drho_dt": overall_density_acc_std,
    "du_dt": overall_x_acc_std,
    "dv_dt": overall_y_acc_std,
    "grad_rho_x/_rho": overall_grad_rho_x_std,
    "grad_rho_y/_rho": overall_grad_rho_y_std,
    "Re": Re_std  # Replace `None` with the value for Re if available
}

# Print the dictionaries
print("Means: ", means)
print("Standard Deviations: ", stds)


# density_data_list = []
# velocity_data_list = []

# density_acc_list = []
# acc_list = []

# for Re in Re_list:
#     Re_folder_path=os.path.join(base_folder, f"Re_{Re}")
#     density_files = [f for f in os.listdir(Re_folder_path) if f.startswith('density')]
#     velocity_files = [f for f in os.listdir(Re_folder_path) if f.startswith('velocity')]
#     density_files.sort(key=lambda x: float(x.split("_")[-1][:-3]))
#     velocity_files.sort(key=lambda x: float(x.split("_")[-1][:-3]))

#     density_data = np.zeros([len(density_files),1,res_x,res_y])
#     velocity_data = np.zeros([len(velocity_files),2,res_x,res_y])
    
#     for i,file in enumerate(density_files):
#         with h5py.File(os.path.join(Re_folder_path, file), 'r+') as h5_file:
#             density_data[i,:,:,:] = h5_file['density'][:]

#     density_acc = density_data[1:] - density_data[:-1]

#     density_data_list.append(density_data)
#     density_acc_list.append(density_acc)

#     #
    
#     for i, file in enumerate(velocity_files):
#         with h5py.File(os.path.join(Re_folder_path, file), 'r+') as h5_file:
#             velocity_data[i,:,:,:] = h5_file['velocity'][:]

#     acc = velocity_data[1:] - velocity_data[:-1]

#     velocity_data_list.append(velocity_data)
#     acc_list.append(acc)

# density_data_list = np.array(density_data_list)
# velocity_data_list = np.array(velocity_data_list)

# density_acc_list = np.array(density_acc_list)
# acc_list = np.array(acc_list)

#print the mean and std of the data
# print("Testing")
# print("\n Density Data Mean: ", np.mean(density_data_list))
# print("Density Data Std: ", np.std(density_data_list))

# print("\n Density Acc Mean: ", np.mean(density_acc_list))
# print("Density Acc Std: ", np.std(density_acc_list))

# print("\n Velocity Data Mean: ", np.mean(velocity_data_list, axis=(0,1,3,4)))
# print("Velocity Data Std: ", np.std(velocity_data_list, axis=(0,1,3,4)))

# print("\n Acceleration Mean: ", np.mean(acc_list, axis=(0,1,3,4)))
# print("Acceleration Std: ", np.std(acc_list, axis=(0,1,3,4)))

print("end")