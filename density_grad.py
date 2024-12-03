import numpy as np
import torch
from scipy.ndimage import binary_dilation

def average_neighbor_density(density, boundary_mask):
    avg_density = torch.zeros_like(density)
    #print(density.shape)
    # Get the indices of the boundary
    boundary_indices = torch.argwhere(boundary_mask)

    for idx in boundary_indices:
        i, j = idx[1], idx[2]
        
        # Extract neighboring values (considering a 3x3 window)
        neighbors = density[0, max(i-1, 0):min(i+2, density.shape[1]),
                            max(j-1, 0):min(j+2, density.shape[2])]

        # Exclude zero values (obstacle cells)
        valid_neighbors = neighbors[neighbors != 0]

        # Calculate the average of valid neighbors
        if valid_neighbors.shape[0]> 0:
            avg_density[0, i, j] = torch.mean(valid_neighbors)
        else:
            avg_density[0, i, j] = 0  # or some default value

    return avg_density


def get_density_grad(mask, data):
    mask = mask.cpu()
    # Dilate the mask to find boundary (this will create a region around obstacles)
    dilated_mask = torch.tensor(binary_dilation(mask))

    # Identify the boundary by subtracting the obstacle mask from the dilated mask
    boundary_mask = dilated_mask & ~mask

    #TODO: Vectorize this
    # Fill the boundary with the average density
    for i in range(data.shape[0]): #loop over the timesteps
        average_density_array= average_neighbor_density(data[i], boundary_mask[i])

        for idx in torch.argwhere(boundary_mask[i]):
            j, k = idx[1], idx[2]
            data[i, 0, j, k] = average_density_array[0, j, k]

    new_density_array = torch.zeros((data.shape[0], data.shape[1], data.shape[2]+2, data.shape[3]+2))

    # Copy the original array into the middle of the new array
    new_density_array[:, :, 1:-1, 1:-1] = data

    # Copy the boundaries
    # Top and bottom rows
    new_density_array[:, :, 0, 1:-1] = data[:, :, 0, :]
    new_density_array[:, :, -1, 1:-1] = data[:, :, -1, :]

    # Left and right columns
    new_density_array[:, :, 1:-1, 0] = data[:, :, :, 0]
    new_density_array[:, :, 1:-1, -1] = data[:, :, :, -1]

    # Corners
    new_density_array[:, :, 0, 0] = data[:, :, 0, 0]        # Top-left
    new_density_array[:, :, 0, -1] = data[:, :, 0, -1]      # Top-right
    new_density_array[:, :, -1, 0] = data[:, :, -1, 0]      # Bottom-left
    new_density_array[:, :, -1, -1] = data[:, :, -1, -1]    # Bottom-right

    #take the gradient of the density field using (HARDCODED)
    dx = 0.1/64

    grad_height = torch.gradient(new_density_array, spacing=dx, dim=2)[0] #across the rows (height)

    # Gradient along the width (axis 3)
    grad_width = torch.gradient(new_density_array, spacing=dx, dim=3)[0] #across the columns (width)

    # The first channel (index 1) will be the height gradient, the second (index 2) will be the width gradient
    # We extract the first component from grad_height and grad_width since they return arrays of the same shape
    spatial_gradients = torch.cat((grad_height, grad_width), dim=1)

    #discard the wall boundaries from the spatial gradient
    spatial_gradients = spatial_gradients[:, :, 1:-1, 1:-1]

    spatial_gradients= spatial_gradients * mask #Sets the final layer of the cylinder waLL to 0.

    feature_x = (spatial_gradients[:, 0:1, :, :] / (data[:, 0:1, :, :])) # spatial gradient along the height (#data doesnt have 0s in the cylinder)
    feature_y = (spatial_gradients[:, 1:2, :, :] / (data[:, 0:1, :, :])) # spatial gradient along the width
    
    grad_feature = torch.cat((feature_x, feature_y), dim=1)

    return grad_feature
