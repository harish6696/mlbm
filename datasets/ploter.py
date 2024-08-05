import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Assuming we have a .pt file named 'velocity_data.pt'
# Load the data
velocity_data = torch.load('velocity_karman_Re_100.pt')  # Uncomment if loading from file

# For illustration, let's generate some random data with the specified shape
#velocity_data = torch.randn(100, 2, 512, 256)

# Define the directory to save the images
output_dir = 'velocity_colormaps_Re_100'
os.makedirs(output_dir, exist_ok=True)

# Iterate through each sample and save the colormap plots
for sample_index in range(velocity_data.shape[0]):
    try:
        u = velocity_data[sample_index, 0, :, :].detach().cpu().numpy()
        v = velocity_data[sample_index, 1, :, :].detach().cpu().numpy()

        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(u ** 2 + v ** 2)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.imshow(velocity_magnitude.T, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Velocity Magnitude')
        plt.title(f'Velocity Magnitude Colormap for Sample {sample_index}')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Save the figure
        output_path = os.path.join(output_dir, f'velocity_colormap_{sample_index:03d}.png')
        plt.savefig(output_path)
        plt.close()

    except Exception as e:
        print(f"Error processing sample {sample_index}: {e}")

print(f"Saved {velocity_data.shape[0]} velocity colormap frames to {output_dir}")