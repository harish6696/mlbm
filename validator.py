# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import matplotlib.pyplot as plt
from torch import FloatTensor
from torch.nn import MSELoss
from mlflow import log_figure
import numpy as np
from vtk import vtkImageData
from vtk.util import numpy_support
from vtk import vtkXMLImageDataWriter
from pathlib import Path


class GridValidator:
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    out_dir : str, optional
        directory to which plots shall be stored
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        loss_fun,
        num_channels: int,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        out_dir: str = "./outputs/validators",
        font_size: float = 28.0,
    ):
        self.norm = norm
        self.criterion = loss_fun #MSE_Loss (L2)
        self.num_channels = num_channels #2 for velocity
        self.font_size = font_size
        self.headers = ("invar", "truth", "prediction", "relative error", "difference target invar")
        self.out_dir = os.path.abspath(os.path.join(os.getcwd(), out_dir))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter

        Returns
        -------
        float
            validation error
        """
        loss = self.criterion(prediction, target) #scalar value MSE_loss
        norm = self.norm

        invar = invar.cpu().numpy()
        target = target.cpu().numpy()
        prediction = prediction.detach().cpu().numpy()

        num_plots = 5

        im = [None] * num_plots
        im = [im] * (self.num_channels + 1)

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(self.num_channels + 1, num_plots, figsize=(15 * num_plots, 15 * (self.num_channels + 1)), sharey=True)

        #plot for each and every channel for the first batch element
        for channel in range(self.num_channels):
            # pick first sample from batch
            invar_plot = invar[0, channel, :, :]
            target_plot = target[0, channel, :, :]
            prediction_plot = prediction[0, channel, :, :]
            im[channel][0] = ax[channel][0].imshow(invar_plot)
            im[channel][1] = ax[channel][1].imshow(target_plot)
            im[channel][2] = ax[channel][2].imshow(prediction_plot)
            im[channel][3] = ax[channel][3].imshow(np.clip((prediction_plot - target_plot) / target_plot, a_min=-1.0, a_max=1.0))
            im[channel][4] = ax[channel][4].imshow(target_plot - invar_plot)

        #magnitude plot in the last row, of the first batch element. invar.shape = [#num_batch_elements,#num_channels,x_res,y_res]
        invar_plot = np.linalg.norm(invar[0, ...], axis=0)  #invar[0, ...].shape = [#num_channels, 512, 256], now axis=0 means it will calculate the norm over the num. channels.
        target_plot = np.linalg.norm(target[0, ...], axis=0)
        prediction_plot = np.linalg.norm(prediction[0, ...], axis=0)
        im[self.num_channels][0] = ax[self.num_channels][0].imshow(invar_plot)
        im[self.num_channels][1] = ax[self.num_channels][1].imshow(target_plot)
        im[self.num_channels][2] = ax[self.num_channels][2].imshow(prediction_plot)
        im[self.num_channels][3] = ax[self.num_channels][3].imshow(np.clip((prediction_plot - target_plot) / target_plot, a_min=-1.0, a_max=1.0))
        im[self.num_channels][4] = ax[self.num_channels][4].imshow(target_plot - invar_plot)

        #final loop to add colorbar to each plot
        for channel in range(self.num_channels + 1):
            for plot in range(num_plots):
                fig.colorbar(im[channel][plot], ax=ax[channel][plot], location="bottom", fraction=0.046, pad=0.04)
                ax[channel][plot].set_title(self.headers[plot])

        log_figure(fig, f"val_step_{step}.png")
        fig.savefig(os.path.join(self.out_dir, f"validation_step_{step}.png"))

        return loss
    
    def compute_only_loss(
        self,    
        target: FloatTensor,
        prediction: FloatTensor,
    ):
        loss = self.criterion(prediction, target)
        return loss
        
class VTKvalidator:
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    out_dir : str, optional
        directory to which plots shall be stored
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        loss_fun,
        data_y_mean,
        data_y_std,
        out_dir: str = "./outputs/validators",
        font_size: float = 28.0,
    ):
        self.criterion = loss_fun
        self.font_size = font_size
        self.data_y_mean = data_y_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.data_y_std = data_y_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.out_dir = os.path.abspath(os.path.join(os.getcwd(), out_dir))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        resolution: int,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter

        Returns
        -------
        float
            validation error
        """
        loss = self.criterion(prediction, target)


        invar = invar.cpu()
        target = target.cpu()
        prediction = prediction.detach().cpu()

        bounce_back_field = invar[0, ...].squeeze()
        true_velocity = target[0, ...]
        predicted_velocity = prediction[0, ...]

        true_velocity = true_velocity * self.data_y_std + self.data_y_mean
        predicted_velocity = predicted_velocity * self.data_y_std + self.data_y_mean


        true_velocity = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, true_velocity)
        predicted_velocity = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, predicted_velocity)

        bounce_back_field = torch.moveaxis(bounce_back_field, 2, 0)
        bounce_back_field = torch.moveaxis(bounce_back_field, 1, 2)
        bounce_back_field = bounce_back_field.flatten().numpy()

        bounce_back_field_array = numpy_support.numpy_to_vtk(bounce_back_field)
        bounce_back_field_array.SetName("bounce_back_field")

        true_velocity = torch.moveaxis(true_velocity, 3, 1)
        true_velocity = torch.moveaxis(true_velocity, 2, 3)
        true_velocity = torch.transpose(true_velocity.flatten(start_dim=1), 0, 1).numpy()

        true_velocity_array = numpy_support.numpy_to_vtk(true_velocity)
        true_velocity_array.SetName("true_velocity")

        predicted_velocity = torch.moveaxis(predicted_velocity, 3, 1)
        predicted_velocity = torch.moveaxis(predicted_velocity, 2, 3)
        predicted_velocity = torch.transpose(predicted_velocity.flatten(start_dim=1), 0, 1).numpy()

        predicted_velocity_array = numpy_support.numpy_to_vtk(predicted_velocity)
        predicted_velocity_array.SetName("predicted_velocity")

        imageData = vtkImageData()
        imageData.SetExtent(
            0,
            resolution,
            0,
            resolution,
            0,
            resolution,
        )
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.GetCellData().AddArray(true_velocity_array)
        imageData.GetCellData().AddArray(predicted_velocity_array)
        imageData.GetCellData().AddArray(bounce_back_field_array)

        vtk_file = f"validation_step_{step}.vti"
        vtk_filename = Path(self.out_dir).joinpath(vtk_file)
        imageData.SetSpacing(1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
        writer = vtkXMLImageDataWriter()
        writer.SetDataModeToBinary()
        writer.SetFileName(str(vtk_filename))
        writer.SetInputData(imageData)
        writer.Write()


        return loss

class DensityVTKvalidator:
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    out_dir : str, optional
        directory to which plots shall be stored
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        loss_fun,
        data_y_mean,
        data_y_std,
        out_dir: str = "./outputs/validators",
        font_size: float = 28.0,
    ):
        self.criterion = loss_fun
        self.font_size = font_size
        self.data_y_mean = data_y_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.data_y_std = data_y_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.out_dir = os.path.abspath(os.path.join(os.getcwd(), out_dir))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        resolution: int,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter

        Returns
        -------
        float
            validation error
        """
        loss = self.criterion(prediction, target)


        invar = invar.cpu()
        target = target.cpu()
        prediction = prediction.detach().cpu()

        bounce_back_field = invar[0, ...].squeeze()
        true_density = target[0, ...]
        predicted_density = prediction[0, ...]

        true_density = true_density * self.data_y_std + self.data_y_mean
        predicted_density = predicted_density * self.data_y_std + self.data_y_mean


        true_density = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, true_density)
        predicted_density = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, predicted_density)

        bounce_back_field = torch.moveaxis(bounce_back_field, 2, 0)
        bounce_back_field = torch.moveaxis(bounce_back_field, 1, 2)
        bounce_back_field = bounce_back_field.flatten().numpy()

        bounce_back_field_array = numpy_support.numpy_to_vtk(bounce_back_field)
        bounce_back_field_array.SetName("bounce_back_field")

        true_density = torch.moveaxis(true_density, 2, 0)
        true_density = torch.moveaxis(true_density, 1, 2)
        true_density = true_density.flatten().numpy()

        true_density_array = numpy_support.numpy_to_vtk(true_density)
        true_density_array.SetName("true_density")

        predicted_density = torch.moveaxis(predicted_density, 2, 0)
        predicted_density = torch.moveaxis(predicted_density, 1, 2)
        predicted_density = predicted_density.flatten().numpy()

        predicted_density_array = numpy_support.numpy_to_vtk(predicted_density)
        predicted_density_array.SetName("predicted_density")

        imageData = vtkImageData()
        imageData.SetExtent(
            0,
            resolution,
            0,
            resolution,
            0,
            resolution,
        )
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.GetCellData().AddArray(true_density_array)
        imageData.GetCellData().AddArray(predicted_density_array)
        imageData.GetCellData().AddArray(bounce_back_field_array)

        vtk_file = f"validation_step_{step}.vti"
        vtk_filename = Path(self.out_dir).joinpath(vtk_file)
        imageData.SetSpacing(1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
        writer = vtkXMLImageDataWriter()
        writer.SetDataModeToBinary()
        writer.SetFileName(str(vtk_filename))
        writer.SetInputData(imageData)
        writer.Write()


        return loss
