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
import numpy as np
from vtk import vtkImageData
from vtk.util import numpy_support
from vtk import vtkXMLImageDataWriter
from pathlib import Path
from typing import Dict

import copy

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
        num_output_channels: int,
        num_input_channels: int,
        case_name:str,
        mean_info: Dict,
        std_info: Dict,
        out_dir: str = "./outputs/validators",
        font_size: float = 28.0,
        mode:str='train',
        **kwargs
    ):
        self.criterion = loss_fun #MSE_Loss (L2)
        self.case_name = case_name
        self.mean= mean_info
        self.std = std_info
        self.num_output_channels = num_output_channels 
        self.num_input_channels = num_input_channels
        self.font_size = font_size
        self.headers = ("input", "truth", "prediction", "abs error", "relative error")
        self.out_dir = os.path.abspath(os.path.join(os.getcwd(), out_dir))
        
        self.obs_mask = kwargs["obs_mask"]
        self.obs_mask=self.obs_mask.float().cpu().numpy()
        self.obs_mask[self.obs_mask == 0] = float('nan')#repalce 0 with nan inside the obstacle mask
        self.obs_mask = np.squeeze(self.obs_mask)

        self.mode = mode #train or infer
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def undo_normalization(self, invar, target, prediction): #only for plotting
        
        #work with the 0th index of the batch because we are only plotting that.
        #assume density is always included in the first channel

        if self.case_name == "raw_in_raw_out": #case 1
            invar[:, 0, :, :] = invar[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            invar[:, 1, :, :] = invar[:, 1, :, :] * self.std["u"] + self.mean["u"]
            invar[:, 2, :, :] = invar[:, 2, :, :] * self.std["v"] + self.mean["v"]

            #check if Re exists in the key of self.mean
            if "Re" in self.mean.keys():
                invar[:, 3, :, :] = invar[:, 3, :, :] * self.std["Re"] + self.mean["Re"]

            target[:, 0, :, :] = target[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            target[:, 1, :, :] = target[:, 1, :, :] * self.std["u"] + self.mean["u"]
            target[:, 2, :, :] = target[:, 2, :, :] * self.std["v"] + self.mean["v"]

            prediction[:, 0, :, :] = prediction[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            prediction[:, 1, :, :] = prediction[:, 1, :, :] * self.std["u"] + self.mean["u"]
            prediction[:, 2, :, :] = prediction[:, 2, :, :] * self.std["v"] + self.mean["v"]


        elif self.case_name == "3_hist_raw_in_raw_out": #case 2
            invar[:, 0, :, :] = invar[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            invar[:, 1, :, :] = invar[:, 1, :, :] * self.std["u"] + self.mean["u"]
            invar[:, 2, :, :] = invar[:, 2, :, :] * self.std["v"] + self.mean["v"]
            invar[:, 3, :, :] = invar[:, 3, :, :] * self.std["rho"] + self.mean["rho"]
            invar[:, 4, :, :] = invar[:, 4, :, :] * self.std["u"] + self.mean["u"]
            invar[:, 5, :, :] = invar[:, 5, :, :] * self.std["v"] + self.mean["v"]
            invar[:, 6, :, :] = invar[:, 6, :, :] * self.std["rho"] + self.mean["rho"]
            invar[:, 7, :, :] = invar[:, 7, :, :] * self.std["u"] + self.mean["u"]
            invar[:, 8, :, :] = invar[:, 8, :, :] * self.std["v"] + self.mean["v"]

            if "Re" in self.mean.keys():
                invar[:, 9, :, :] = invar[:, 9, :, :] * self.std["Re"] + self.mean["Re"]

            target[:, 0, :, :] = target[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            target[:, 1, :, :] = target[:, 1, :, :] * self.std["u"] + self.mean["u"]
            target[:, 2, :, :] = target[:, 2, :, :] * self.std["v"] + self.mean["v"]

            prediction[:, 0, :, :] = prediction[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            prediction[:, 1, :, :] = prediction[:, 1, :, :] * self.std["u"] + self.mean["u"]
            prediction[:, 2, :, :] = prediction[:, 2, :, :] * self.std["v"] + self.mean["v"]


        elif self.case_name == "raw_and_grad_rho_in_raw_out": #case 3
            invar[:, 0, :, :] = invar[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            invar[:, 1, :, :] = invar[:, 1, :, :] * self.std["u"] + self.mean["u"]
            invar[:, 2, :, :] = invar[:, 2, :, :] * self.std["v"] + self.mean["v"]
            invar[:, 3, :, :] = invar[:, 3, :, :] * self.std["grad_rho_x/_rho"] + self.mean["grad_rho_x/_rho"]
            invar[:, 4, :, :] = invar[:, 4, :, :] * self.std["grad_rho_y/_rho"] + self.mean["grad_rho_y/_rho"]

            #check if Re exists in the key of self.mean
            if "Re" in self.mean.keys():
                invar[:, 5, :, :] = invar[:, 5, :, :] * self.std["Re"] + self.mean["Re"]

            target[:, 0, :, :] = target[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            target[:, 1, :, :] = target[:, 1, :, :] * self.std["u"] + self.mean["u"]
            target[:, 2, :, :] = target[:, 2, :, :] * self.std["v"] + self.mean["v"]

            prediction[:, 0, :, :] = prediction[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            prediction[:, 1, :, :] = prediction[:, 1, :, :] * self.std["u"] + self.mean["u"]
            prediction[:, 2, :, :] = prediction[:, 2, :, :] * self.std["v"] + self.mean["v"]

        elif self.case_name=="acc_in_acc_out": #case 4
            invar[:, 0, :, :] = invar[:, 0, :, :] * self.std["drho_dt"] + self.mean["drho_dt"]
            invar[:, 1, :, :] = invar[:, 1, :, :] * self.std["du_dt"] + self.mean["du_dt"]
            invar[:, 2, :, :] = invar[:, 2, :, :] * self.std["dv_dt"] + self.mean["dv_dt"]

            if "Re" in self.mean.keys():
                invar[:, 3, :, :] = invar[:, 3, :, :] * self.std["Re"] + self.mean["Re"]
            
            #(target and prediction are already renormalized)

        elif self.case_name == "2_hist_acc_in_acc_out": #case 5
            invar[:, 0, :, :] = invar[:, 0, :, :] * self.std["drho_dt"] + self.mean["drho_dt"]
            invar[:, 1, :, :] = invar[:, 1, :, :] * self.std["du_dt"] + self.mean["du_dt"]
            invar[:, 2, :, :] = invar[:, 2, :, :] * self.std["dv_dt"] + self.mean["dv_dt"]
            invar[:, 3, :, :] = invar[:, 3, :, :] * self.std["drho_dt"] + self.mean["drho_dt"]
            invar[:, 4, :, :] = invar[:, 4, :, :] * self.std["du_dt"] + self.mean["du_dt"]
            invar[:, 5, :, :] = invar[:, 5, :, :] * self.std["dv_dt"] + self.mean["dv_dt"]

            if "Re" in self.mean.keys():
                invar[:, 6, :, :] = invar[:, 6, :, :] * self.std["Re"] + self.mean["Re"]
            
            # target[:, 0, :, :] = target[:, 0, :, :] * self.std["drho_dt"] + self.mean["drho_dt"]
            # target[:, 1, :, :] = target[:, 1, :, :] * self.std["du_dt"] + self.mean["du_dt"]
            # target[:, 2, :, :] = target[:, 2, :, :] * self.std["dv_dt"] + self.mean["dv_dt"]

            # prediction[:, 0, :, :] = prediction[:, 0, :, :] * self.std["drho_dt"] + self.mean["drho_dt"]
            # prediction[:, 1, :, :] = prediction[:, 1, :, :] * self.std["du_dt"] + self.mean["du_dt"]
            # prediction[:, 2, :, :] = prediction[:, 2, :, :] * self.std["dv_dt"] + self.mean["dv_dt"]

        elif self.case_name == "acc_and_raw_in_raw_out": #case 6
            invar[:, 0, :, :] = invar[:, 0, :, :] * self.std["drho_dt"] + self.mean["drho_dt"]
            invar[:, 1, :, :] = invar[:, 1, :, :] * self.std["du_dt"] + self.mean["du_dt"]
            invar[:, 2, :, :] = invar[:, 2, :, :] * self.std["dv_dt"] + self.mean["dv_dt"]
            invar[:, 3, :, :] = invar[:, 3, :, :] * self.std["rho"] + self.mean["rho"]
            invar[:, 4, :, :] = invar[:, 4, :, :] * self.std["u"] + self.mean["u"]
            invar[:, 5, :, :] = invar[:, 5, :, :] * self.std["v"] + self.mean["v"]

            #check if Re exists in the key of self.mean
            if "Re" in self.mean.keys():
                invar[:, 6, :, :] = invar[:, 6, :, :] * self.std["Re"] + self.mean["Re"]

            target[:, 0, :, :] = target[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            target[:, 1, :, :] = target[:, 1, :, :] * self.std["u"] + self.mean["u"]
            target[:, 2, :, :] = target[:, 2, :, :] * self.std["v"] + self.mean["v"]

            prediction[:, 0, :, :] = prediction[:, 0, :, :] * self.std["rho"] + self.mean["rho"]
            prediction[:, 1, :, :] = prediction[:, 1, :, :] * self.std["u"] + self.mean["u"]
            prediction[:, 2, :, :] = prediction[:, 2, :, :] * self.std["v"] + self.mean["v"]

        else:
            #raise a not implenmented error
            raise NotImplementedError("This case is not implemented/found")

        return invar, target, prediction
        

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
        # loss = self.criterion(prediction, target) #scalar value MSE_loss
        # norm = self.norm

        #undo normalization
        invar, target, prediction = self.undo_normalization(copy.deepcopy(invar), copy.deepcopy(target), copy.deepcopy(prediction.detach()))

        invar = invar.cpu().numpy()
        target = target.cpu().numpy()
        prediction = prediction.cpu().numpy()

        num_plots = 5
        
        colormap = 'viridis'

        im = [None] * num_plots
        im = [im] * (self.num_input_channels)

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(self.num_input_channels, num_plots, figsize=(15 * num_plots, 15 * (self.num_input_channels + 1)), sharey=True)

        #plot for each and every channel for the first batch element
        for input_channel in range(self.num_input_channels):
            invar_plot = invar[0, input_channel, :, :]
            im[input_channel][0] = ax[input_channel][0].imshow(invar_plot*self.obs_mask, cmap=colormap)
            fig.colorbar(im[input_channel][0], ax=ax[input_channel][0], location="bottom", fraction=0.046, pad=0.04)
            ax[input_channel][0].set_title(self.headers[0])

        for channel in range(self.num_output_channels): 
            target_plot = target[0, channel, :, :]
            prediction_plot = prediction[0, channel, :, :]
            
            im[channel][1] = ax[channel][1].imshow(target_plot*self.obs_mask, cmap=colormap)
            fig.colorbar(im[channel][1], ax=ax[channel][1], location="bottom", fraction=0.046, pad=0.04)
            ax[channel][1].set_title(self.headers[1])

            im[channel][2] = ax[channel][2].imshow(prediction_plot*self.obs_mask, cmap=colormap)
            fig.colorbar(im[channel][2], ax=ax[channel][2], location="bottom", fraction=0.046, pad=0.04)
            ax[channel][2].set_title(self.headers[2])
            
            im[channel][3] = ax[channel][3].imshow(np.abs(target_plot - prediction_plot)*self.obs_mask, cmap=colormap)
            fig.colorbar(im[channel][3], ax=ax[channel][3], location="bottom", fraction=0.046, pad=0.04)
            ax[channel][3].set_title(self.headers[3])

            im[channel][4] = ax[channel][4].imshow(np.clip(((prediction_plot - target_plot) / target_plot)*self.obs_mask, a_min=-1.0, a_max=1.0), cmap=colormap)
            fig.colorbar(im[channel][4], ax=ax[channel][4], location="bottom", fraction=0.046, pad=0.04)
            ax[channel][4].set_title(self.headers[4]) 

        if self.mode == "infer":
            fig.savefig(os.path.join(self.out_dir, f"rollout_step_{step}.png"))
        else:
            fig.savefig(os.path.join(self.out_dir, f"validation_step_{step}.png"))
    
    
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
