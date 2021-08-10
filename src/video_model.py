import os
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
from efficientnet_pytorch import EfficientNet

import numpy as np
from PIL import Image


class FeatureExtractor(nn.Module):
    """A convolutional neural network which extracts the features of an image."""

    def __init__(self, num_channels: int):
        """
        Paramters
        ---------
        num_channels (int) - the number of channels in an image
        """
        super(FeatureExtractor, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=3,
                      kernel_size=(3, 3), padding="same"),
            nn.ReLU()
        )
        self.base_model = EfficientNet.from_pretrained(
            'efficientnet-b0', include_top=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data (torch.Tensor) - the images we want features extracted from of 
            shape (batch_sise, num_channels, height, width)

        Output
        ------
        (torch.Tensor) - tensor of extracted images of shape (batch_size, 1280)
        """
        conv_out = self.conv_layer(data)
        extracted_out = self.base_model(conv_out)

        return extracted_out.view(extracted_out.shape[0], extracted_out.shape[1])


class VideoMRIModel(nn.Module):
    """A CNN+RNN network that processes a single MRI scan"""
    def __init__(self, lstm_units: int):
        """
        Paramters
        ---------
        lstm_units (int) - the number of hidden units in the LSTM
        """
        super(VideoMRIModel, self).__init__()

        self.feature_extractor = FeatureExtractor(1)
        self.lstm = nn.LSTM(1280, lstm_units, batch_first=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters 
        ----------
        data (torch.Tensor) - a the mri scan we want processed of shape
            (batch_size, seq_length, 1, height, width).  Note that MRI scans
            only have a channel of 1
        
        Output
        ------
        (torch.Tensor) - the final hidden layer of shape (lstm_units)
        """
        batch_size, seq_length, _, _, _ = data.size()
        assert batch_size == 1, f"batch_size must be 1 but got {batch_size}"

        data_reshaped = torch.squeeze(data, 0)
        features_extracted = self.feature_extractor(data_reshaped)
        features_reshaped = features_extracted.view(1, seq_length, -1)

        _, (hidden_states, _) = self.lstm(features_reshaped)

        return torch.squeeze(hidden_states)


class VideoMRIEnsemble(nn.Module):
    """An ensemble of different VideoMRIModels combined by a linear layer"""
    def __init__(self, scans: List[str], lstm_units: int):
        """
        Parameters
        ----------
        scans (List[str]) - a list of the types of scans under consideration
        lstm_units (int) - the number of hidden nodes used in the LSTM
        """
        super(VideoMRIEnsemble, self).__init__()

        self.video_models = nn.ModuleDict(
            {scan: VideoMRIModel(lstm_units) for scan in scans})

        self.linear = nn.Sequential(
            nn.Linear(len(scans)*lstm_units, 1),
            nn.Sigmoid()
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        data (Dict[str, torch.Tensor]) - a mapping from MRI type to video tensor of that type

        Output
        ------
        Classifcation as to whether the MGTM promoter has been methylized
        """
        MRI_output = torch.cat([
            self.video_models[scan](video) for scan, video in data.items()
        ])
        classification = self.linear(MRI_output)
        return classification


class VideoMRIDataset(Dataset):
    """A dataset of folder names for patients"""
    mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __init__(self, file_data: np.ndarray, class_data: np.ndarray,
                 directory: str, image_size: Tuple[int, int]):
        """
        Parameters
        ----------
        file_data (np.ndarray) - array of folder names
        class_data (np.ndarray) - array of classes that match up with patients
            enumerated in file_data
        directory (str) - the directory of the trianing folder
        image_size (Tuple[int, int]) - the dimensions of the image
        """
        self.file_data = file_data
        self.class_data = class_data
        self.directory = directory
        self.image_size = image_size

        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.file_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Given an index of a patient, get that patient's MRI videos
        
        Parameters
        ----------
        idx (int) - the index of the item we are trying to obtain

        Output
        ------
        (Dict[str, torch.Tensor]) - a mapping from mri_type to video tensor
        (int) - the cassification of the patient
        """
        patient_dir = os.path.join(
            self.directory, self.file_data[idx])

        tensor_dict = {}
        for mri_type in VideoMRIDataset.mri_types:
            mri_dir = os.path.join(patient_dir, mri_type)

            if os.path.isdir(mri_dir):
                images = [self.transforms(Image.open(os.path.join(mri_dir, image_name)))
                          for image_name in os.listdir(mri_dir)]

                tensor_dict[mri_type] = torch.stack(images)
            else:
                tensor_dict[mri_type] = torch.zeros(
                    1, 1, *self.image_size, dtype=torch.float32)

        return tensor_dict, self.class_data[idx]
