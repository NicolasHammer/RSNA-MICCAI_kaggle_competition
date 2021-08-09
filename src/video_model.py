import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
from efficientnet_pytorch import EfficientNet

import numpy as np
from PIL import Image


class FeatureExtractor(nn.Module):
    def __init__(self, num_channels: int):
        super(FeatureExtractor, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=3,
                      kernel_size=(3, 3), padding="same"),
            nn.ReLU()
        )
        self.base_model = EfficientNet.from_pretrained(
            'efficientnet-b0', include_top=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_layer(data)
        extracted_out = self.base_model(conv_out)

        return extracted_out.view(extracted_out.shape[0], extracted_out.shape[1])


class VideoMRIModel(nn.Module):
    def __init__(self, num_channels: int, lstm_units: int):
        super(VideoMRIModel, self).__init__()

        self.feature_extractor = FeatureExtractor(num_channels)
        self.lstm = nn.LSTM(1280, lstm_units, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(lstm_units, 1),
            nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, C, H, W = data.size()

        data_reshaped = data.view(batch_size*seq_length, C, H, W)
        features_extracted = self.feature_extractor(data_reshaped)
        features_reshaped = features_extracted.view(batch_size, seq_length, -1)

        _, (hidden_states, _) = self.lstm(features_reshaped)
        classification = self.linear(torch.squeeze(hidden_states, 0))

        return classification.view(batch_size)


class VideoMRIDataset(Dataset):
    mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __init__(self, file_data: np.ndarray, class_data: np.ndarray, seq_length: int, directory: str, image_size: Tuple[int, int]):
        assert(seq_length > 0)

        self.file_data = file_data
        self.class_data = class_data
        self.seq_length = seq_length
        self.directory = directory
        self.image_size = image_size

        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.file_data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Given an index of a patient, get that patient's sequence of images"""
        patient_dir = os.path.join(
            self.directory, self.file_data[idx])

        final_tensor = []
        for mri_type in VideoMRIDataset.mri_types:
            mri_dir = os.path.join(patient_dir, mri_type)

            if os.path.isdir(mri_dir):
                image_names = os.listdir(mri_dir)

                start_idx = np.random.randint(
                    0, len(image_names) - self.seq_length + 1)
                images = [self.transforms(Image.open(os.path.join(mri_dir, image_names[i])))
                        for i in range(start_idx, start_idx + self.seq_length)]

                final_tensor.append(torch.stack(images))
            else:
                final_tensor.append(torch.zeros(10, 1, *self.image_size, dtype=torch.float32))

        return torch.cat(final_tensor, 1), self.class_data[idx]
