from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, window_size, folder_dir):
        self.data_list = []
        self.window_size = window_size
        self.folder_dir = folder_dir

        for filename in os.listdir(folder_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder_dir, filename)
                data = pd.read_csv(filepath, header=None).to_numpy()

                # [N, 100] → features: first 99 elements, labels: last element
                features = torch.FloatTensor(data[:, 0:99])   # joints + angles
                labels = torch.FloatTensor(data[:, 99])       # label (class_num)

                labels = labels.reshape(-1, 1)  # reshape to [N, 1]

                for i in range(len(features) - window_size):
                    features_subset = features[i:i + window_size]
                    label_subset = labels[i]  # use label from the beginning of sequence
                    self.data_list.append([features_subset, label_subset])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return x, y
