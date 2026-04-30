import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class SoliDataset(Dataset):
    def __init__(self, folder_path, seq_len=40, channel=0, allowed_sessions=None):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.channel = channel

        self.files = []
        for f in os.listdir(folder_path):
            if f.endswith(".h5"):
                session = int(f.split('_')[1])  # extract session ID

                # filter by session
                if allowed_sessions is not None and session not in allowed_sessions:
                    continue

                file_path = os.path.join(folder_path, f)
                with h5py.File(file_path, 'r') as hf:
                    label = int(hf['label'][()][0][0])

                    if label != 11:  # skip background
                        self.files.append(f)

        print("Sample files:", self.files[:10])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.folder_path, file_name)

        with h5py.File(file_path, 'r') as f:
            # load selected channel
            data = f[f'ch{self.channel}'][()]   # shape: (T, 1024)
            label = f['label'][()]              # shape: (T, 1)

        # reshape to (T, 32, 32)
        data = data.reshape(-1, 32, 32)

        # normalize (simple)
        data = data / (np.max(data) + 1e-6)

        # fix sequence length
        T = data.shape[0]

        if T > self.seq_len:
            data = data[:self.seq_len]
        else:
            pad_len = self.seq_len - T
            pad = np.zeros((pad_len, 32, 32))
            data = np.concatenate((data, pad), axis=0)

        # add channel dimension → (T, 1, 32, 32)
        data = np.expand_dims(data, axis=1)

        # get single label
        label = int(label[0][0])

        # convert to torch tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return data, label