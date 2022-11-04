import numpy as np
import torch

class RockPaperScissorDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]

def read_data():
    paper_data_path = "rock_paper_scissor/network/gesture_prediction/data/paper.npy"
    rock_data_path = "rock_paper_scissor/network/gesture_prediction/data/rock.npy"
    scissor_data_path = "rock_paper_scissor/network/gesture_prediction/data/scissor.npy"

    paper_data = np.load(paper_data_path).astype(np.float32)
    paper_data = paper_data[:100, :15, :]
    rock_data = np.load(rock_data_path).astype(np.float32)
    rock_data = rock_data[:100, :15, :]
    scissor_data = np.load(scissor_data_path).astype(np.float32)
    scissor_data = scissor_data[:100, :15, :]
    # concatenate data
    data = np.concatenate((paper_data, rock_data, scissor_data), axis=0)
    labels = np.concatenate((np.repeat(0, len(paper_data)), np.repeat(1, len(rock_data)), np.repeat(2, len(scissor_data))))
    return data, labels