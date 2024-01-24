import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import cv2

class HandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.encoded_labels = {
            "up": np.array([1., 0., 0., 0., 0.]),
            "left": np.array([0., 1., 0., 0., 0.]),
            "down": np.array([0., 0., 1., 0., 0.]),
            "right": np.array([0., 0., 0., 1., 0.]),
            "open": np.array([0., 0., 0., 0., 1.])
        }

        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, "images")
        self.labels_dir = os.path.join(self.root_dir, "labels")

        self.sizes = {
            "up": len(os.listdir(os.path.join(self.images_dir, "up"))),
            "left": len(os.listdir(os.path.join(self.images_dir, "left"))),
            "down": len(os.listdir(os.path.join(self.images_dir, "down"))),
            "right": len(os.listdir(os.path.join(self.images_dir, "right"))),
            "open": len(os.listdir(os.path.join(self.images_dir, "open")))
        }

    def __len__(self):
        size = 0
        for gesture in self.encoded_labels.keys():
            size += len(os.listdir(os.path.join(self.images_dir, gesture)))
        return size
    
    def __getitem__(self, idx):
        idx = idx + 1
        gesture = None
        for g in self.encoded_labels.keys():
            if idx < self.sizes[g]:
                gesture = g
                break
            else:
                idx -= self.sizes[g]
        image_path = os.path.join(self.images_dir, gesture, f"{gesture}_{idx}.jpg")
        label_path = os.path.join(self.labels_dir, gesture, f"{gesture}_{idx}.jpg.csv")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        df = pd.read_csv(label_path)
        if len(df) == 0:
            return image, np.array([0., 0., 0., 0., 0.]), 0, 0, 0, 0
        
        df = df.iloc[0] 
        return image, self.encoded_labels[gesture], df["min_x"], df["max_x"], df["min_y"], df["max_y"]




# if __name__ == "__main__":
#     dataset = HandDataset(root_dir="../dataset-creating/data")
#     print(len(dataset))
#     print(dataset[795])
     
