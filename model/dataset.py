import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import cv2
from PIL import Image

# S = 7
# [c1, c2, c3, c4, c5, pc, x, y, w, h]
# matrix = np.zeros((S, S, 10))
class HandDataset(Dataset):
    def __init__(self, root_dir, S=7, B=1, C=5, transform=None):
        self.transform = transform
        self.encoded_labels = {
            "up": np.array([1., 0., 0., 0., 0.]),
            "left": np.array([0., 1., 0., 0., 0.]),
            "down": np.array([0., 0., 1., 0., 0.]),
            "right": np.array([0., 0., 0., 1., 0.]),
            "open": np.array([0., 0., 0., 0., 1.])
        }

        self.label_index = {
            "up": 0,
            "left": 1,
            "down": 2,
            "right": 3,
            "open": 4
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

        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        size = 0
        for gesture in self.encoded_labels.keys():
            size += len(os.listdir(os.path.join(self.images_dir, gesture)))
        return size
    
    def __getitem__(self, idx):
        gesture = None
        for g in self.encoded_labels.keys():
            if idx < self.sizes[g]:
                gesture = g
                break
            else:
                idx -= self.sizes[g]
        image_path = os.path.join(self.images_dir, gesture, f"{gesture}_{idx}.jpg")
        label_path = os.path.join(self.labels_dir, gesture, f"{gesture}_{idx}.jpg.csv")

        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)


        
        
        df = pd.read_csv(label_path)

        if len(df) == 0:
            return image, np.zeros((self.S, self.S, self.C + 5 * self.B))

        df = df.iloc[0] 
        
        class_label = self.label_index[gesture]
        x = (df["min_x"] + df["max_x"]) / 2
        y = (df["min_y"] + df["max_y"]) / 2
        w = df["max_x"] - df["min_x"]
        h = df["max_y"] - df["min_y"]

        label_matrix = np.zeros((self.S, self.S, self.C + 5 * self.B))
        i, j = int(self.S * y), int(self.S * x)
        x_cell, y_cell = self.S * x - j, self.S * y - i
        width_cell, height_cell = w * self.S, h * self.S
        if label_matrix[i, j, self.C] == 0:
            label_matrix[i, j, self.C] = 1
            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            label_matrix[i, j, self.C + 1: self.C + 5] = box_coordinates
            label_matrix[i, j, class_label] = 1

        return image, label_matrix




# if __name__ == "__main__":
#     dataset = HandDataset(root_dir="../dataset-creating/data")
#     print(len(dataset))
#     print(dataset[795])
     
