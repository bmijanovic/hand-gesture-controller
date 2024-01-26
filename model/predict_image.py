import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import HandDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from PIL import Image
import os

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit_new_32b11e.pth.tar"
ROOT_DIR = "../dataset-creating/data/"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img
    
# transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    
transform = Compose([
    transforms.Resize((448, 448)), 
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False), 
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), 
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, fill=0), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def main():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    model = YOLOv1(split_size=7, num_boxes=1, num_classes=5).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    for image in os.listdir("."):
        if image.endswith(".jpg"):
            print(image)
            image_path = image

            # image_path = "../dataset-creating/data/images/up/up_11.jpg"
            # image_path = "test2.jpg"

            # image = Image.open(image_path)
            # image = transform(image)
            # image = image.unsqueeze(0)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transforms.ToPILImage()(image)
            image = transform(image)
            image = image.unsqueeze(0)

            model.eval()

            with torch.no_grad():
                image = image.to(DEVICE)
                out = model(image)
                bboxes = cellboxes_to_boxes(out)
                bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                print(bboxes)
                # plot_image(x[0].permute(1,2,0).to("cpu"), bboxes)





    





if __name__ == "__main__":
    main()