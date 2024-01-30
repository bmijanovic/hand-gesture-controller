import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
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
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "../nns/pokusaj2.pth.tar"
TRAIN_ROOT_DIR = "../dataset-creating/data/train/"
TEST_ROOT_DIR = "../dataset-creating/data/test/"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img
    
transform = Compose([
    transforms.Resize((448, 448)), 
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
    transforms.ToTensor(), 
])

# transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def train(train_dataloader, model, optimizer, loss_fn):
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # for x, y in train_dataloader:
        #     x = x.to(DEVICE)
        #     for idx in range(5):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #         plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        pred_boxes, target_boxes = get_bboxes(
            train_dataloader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_dataloader, model, optimizer, loss_fn)



        if epoch % 5 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)


    
def test_fn(test_loader, model):
    tacno, ukupno = 0, 0
    # provide image, get label
    for x, y in test_loader:
        ukupno += 1
        x = x.to(DEVICE)
        out = model(x)
        bboxes = cellboxes_to_boxes(out)
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        if bboxes != []:
            # print(y.tolist()[0])
            
            for row in y.tolist()[0]:
                for col in row:
                    if col[5] == 1:
                        y_train = torch.argmax(torch.tensor(col[:5]))
                        print(y_train)
                        if y_train.item() == bboxes[0][0]:
                            tacno += 1
                            break
            # print(bboxes)
            # plot_image(x[0].permute(1,2,0).to("cpu"), bboxes)

        print(f"Tacno: {tacno}, ukupno: {ukupno}, tacnost: {tacno/ukupno}")






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

    # train_dataset = HandDataset(root_dir=TRAIN_ROOT_DIR, transform=transform, S=7, B=1, C=5)

    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     shuffle=True,
    #     drop_last=False,
    # )
    # train(train_dataloader, model, optimizer, loss_fn)

    test_dataset = HandDataset(root_dir=TEST_ROOT_DIR, transform=transform, S=7, B=1, C=5)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_fn(test_dataloader, model)


    # images = []
    # # load all .jpg files from current directory
    # for filename in os.listdir("."):
    #     if filename.endswith(".jpg"):
    #         images.append(filename)

    # for image in images:
    #     img = Image.open(image)
    #     img = transform(img)
    #     img = img.unsqueeze(0)
    #     model.eval()

    #     with torch.no_grad():
    #         predictions = model(img.to(DEVICE))
    #         bboxes = cellboxes_to_boxes(predictions)
    #         bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #         print(bboxes)
    

    # # plot both images at once
    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # for idx, image in enumerate(images):
    #     img = Image.open(image)
    #     ax[idx].imshow(img)
    
    # plt.show()
    
    
    




if __name__ == "__main__":
    main()