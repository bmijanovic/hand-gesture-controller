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
LOAD_MODEL_FILE = "overfit3.pth.tar"
ROOT_DIR = "../dataset-creating/data/"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img
    
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


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

    # open camera and capture image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        
        # convert to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToPILImage()(frame)
        frame = transform(frame)
        frame = frame.unsqueeze(0)

        # get prediction
        model.eval()
        with torch.no_grad():
            predictions = model(frame.to(DEVICE))
            # predictions = predictions[0].squeeze(0)
            bboxes = cellboxes_to_boxes(predictions)
            bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            if bboxes != []: 
                print(bboxes)

            # draw boxes and show image
            # plot_image(frame, boxes)

        if cv2.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()