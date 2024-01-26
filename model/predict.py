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
import mediapipe as mp
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
LOAD_MODEL_FILE = "../nns/overfit_new2_32b11e.pth.tar"
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


mp_hands_sol = mp.solutions.hands
mp_hands = mp_hands_sol.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

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
        # cv2.imshow("frame", frame)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame2 = frame.copy()
        
        # convert to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToPILImage()(frame)
        frame = transform(frame)
        frame = frame.unsqueeze(0)

        # draw hand landmarks
        results = mp_hands.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame2, hand_landmarks, mp_hands_sol.HAND_CONNECTIONS)
                # draw bounding box
                x1 = int(hand_landmarks.landmark[0].x * 640)
                y1 = int(hand_landmarks.landmark[0].y * 480)
                x2 = int(hand_landmarks.landmark[0].x * 640)
                y2 = int(hand_landmarks.landmark[0].y * 480)
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    

                
    

        # get prediction
        model.eval()
        with torch.no_grad():
            predictions = model(frame.to(DEVICE))
            # predictions = predictions[0].squeeze(0)
            bboxes = cellboxes_to_boxes(predictions)
            bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            if bboxes != []: 
                print(bboxes)
                _, _, x, y, w, h = bboxes[0]
                x1 = int((x-w/2) * 640)
                y1 = int((y-h/2) * 480)
                x2 = int((x+w/2) * 640)
                y2 = int((y+h/2) * 480)
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("frame", frame2)
            else:
                cv2.imshow("frame", frame2)
                


        if cv2.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()