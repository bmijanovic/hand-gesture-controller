import cv2
import mediapipe as mp
import os
import pandas as pd

mp_hands_sol = mp.solutions.hands
mp_hands = mp_hands_sol.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils


poses = ['open', 'left', 'right', 'up', 'down']
# poses = ['up']



def capture_handgesture(directory):
    current_gesture = directory.split('/')[3]
    for image_file in os.listdir(directory):
        
        image = cv2.imread(f'{directory}/{image_file}')
        
        results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                coords = list(map(lambda data_point: [data_point.x, data_point.y], hand_landmarks.landmark))

                min_x = min(coords, key=lambda x: x[0])[0]
                max_x = max(coords, key=lambda x: x[0])[0]
                min_y = min(coords, key=lambda x: x[1])[1]
                max_y = max(coords, key=lambda x: x[1])[1]

                margin = 0.03

                min_x = max(0, min_x - margin)
                max_x = min(1, max_x + margin)
                min_y = max(0, min_y - margin)
                max_y = min(1, max_y + margin)

                datapoint = {
                    'gesture': current_gesture,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y
                }
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands_sol.HAND_CONNECTIONS)
                cv2.rectangle(image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])), (int(max_x * image.shape[1]), int(max_y * image.shape[0])), (0, 255, 0), 2)
                cv2.imshow("image", image)
                cv2.waitKey(0)


                df = pd.DataFrame([datapoint])

                df.to_csv("data/labels/" + current_gesture + "/" + image_file + ".csv", index=False)
        else:
            os.remove("data/images/" + current_gesture + "/" + image_file)



for pose in poses:
    capture_handgesture(f'./data/images/{pose}')


