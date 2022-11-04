from sre_constants import SRE_FLAG_DEBUG
import time
import cv2
import mediapipe as mp

import numpy as np

import sys
sys.path.append("..")

from rock_paper_scissor.utils.ui import text_animation
from rock_paper_scissor.utils.keypoints_utils import landmark_to_array, fix_orientation

cap = cv2.VideoCapture(0)

dataset = []

def capture_gesture(data_length, hands, text_interval = 15):
    """
    Capture gesture data from webcam
    
    Parameters:
    data_length (int): number of frames to capture
    hands (mediapipe.solutions.hands): mediapipe hands object
    text_interval (int): interval for text animation
    """
    captured_frame_count = 0    # number of frames captured
    data = []                   # data to return
    has_hand_frame = 0          # number of frames with hand detected

    # Capture frame-by-frame
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 1)

        # only show text animation and capture data when hand is detected
        if results.multi_hand_world_landmarks:
            has_hand_frame += 1
            if has_hand_frame < text_interval:
                # display text "rock" on the center of screen
                image = text_animation("rock", has_hand_frame, image, text_interval)
            elif has_hand_frame < text_interval * 2:
                # display text "paper" on the screen
                image = text_animation("paper", has_hand_frame, image, text_interval)
            elif has_hand_frame < text_interval * 3:
                # display text "scissor" on the screen
                image = text_animation("scissor", has_hand_frame, image, text_interval)
            elif has_hand_frame < text_interval * 4:
                # display text "shoot" on the screen
                image = text_animation("shoot", has_hand_frame, image, text_interval, color=(0, 255, 0))
            
            # exit loop when last text animation is displayed
            if has_hand_frame == text_interval * 4:
                break

            if has_hand_frame >= text_interval*3 - text_interval/3:
                captured_frame_count += 1
                # uncomment the following line to display a red border when capturing data
                # image = cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 5)
                
                if captured_frame_count > data_length:
                    continue
                landmark_array = landmark_to_array(results.multi_hand_world_landmarks[0])
                landmark_array = fix_orientation(landmark_array)
                data.append(landmark_array)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    dataset.append(data)


def capture_data(file_name, data_length, text_interval = 15):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        for i in range(data_length):
            start_time = time.time()
            capture_gesture(20, hands, text_interval=20)
            print(i)
        cap.release()
        print(dataset)
        # save data to file
        np.save(file_name, dataset)