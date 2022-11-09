import cv2
import mediapipe as mp

import numpy as np
import torch
from rock_paper_scissor.network.gesture_prediction.model import Model
from rock_paper_scissor.network.gesture_recognition.model import Hand_MLP

from rock_paper_scissor.utils.keypoints_utils import landmark_to_array, fix_orientation, pre_process_landmark

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Game():

    _HAND_MISSING_THRESHOLD = 2
    _VIDEO_FRAME_PATH = "assets/gesture_vframes"

    def __init__(self, prediction_model_pth, recognition_model_pth):
        # initialize mediapipe
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3)
        self.prediction_model_pth = prediction_model_pth
        self.recognition_model_pth = recognition_model_pth

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # load model
        self.prediction_model = Model().to(self.device)
        self.prediction_model.load_state_dict(torch.load(prediction_model_pth))

        self.recognition_model = Hand_MLP(63, 30, 3).to(self.device)
        self.recognition_model.load_state_dict(
            torch.load(recognition_model_pth))

    def play(self):
        self.game(self.hands, 50, 15, 4)

    # get frame of gesture from folder
    def _get_frame(self, path, index, gesture="pending"):
        file_name = path + "/" + gesture + "_%03d.jpg" % index
        # check if file exists
        if os.path.isfile(file_name):
            return cv2.imread(file_name)

    # def predic static gesture
    def _recognize_gesture(self, keypoints):
        # input 1*42
        # keypoints 21*3
        data = np.array(keypoints)
        data = pre_process_landmark(data)
        data = np.array(data)

        data = torch.from_numpy(data).float()
        data = data.to(self.device)
        data = data.unsqueeze(0)
        output = self.recognition_model(data)
        _, predicted = torch.max(output.data, 1)
        return predicted

        # predict gesture
    def predict_gesture(self, keypoints):
        data = np.array(keypoints)
        data = torch.from_numpy(data).float()
        data = data.to(self.device)
        data = data.unsqueeze(0)
        # convert channel to first dimension
        data = data.permute(0, 3, 1, 2)
        output = self.prediction_model(data)
        _, predicted = torch.max(output.data, 1)
        return predicted

    # extract keypoints

    def extract_keypoints(self, img, hands):
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if mp_results.multi_hand_world_landmarks:
            result = landmark_to_array(
                mp_results.multi_hand_world_landmarks[0])
            result = fix_orientation(result)
            return result
        else:
            return None

    # draw frame
    def draw_frame(self, image, gesture_img, computer_color, player_color):
        # resize image to 720*540
        image = cv2.resize(image, (720, 540))
        # flip image
        image = cv2.flip(image, 1)
        # add a padding on the right side of the image
        image = cv2.copyMakeBorder(
            image, 0, 0, 720, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if gesture_img is not None:
            # resize gesture image to 720*540
            gesture_img = cv2.resize(gesture_img, (720, 540))
            # combine image and gesture image
            image[-540:, :720] = gesture_img
        # draw a border on the right side of the image
        image = cv2.rectangle(image, (723, 3), (1437, 537), player_color, 6)
        # draw a border on the left side of the image
        image = cv2.rectangle(image, (3, 3), (717, 537), computer_color, 6)
        return image

    # capture gesture

    def game(self, hands, pending_time=100, recording_time=15, animation_time=7, static_gesture_wait_time=5):
        # initialize camera
        cap = cv2.VideoCapture(0)
        # set frame to 60 fps
        cap.set(cv2.CAP_PROP_FPS, 60)

        # initialize keypoints data
        keypoints_series = []

        current_frame = 0

        video_frame = 0

        frame = None

        gesture = "pending"

        final_gesture = "pending"

        border_computer_color = (0, 0, 0)
        border_player_color = (0, 0, 0)

        hand_missing_frames = -1
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            keypoints = self.extract_keypoints(image, hands)

            if keypoints is not None:
                hand_missing_frames = 0
                if current_frame < pending_time:
                    video_frame += 1
                    if current_frame >= pending_time - recording_time:
                        keypoints_series.append(keypoints)
                elif current_frame == pending_time:
                    prediction = self.predict_gesture(keypoints_series)
                    video_frame = 0
                    if prediction == 0:
                        gesture = "scissor"
                    elif prediction == 1:
                        gesture = "paper"
                    elif prediction == 2:
                        gesture = "rock"
                elif current_frame < pending_time + animation_time:
                    video_frame += 1
                else:
                    video_frame = animation_time
                if current_frame == pending_time + animation_time + static_gesture_wait_time:
                    static_prediction = self._recognize_gesture(keypoints)
                    if static_prediction == 0:
                        final_gesture = "rock"
                    elif static_prediction == 1:
                        final_gesture = "scissor"
                    elif static_prediction == 2:
                        final_gesture = "paper"
                    # determin if the player wins
                    if gesture == final_gesture:
                        print("DRAW")
                        border_computer_color = (255, 0, 0)
                        border_player_color = (255, 0, 0)
                    elif (gesture == "scissor" and final_gesture == "paper") or (gesture == "paper" and final_gesture == "rock") or (gesture == "rock" and final_gesture == "scissor"):
                        print("COMPUTER WIN")
                        border_computer_color = (0, 255, 0)
                    else:
                        print("PLAYER WIN")
                        border_player_color = (0, 255, 0)

                # get frame of gesture
                frame = self._get_frame(
                    Game._VIDEO_FRAME_PATH, video_frame, gesture)
                # draw frame
                image = self.draw_frame(
                    image, frame, border_computer_color, border_player_color)
                # draw text
                cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                current_frame += 1
                # print(current_frame, video_frame)
            else:
                hand_missing_frames += 1
                if hand_missing_frames > Game._HAND_MISSING_THRESHOLD:
                    current_frame = 0
                    video_frame = 0
                    keypoints_series = []
                    gesture = "pending"
                    final_gesture = "pending"
                    border_computer_color = (0, 0, 0)
                    border_player_color = (0, 0, 0)
                image = self.draw_frame(
                    image, frame, border_computer_color, border_player_color)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
