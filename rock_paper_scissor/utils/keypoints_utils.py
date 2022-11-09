import itertools
import cv2
import mediapipe as mp

import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def landmark_to_array(landmarks):
  # convert landmarks to numpy array
  landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark])
  return landmarks

def fix_orientation(landmarks):
  # make the first landmark the start point
  landmarks -= landmarks[9]
  # get the angle of the first landmark to the z axis
  angle = np.arctan2(landmarks[0, 1], landmarks[0, 0])
  # rotate the landmarks to the z axis
  rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]])
  landmarks = np.dot(landmarks, rotation_matrix)
  return landmarks


def pre_process_landmark(landmark_list_xyz):
    temp_landmark_list_xyz = []
    # Convert to relative coordinates
    origin_x = None
    origin_y = None
    origin_z = None

    for idx, landmark_point in enumerate(landmark_list_xyz):
        if idx == 0:

            origin_x = landmark_point[0]

            origin_y = landmark_point[1]

            origin_z = landmark_point[2]

           

        xPos = landmark_list_xyz[idx][0] - origin_x

        yPos = landmark_list_xyz[idx][1] - origin_y

        zPos = landmark_list_xyz[idx][2] - origin_z


        temp_landmark_list_xyz.append([xPos,yPos,zPos])

       

    # Convert to a one-dimensional list

    temp_landmark_list_xyz = list(

        itertools.chain.from_iterable(temp_landmark_list_xyz))



    # Normalization

    max_value = max(list(map(abs, temp_landmark_list_xyz)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list_xyz = list(map(normalize_, temp_landmark_list_xyz))
    return temp_landmark_list_xyz

if __name__ == '__main__':

  # For webcam input:
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      img_w = image.shape[1]
      img_h = image.shape[0]
      if results.multi_hand_world_landmarks:
        landmark_array = landmark_to_array(results.multi_hand_world_landmarks[0])
        landmark_array = fix_orientation(landmark_array)
        print(landmark_array[0])
        # display the landmarks using cv2
        for i, landmark in enumerate(landmark_array):
          if i == 9:
            color = (0, 255, 0)
          elif i == 0:
            color = (0, 0, 255)
          else:
            color = (255, 0, 0)
          cv2.circle(image, (int(landmark[0]*img_w+ img_w/2), int(landmark[1]*img_h+ img_h/2)), 3, color, -1)
          
          # cv2.putText(image, str(i), (int(landmark[0]), int(landmark[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()