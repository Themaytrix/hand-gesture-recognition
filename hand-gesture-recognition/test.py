import copy
import csv
import itertools
from concurrent.futures import process

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from keypoint_classifier import KeyPointClassifier


def main():
    # initializing mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # reading label
    with open("hand-gesture-recognition/label.csv", encoding="utf-8-sig") as f:
        keypoint_labels = csv.reader(f)
        keypoint_labels = [row[0] for row in keypoint_labels]

    keypoint_classifier = KeyPointClassifier()

    # activating webcam
    cap = cv2.VideoCapture(0)
    # using handdetection model
    mode = 0
    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            key = cv2.waitKey(10)
            number, mode = selectKey(key, mode)

            # mapping hand annotations
            # frame.flags.writeable = False
            # converting frame colors to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            # passing frame into the handle model and captured in results variable
            results = hands.process(frame)
            display_info(frame, mode)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # drawing annotations
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    # finding positions
                    landmark_list = find_possitons(frame, hand_landmarks)
                    # display_info(frame, mode)
                    # drawing bounds
                    rect_coordinates = calculating_bound(frame, hand_landmarks)
                    cv2.rectangle(
                        frame,
                        (rect_coordinates[0], rect_coordinates[1]),
                        (rect_coordinates[2], rect_coordinates[3]),
                        (255, 123, 50),
                        1,
                    )
                    # preprocessing landmark_points
                    processed_landmark_list = pre_processing_lms(landmark_list)
                    # logging points into csv file
                    loggingcsv(number, mode, processed_landmark_list)
                    # collecting hand sign index

                    # # corr = correlationnn(pre_processing_lms, 0.85)
                    # print(type(processed_landmark_list))
                    # processed_landmark_list = pd.DataFrame(processed_landmark_list)
                    # corr = correlationnn(processed_landmark_list, 0.85)
                    # print(len(set(corr)))
                    # inference_lms = processed_landmark_list.drop(corr, axis=1)

                    hand_sign_id = keypoint_classifier(processed_landmark_list)
                    # drawing info text
                    if mode == 0:
                        info_text(
                            frame,
                            handedness,
                            keypoint_labels[hand_sign_id],
                            rect_coordinates,
                        )

            cv2.imshow("MediaPipe Hands", frame)

            if cv2.waitKey(10) == ord("q"):
                break

    cap.release()


# collecting landmarks into an array
def find_possitons(frame, landmarks):
    landmark_points = []
    for _, lms in enumerate(landmarks.landmark):
        # getting the dimensions of the capture
        h, w, c = frame.shape  # returns height, width, channel number
        # changing the values of landmarks into pixels
        cap_x, cap_y = min(int(lms.x * w), w - 1), min(int(lms.y * h), h - 1)
        landmark_points.append([cap_x, cap_y])
    return landmark_points


# Calculating the rectangle bounds for hands
def calculating_bound(frame, landmarks):
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    landmark_array = np.empty((0, 2), int)
    for _, lms in enumerate(landmarks.landmark):
        x_val = min(int(lms.x * frame_width), frame_width)
        y_val = min(int(lms.y * frame_height), frame_height)
        landmark_point = [np.array((x_val, y_val))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, width, height = cv2.boundingRect(landmark_array)
    return [x, y, width + x, height + y]


def pre_processing_lms(landmarklist):
    temp_lms = copy.deepcopy(landmarklist)
    # getting the relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_lms):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_lms[index][0] = temp_lms[index][0] - base_x
        temp_lms[index][1] = temp_lms[index][1] - base_y

    # converting the array to a one-demension. itertools.chain returns elements of lists until loop is exhausted
    temp_lms = list(itertools.chain.from_iterable(temp_lms))
    return temp_lms


def selectKey(key, mode):
    numbers = -1
    if 48 <= key <= 57:
        numbers = key - 48
    if key == 110:  # n
        mode = 1
    if key == 111:  # o
        mode = 0
    return numbers, mode


# to write to csv file
def loggingcsv(number, mode, landmarks):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        print(number)
        csv_path = "hand-gesture-recognition/keypoints.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmarks])
    return


# display handedness and predictions
def info_text(frame, handedness, hand_sign_text, rect_coordinates):
    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv2.putText(
        frame,
        info_text,
        (rect_coordinates[0], rect_coordinates[1] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    return frame


def display_info(frame, mode):
    start_text = 'Press Key "n" to collect training points'
    keying = "....keying... mode:" + str(mode)
    instruction = "Press (0-9) to key in labels"
    revert = 'Press "o" to go back to predicting...'

    if mode == 0:
        cv2.putText(
            frame, start_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (250, 220, 150)
        )
    if mode == 1:
        cv2.putText(
            frame, keying, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (250, 220, 150)
        )
        cv2.putText(
            frame, instruction, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (250, 220, 150)
        )
        cv2.putText(
            frame, revert, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 200)
        )


def correlationnn(dataset, threshold):
    col_cor = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_cor.add(colname)
    return col_cor


if __name__ == "__main__":
    main()
