print("INFO: Initializing System")
import copy
import csv
import os
import datetime

import pyautogui
import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv

from slr.model.classifier import KeyPointClassifier

from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks

from slr.utils.draw_debug import get_result_image
from slr.utils.draw_debug import get_fps_log_image
from slr.utils.draw_debug import draw_bounding_rect
from slr.utils.draw_debug import draw_hand_label
from slr.utils.draw_debug import show_fps_log
from slr.utils.draw_debug import show_result

from slr.utils.pre_process import calc_bounding_rect
from slr.utils.pre_process import calc_landmark_list
from slr.utils.pre_process import pre_process_landmark

from slr.utils.logging import log_keypoints
from slr.utils.logging import get_dict_form_list
from slr.utils.logging import get_mode
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LegacyClassification:
    def __init__(self, category):
        self.label = category.category_name
        self.score = category.score


class LegacyHandedness:
    def __init__(self, categories):
        self.classification = [LegacyClassification(c) for c in categories]


class LegacyHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class LegacyResults:
    def __init__(self, hand_landmarks, handedness):
        if hand_landmarks:
            self.multi_hand_landmarks = [LegacyHandLandmarks(hl) for hl in hand_landmarks]
            self.multi_handedness = [LegacyHandedness(h) for h in handedness]
        else:
            self.multi_hand_landmarks = None
            self.multi_handedness = None


class HandsAdapter:
    def __init__(self, max_num_hands, min_detection_confidence, min_tracking_confidence):
        base_options = python.BaseOptions(model_asset_path='slr/model/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        raw_result = self.detector.detect(mp_image)
        return LegacyResults(raw_result.hand_landmarks, raw_result.handedness)


def main():
    #: -
    #: Getting all arguments
    load_dotenv()
    args = get_args()

    keypoint_file = "slr/model/keypoint.csv"
    counter_obj = get_dict_form_list(keypoint_file)

    #: cv Capture
    CAP_DEVICE = args.device
    CAP_WIDTH = args.width
    CAP_HEIGHT = args.height

    #: mp Hands
    # USE_STATIC_IMAGE_MODE = args.use_static_image_mode
    USE_STATIC_IMAGE_MODE = True
    MAX_NUM_HANDS = args.max_num_hands
    MIN_DETECTION_CONFIDENCE = args.min_detection_confidence
    MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence

    #: Drawing Rectangle
    USE_BRECT = args.use_brect
    MODE = args.mode
    DEBUG = int(os.environ.get("DEBUG", "0")) == 1
    CAP_DEVICE = args.device

    print("INFO: System initialization Successful")
    print("INFO: Opening Camera")

    #: -
    #: Capturing image
    cap = cv.VideoCapture(CAP_DEVICE)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera device with index {CAP_DEVICE}.")
        print("Please check if your webcam is plugged in, active, and not in use by another application.")
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    background_image = cv.imread("resources/background_prediction.png")

    #: Background Image
    background_image = cv.imread("resources/background.png")
    # result_image = cv.imread("resources/result.png")

    #: -
    #: Setup hands
    hands = HandsAdapter(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    #: -
    #: Load Model
    keypoint_classifier = KeyPointClassifier()

    #: Loading labels
    keypoint_labels_file = "slr/model/label.csv"
    with open(keypoint_labels_file, encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in key_points]

    #: -
    #: FPS Measurement
    cv_fps = CvFpsCalc(buffer_len=10)
    print("INFO: System is up & running")
    #: -
    #: Main Loop Start Here...
    cv.namedWindow("Sign Language Recognition")
    while True:
        #: FPS of open cv frame or window
        fps = cv_fps.get()

        #: -
        #: Setup Quit key for program
        key = cv.waitKey(1)
        if key == 27:  # ESC key
            print("INFO: Exiting...")
            break
        elif key == 57:  # 9
            name = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(f'ss/{name}.png')

        # Check if window is closed by user (cross button)
        if cv.getWindowProperty("Sign Language Recognition", cv.WND_PROP_VISIBLE) < 1:
            print("INFO: Window closed by user. Exiting...")
            break


        #: -
        #: Camera capture
        success, image = cap.read()
        if not success:
            print("ERROR: Failed to read frame from camera. Exiting...")
            break
        
        image = cv.resize(image, (CAP_WIDTH, CAP_HEIGHT))
        
        #: Flip Image for mirror display
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        #: Converting to RBG from BGR
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)  #: Hand's landmarks
        image.flags.writeable = True

        #: -
        #: DEBUG - Showing Debug info
        if DEBUG:
            MODE = get_mode(key, MODE)
            fps_log_image = show_fps_log(fps_log_image, fps)

        #: -
        #: Start Detection
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                #: Calculate  BoundingBox
                use_brect = True
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                #: Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                #: Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                #: -
                #: Checking if in Prediction Mode or in Logging Mode
                #: If Prediction Mode it will predict the hand gesture
                #: If in Logging Mode it will Log key-points or landmarks to the csv file

                if MODE == 0:  #: Prediction Mode / Normal mode
                    #: Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 25:
                        hand_sign_text = ""
                    else:
                        hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    #: Showing Result
                    result_image = show_result(result_image, handedness, hand_sign_text)

                elif MODE == 1:  #: Logging Mode
                    log_keypoints(key, pre_processed_landmark_list, counter_obj, data_limit=1000)

                #: -
                #: Drawing debug info
                debug_image = draw_bounding_rect(debug_image, use_brect, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        #: -
        #: Set main video footage on Background

        # if MODE == 0:  #: Prediction Mode / Normal mode

        #     #: Changing to Prediction Background and setting main video footage on Background 
        #     background_image = cv.imread("resources/background_prediction.png")
        #     background_image[170:170 + 480, 50:50 + 640] = debug_image
        #     background_image[240:240 + 127, 731:731 + 299] = result_image
        #     background_image[678:678 + 30, 118:118 + 640] = fps_log_image

        # elif MODE == 1:  #: Logging Mode

        #     #: Changing to Logging Background and setting main video footage on Background
        #     background_image = cv.imread("resources/background_logging.png")
        #     background_image[170:170 + 480, 50:50 + 640] = debug_image
        #     background_image[240:240 + 127, 731:731 + 299] = result_image
        #     background_image[678:678 + 30, 118:118 + 640] = fps_log_image
        
        
        background_image[170:170 + 480, 50:50 + 640] = debug_image
        background_image[240:240 + 127, 731:731 + 299] = result_image
        background_image[678:678 + 30, 118:118 + 640] = fps_log_image

        # cv.imshow("Result", result_image)
        # cv.imshow("Main Frame", debug_image)
        
        cv.imshow("Sign Language Recognition", background_image)

    cap.release()
    cv.destroyAllWindows()

    print("INFO: Bye")


if __name__ == "__main__":
    main()
