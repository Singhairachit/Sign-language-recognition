import numpy as np
import cv2 as cv


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='slr/model/slr_model.tflite',
        num_threads=1,
    ):
        #: Initializing network using OpenCV DNN readNetFromTFLite
        self.net = cv.dnn.readNetFromTFLite(model_path)

    def __call__(self, landmark_list):
        #: Feeding landmarks to the network
        input_blob = np.array([landmark_list], dtype=np.float32)
        self.net.setInput(input_blob)

        #: Forward pass through the network
        result = self.net.forward()
        
        if max(np.squeeze(result)) > 0.5:
            #: Getting index of maximum accurate label
            result_index = np.argmax(np.squeeze(result))
            
            return result_index
        else:
            return 25

            
