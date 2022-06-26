import numpy as np
import os
from tflite_runtime.interpreter import Interpreter
from picamer2 import *
from PIL import Image
import cv2

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")

corresponding = {
    0: "cardboard",
    1: "plastic",
    2: "metal",
    3: "glass",
    4: "paper"
}
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.still_configuration(main={"size": (320, 320)}))
picam2.start()

detection_threshold = 0.5

try:
    while True:
        print("Got to the loop")
        image = picam2.capture_array("")

        print(image)

        # Create input tensor out of raw features
        interpreter.set_tensor(input_details[0]['index'], image)

        print("Running inference")
        # Run inference
        interpreter.invoke()

        # Gets detection outputs
        detected_scores = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        detected_classes = np.squeeze(interpreter.get_tensor(output_details[3]['index']))

        classes = []
        for i, score in enumerate(detected_scores):
          if score >= detection_threshold:
            classes.append((score, corresponding[int(detected_classes[i])]))
        print(classes)
        rawCapture.truncate(0)

except KeyboardInterrupt:
    print("Bye Bye")