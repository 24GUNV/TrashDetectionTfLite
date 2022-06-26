import numpy as np
import os
import picamera
import picamera.array
from tflite_runtime.interpreter import Interpreter
from time import sleep

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

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

detection_threshold = 0.5

with picamera.PiCamera() as camera:
    sleep(0.5) # waiting for the camera to warm up
    try:
        camera.resolution = (320, 320)
        while True:
            output = np.empty((1*320*320*3), dtype=np.uint8)
            camera.capture(output, 'rgb')
            output = output.reshape((1, 320, 320, 3))

            # Create input tensor out of raw features
            interpreter.set_tensor(input_details[0]['index'], output)

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
            sleep(0.5)

    except KeyboardInterrupt:
        print("Bye Bye")