import streamlit as st
import numpy as np
import os
import av
from streamlit_webrtc import *
import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)

import cv2
from PIL import Image

model_path = 'streamlit_deploy/model.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Classes
classes = {
    0: "cardboard",
    1: "plastic",
    2: "metal",
    3: "glass",
    4: "paper"
}

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(5, 3), dtype=np.uint8)


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.image.convert_image_dtype(image_path, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


def upload_image():
    st.title("Upload an image!")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])

    detection_threshold = st.slider('What should the detection_threshold be?', 0.0, 1.0, 0.3)
    st.write(f"The Detection Threshold is {detection_threshold}")

    if uploaded_file is not None:
        file = np.array(Image.open(uploaded_file))
        detection_result_image = run_odt_and_draw_results(
            file,
            interpreter,
            threshold=detection_threshold
        )

        img = Image.fromarray(detection_result_image)

        st.image(img, 'Results!')


def livestream():

    class VideoProcessor:
        def recv(self, frame):
            DETECTION_THRESHOLD = 0.3

            arr = np.array(frame)

            # Run inference and draw detection result on the local copy of the original file
            detection_result_image = run_odt_and_draw_results(
                arr,
                interpreter,
                threshold=DETECTION_THRESHOLD
            )

            # Show the detection result
            return av.VideoFrame.from_ndarray(detection_result_image, format="brg24")
    st.write("Doesnt work for me due to network issues")

    webrtc_streamer(key="example", rtc_configuration=
    {
        "iceServers": [{"urls": ["stun.l.google.com:19302"]}],
    })

def intro():
    st.title('AI Builders Demo')
    st.write('Made by Gun from Hidden Hammers')
    st.write("Made using Tensorflow lite")
    st.image(Image.open('streamlit_deploy/sources/tf_lite.png'))


page_names_to_funcs = {
    "Introduction": intro,
    "Upload an image!": upload_image,
    "Do it Live!": livestream,
}

demo_name = st.sidebar.selectbox("Versions", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
