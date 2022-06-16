import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
from icevision.all import *
from icevision.models import *


# Loading in the model
model_loaded = model_from_checkpoint("/content/drive/MyDrive/AI/garbageclassification.pth")
model_type = model_loaded["model_type"]
backbone = model_loaded["backbone"]
class_map = model_loaded["class_map"]
img_size = model_loaded["img_size"]
model = model_loaded["model"]

img_size = model_loaded["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(224), tfms.A.Normalize()])
# Function to return the predictions
def predictions(frame):
  img = PIL.Image.fromarray(frame)
  pred_dict = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.5)
  return pred_dict['detection']

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # makes predictions about the current frame
        detections = predictions(frame)

        for i in range(len(detections['bboxes'])):
          left, top, right, bottom = detections['bboxes'][i].xmin,detections['bboxes'][i].ymin,detections['bboxes'][i].xmax,detections['bboxes'][i].ymax
          img = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
          img = cv2.putText(img, "{} [{:.2f}]".format(detections['labels'][i], float(detections['scores'][i])),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)