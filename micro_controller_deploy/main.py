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

interpreter.allocate_tensors()

# input details
print(input_details)
# output details
print(output_details)
for x in output_details:
  print(x)

img = cv2.imread("aibuilders/object_detection/images/paper/paper1.jpg")
new_img = cv2.resize(img, (320, 320))

# Create input tensor out of raw features
interpreter.set_tensor(input_details[0]['index'], [new_img])

# Run inference
interpreter.invoke()

# Gets detection outputs
detected_scores = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
detected_classes = np.squeeze(interpreter.get_tensor(output_details[3]['index']))

# Print the results of inference
print(detected_classes)
print(detected_scores)

detection_threshold = 0.3

classes = []
for i, score in enumerate(detected_scores):
  if score >= detection_threshold:
    classes.append((score, corresponding[int(detected_classes[i])]))
print(classes)