import cv2
import numpy as np

# Load YOLOv3 model with OpenCV
net = cv2.dnn.readNet('./Yolo/yolov3.weights', './Yolo/yolov3.cfg')

# Load class names
classes = []
with open('./Yolo/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load an image
image = cv2.imread('./Media/object_sample.jpg')

# Get image dimensions
height, width, _ = image.shape

# Prepare the image for YOLOv3
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the input to the network
net.setInput(blob)

# Get output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Forward pass
outs = net.forward(layer_names)

# Lists for class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Minimum confidence threshold for detection
conf_threshold = 0.5

# Non-maximum suppression threshold
nms_threshold = 0.4

# Process each output layer
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # YOLO returns center (x, y) of the object, width, height, angles, and class_id
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Append to lists
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes and labels
for i in indices:
    i = i.item()
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]

    # Draw rectangle and label
    color = (0, 255, 0)  # BGR color
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./Media/output_object_dr.jpg', image)