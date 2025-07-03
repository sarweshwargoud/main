import os
import cv2
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
import pytesseract
import pyttsx3
import torch
import torch.nn.functional as F


print("All  modules are working!")


from torchvision import transforms as trn
from torch.autograd import Variable
from PIL import Image

from utils import label_map_util as label_map_util

from utils import visualization_utils as vis_util

# === Voice Engine ===
engine = pyttsx3.init()

# === Model Setup ===
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# === Download model if not present ===
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model...')
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall()
    print('Download complete.')

# === Load TensorFlow Model ===
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# === Load Label Map ===
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# === Scene Recognition (Places365) Setup ===
arch = 'resnet18'
scene_model_file = f'whole_{arch}_places365_python36.pth.tar'
scene_label_file = 'categories_places365.txt'

if not os.path.exists(scene_model_file):
    print('Downloading Places365 model...')
    urllib.request.urlretrieve(f'http://places2.csail.mit.edu/models_places365/{scene_model_file}', scene_model_file)

if not os.path.exists(scene_label_file):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt',
        scene_label_file
    )

scene_model = scene_model = torch.load(scene_model_file, map_location=torch.device('cpu'), weights_only=False)

scene_model.eval()

scene_classes = [line.strip().split(' ')[0][3:] for line in open(scene_label_file)]
centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Pytesseract Setup ===
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === Start Webcam ===
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            # === Save scene & classify if 'b' pressed ===
            if key == ord('b'):
                filename = 'scene.jpg'
                cv2.imwrite(filename, image_np)

                img = Image.open(filename)
                input_img = centre_crop(img).unsqueeze(0)
                input_var = Variable(input_img)

                logit = scene_model.forward(input_var)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)

                print("\nScene prediction:")
                engine.say("Possible scene may be")
                for i in range(5):
                    print(f"{probs[i]:.3f} -> {scene_classes[idx[i]]}")
                    engine.say(scene_classes[idx[i]])
                engine.runAndWait()

            # === Text-to-speech for OCR if 'r' pressed ===
            if key == ord('r'):
                text = pytesseract.image_to_string(image_np)
                print(f"OCR Text: {text}")
                engine.say("Text detected")
                engine.say(text)
                engine.runAndWait()

            # === Exit if 't' pressed ===
            if key == ord('t'):
                break

            # === Object Detection ===
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4
            )

            # === Proximity Alert System ===
            for i, box in enumerate(boxes[0]):
                class_id = int(classes[0][i])
                score = scores[0][i]
                if score < 0.5:
                    continue

                mid_x = (box[1] + box[3]) / 2
                mid_y = (box[0] + box[2]) / 2
                apx_distance = round((1 - (box[3] - box[1])) ** 4, 1)
                cv2.putText(image_np, f"{apx_distance}", (int(mid_x * 800), int(mid_y * 450)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                danger_zone = 0.5
                centered = 0.3 < mid_x < 0.7

                alert_msg = None
                if class_id in [3, 6, 8] and apx_distance <= danger_zone and centered:
                    alert_msg = "Warning - Vehicle Approaching"
                elif class_id == 44 and apx_distance <= danger_zone and centered:
                    alert_msg = "Warning - Bottle very close"
                elif class_id == 1 and apx_distance <= danger_zone and centered:
                    alert_msg = "Warning - Person very close"

                if alert_msg:
                    print(alert_msg)
                    cv2.putText(image_np, "WARNING!!!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    engine.say(alert_msg)
                    engine.runAndWait()

            # === Display Output ===
            cv2.imshow('Object & Scene Detection', cv2.resize(image_np, (1024, 768)))

# === Clean Up ===
cap.release()
cv2.destroyAllWindows()
