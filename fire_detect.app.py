import sys
import numpy as np
from imageai.Detection.Custom import CustomObjectDetection


detector = None
labels = []
anchors = []

weights = ''
config = ''
conf_threshold = 0.5
iou_threshold = 0.4


def on_set(k, v):
    if k == 'weights':
        global weights
        weights = v
    elif k == 'config':
        global config
        config = v
    elif k == 'conf_threshold':
        global conf_threshold
        conf_threshold = float(v)
    elif k == 'iou_threshold':
        global iou_threshold
        iou_threshold = float(v)


def on_get(k):
    if k == 'weights':
        return weights
    elif k == 'config':
        return config
    elif k == 'conf_threshold':
        return str(conf_threshold)
    elif k == 'iou_threshold':
        return str(iou_threshold)


def on_init():
    global detector
    global labels
    global anchors

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=weights)
    detector.setJsonPath(configuration_json=config)
    detector.loadModel()

    detection_model_json = json.load(open(config))

    labels = detection_model_json["labels"]
    anchors = detection_model_json["anchors"]


def on_run(image):
    draw_image, objects = detector.detectObjectsFromImage(input_image=image, input_type="array", output_type="array",
                                                          minimum_percentage_probability=conf_threshold * 100, nms_treshold=iou_threshold)

    bboxes = []
    for o in objects:
        b = o['box_points']
        b.appen(o['percentage_probability'] / 100)
        b.appen(o['name'])
        bboxes.append(b)

    return {
        'image': draw_image,
        'bboxes': np.array(bboxes)
    }
