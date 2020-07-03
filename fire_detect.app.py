import sys
import json
import numpy as np
import tensorflow as tf
from imageai.Detection.Custom import CustomObjectDetection


detector = None
labels = []
anchors = []

weights = ''
config = ''
conf_threshold = 50
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
        conf_threshold = float(v) * 100
    elif k == 'iou_threshold':
        global iou_threshold
        iou_threshold = float(v)


def on_get(k):
    if k == 'weights':
        return weights
    elif k == 'config':
        return config
    elif k == 'conf_threshold':
        return str(conf_threshold / 100)
    elif k == 'iou_threshold':
        return str(iou_threshold)


def on_init():
    global detector
    global labels
    global anchors

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            sys.stdout.write(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs\n")
            sys.stdout.flush()
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            sys.stdout.write(e)
            sys.stdout.flush()

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=weights)
    detector.setJsonPath(configuration_json=config)
    detector.loadModel()

    detection_model_json = json.load(open(config))

    labels = detection_model_json["labels"]
    anchors = detection_model_json["anchors"]
    return True


def on_run(image):
    draw_image, objects = detector.detectObjectsFromImage(input_image=image, input_type="array", output_type="array",
                                                          minimum_percentage_probability=conf_threshold, nms_treshold=iou_threshold)

    bboxes = []
    for o in objects:
        b = o['box_points']
        b.append(o['percentage_probability'] / 100)
        b.append(labels.index(o['name']))
        bboxes.append(b)

    sys.stdout.write(f"[fire_detect.on_run] conf_threshold {conf_threshold}\n")
    sys.stdout.write(f"[fire_detect.on_run] iou_threshold {iou_threshold}\n")
    sys.stdout.write(f"[fire_detect.on_run] objects {objects}\n")
    sys.stdout.write(f"[fire_detect.on_run] bboxes {bboxes}\n")
    sys.stdout.flush()

    return {
        'image': draw_image,
        'bboxes': np.array(bboxes)
    }
