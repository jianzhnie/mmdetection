# Import the required modules
import cloudpickle as pickle
from mmdet.apis import init_detector
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


class InferenceModel:

	def __init__(self, model, predict):
		self.model = model
		self.predict = predict

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def custom_inference_detector(model, img):
    """Inference image(s) with the detector.test_pipeline

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    print(result)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    
    # Process detection mask
    if len(bbox_result ) < 1:
        output_dict = {}
        output_dict['num_detections'] = 0
        output_dict['detection_classes'] = []
        output_dict['detection_boxes'] = []
        output_dict['detection_scores'] = []
    
    else:
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        output_dict = {}
        output_dict['num_detections'] = bboxes.shape[0]
        output_dict['detection_classes'] = labels
        output_dict['detection_boxes'] = bboxes[:,:4]
        output_dict['detection_scores'] = bboxes[:,-1]

    # Process detection mask
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        output_dict['detection_masks'] = np.array(segms)
    return output_dict

def pickle_dump(obj, file):
	pickled_lambda = pickle.dumps(obj)
	with open(file, 'wb') as f:
		f.write(pickled_lambda)


def pickle_load(file):
	with open(file, 'rb') as f:
		model_s = f.read()
	return pickle.loads(model_s)


def save_image(data):
	imgname = "temp.jpg"
	with open(imgname, 'wb')as f:
		f.write(data)
	return imgname


def dump_infer_model(checkpoint_file, config_file, output_file, labelfile,
                     device='cuda:0'):
	print("------------------------------")
	print("START EXPORT MODEL")
	print("------------------------------")
	model = init_detector(config=config_file, checkpoint=checkpoint_file, device=device)
	infer_model = InferenceModel(model, custom_inference_detector)
	pickle_dump(infer_model, output_file)
	writeLabels(infer_model.model.CLASSES, labelfile)
	print("------------------------------")
	print("SUCCESS EXPORT MODEL")
	print("------------------------------")


from PIL import Image
import io
import os
import json


def model_infer(pickle_file, img_bytes):
	img = Image.open(io.BytesIO(img_bytes))
	inputImg = np.asarray(img)
	infer_model = pickle_load(pickle_file)
	print(infer_model.model.CLASSES)
	result = infer_model.predict(infer_model.model, inputImg)
	print(result)


def writeLabels(CLASSES, label_file):
	data = []
	with open(label_file, 'w') as jsonfile:
		for key, value in enumerate(CLASSES, start=1):
			data.append({"id": int(key), "display_name": value})
		json.dump(data, jsonfile)


if __name__ == '__main__':
	# dump model
	config_file = 'configs_my/faster_rcnn_r50_fpn_1x.py'
	checkpoint_file = 'work_dir/epoch_12.pth'
	output_file = 'work_dir/export_model.pkl'
	label_file = 'work_dir/class_names.json'
	dump_infer_model(checkpoint_file, config_file, output_file, label_file)
	img_file = 'demo/0.jpg'
	img_bytes = open(img_file, 'rb').read()
	model_infer("work_dir/export_model.pkl", img_bytes)