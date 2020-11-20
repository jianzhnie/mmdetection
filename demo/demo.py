from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
from PIL import Image
import io
import os
import numpy as np
import cv2


config_file = 'configs_my/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'work_dir/epoch_11.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img_file = 'demo/xray/0.jpg'
img_bytes = open(img_file, 'rb').read()
img = Image.open(io.BytesIO(img_bytes))
inputImg = np.asarray(img).astype(np.uint8)

result = inference_detector(model, inputImg)
# show the results
show_result(img_file, result, model.CLASSES, score_thr=0.5, out_file='demo/xray/0_pred_1.jpg')


img_np_arr = np.frombuffer(img_bytes, np.uint8)
inputImg = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
result = inference_detector(model, inputImg)
# show the results
show_result(img_file, result, model.CLASSES, score_thr=0.5, out_file='demo/xray/0_pred_2.jpg')
