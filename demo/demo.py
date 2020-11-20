from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = 'configs_my/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'work_dir/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'demo/0.jpg'
result = inference_detector(model, img)
print(result)
# show the results
show_result_pyplot(img, result, model.CLASSES)