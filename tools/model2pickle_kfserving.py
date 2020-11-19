import kfserving
from typing import List, Dict
import torch
from PIL import Image
import base64
import io
import cloudpickle as pickle
import numpy as np
import os
import argparse

DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/mnt/models"

def pickle_load(file):
    with open(file, 'rb') as f:
        model_s = f.read()
    return pickle.loads(model_s)

class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.infer_model = pickle_load(os.path.join(DEFAULT_LOCAL_MODEL_DIR,"export_model.pkl"))
        self.ready = True

    def model_infer(self, inputImg):
        inputImg = np.asarray(inputImg)[0,:]
        result = self.infer_model.predict(self.infer_model.model, inputImg.astype(np.uint8))
        return result

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        output = self.model_infer(inputs)
        return {"predictions": [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in output.items() }]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    parser.add_argument('--model_dir', required=False,
                        help='A URI pointer to the model directory')
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                        help='The name that the model is served under.')
    args, _ = parser.parse_known_args()
    model = KFServingSampleModel(args.model_name)
    model.load()
    kfserving.KFServer(workers=1).start([model])