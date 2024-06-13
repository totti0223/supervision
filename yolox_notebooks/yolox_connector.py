import sys
sys.path.append("../YOLOX")
# import argparse
import os
import time
from loguru import logger
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np

class YOLOX:
    def __init__(self, model_path=None, exp_path=None, ckpt_path=None, tsize=None, nms=None, conf=None, class_names=None):
        if exp_path is None:
            # get the path below YOLOX which was added from the sys.path.append
            self.exp_path = os.path.join("../YOLOX/exps/default/yolox_tiny.py")
        self.exp = get_exp(self.exp_path, None)
        if conf is not None:
            self.exp.test_conf = conf
        if nms is not None:
            self.exp.nmsthre = nms
        if tsize is not None:
            self.exp.test_size = (tsize, tsize)
            
        if ckpt_path is None:
            self.ckpt_path = "../YOLOX/yolox_tiny.pth"            
        self.model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
        
        
        self.DEVICE = "cpu"
        if torch.cuda.is_available():
            logger.info("Using GPU.")
            self.DEVICE = "cuda"
            self.model.cuda()
        else:
            logger.info("Using CPU.")
        self.model.eval()
        ckpt = torch.load(self.ckpt_path, map_location=self.DEVICE)
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        
        
        self.preproc = ValTransform(legacy=False)
        self.decoder = None
        if class_names is None:
            self.model.names = COCO_CLASSES
        else:
            self.model.names = class_names
        
    def __call__(self, img):
        # image should be a bgr image uint8 read by cv2.imread
        ratio = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
        img, _ = self.preproc(img, None, self.exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.DEVICE == "cuda":
            img = img.cuda()
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
                
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf,
                self.exp.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            
        output = outputs[0]
        
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        bboxes = bboxes.cpu().detach().numpy().copy()
        
        clss = output[:, 6]
        clss = clss.cpu().detach().numpy().copy().astype(int)
        scores = output[:, 4] * output[:, 5]
        scores = scores.cpu().detach().numpy().copy()
        logger.info(self.model.names)
        logger.info(clss) 
        return [bboxes, scores, clss, [self.model.names[x] for x in clss]]
    
    
        
        
        
        
        
        