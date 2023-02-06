# coding=utf-8
"""
  run object detection and tracking inference
"""

import argparse
import cv2
import math
import json
import random
import sys
import time
import threading
import operator
import os
import sys
import json
import base64
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# remove all the annoying warnings from tf v1.10 to v1.13
import logging
logging.getLogger("tensorflow").disabled = True


# from tqdm import tqdm

import numpy as np
import tensorflow as tf

# detection stuff
from models import get_model
from models import resizeImage

from nn import fill_full_mask
from utils import get_op_tensor_name
from utils import parse_nvidia_smi
from utils import sec2time
from utils import PerformanceLogger

import pycocotools.mask as cocomask

# tracking stuff
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import create_obj_infos,linear_inter_bbox,filter_short_objs

# for mask
import pycocotools.mask as cocomask

# class ids stuff
from class_ids import targetClass2id_new_nopo
from class_ids import coco_obj_class_to_id
from class_ids import coco_obj_id_to_class
from class_ids import coco_obj_to_actev_obj
from class_ids import coco_id_mapping

from state import write_state
from state import read_state
from state import initialize_state
from detector import Detector

targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id[one]: one for one in targetClass2id}



class Server:
  def __init__(self, args):
    self.args = args
    self.logger = logging.getLogger('server')
    handler = logging.NullHandler()
    self.logger.addHandler(handler)
    self.detector = Detector(args)
    self.logger.info("Server started")
    
  def run_detection(self, image, cur_frame):
    self.logger.info(f"Running inference on {cur_frame} frame")
    im = image.astype("float32")
    self.detector.run_inference(im, cur_frame)

  def reset_tracker_state(self):
    self.detector.reset_tracker_state()
    self.logger.info("tracker state has been reset")
    