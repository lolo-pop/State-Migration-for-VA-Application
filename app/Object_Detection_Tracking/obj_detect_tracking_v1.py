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
import flask 
from flask import request, jsonify
from flask_cors import CORS 
import json
import base64
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# remove all the annoying warnings from tf v1.10 to v1.13
import logging
logging.getLogger("tensorflow").disabled = True

import matplotlib
# avoid the warning "gdk_cursor_new_for_display:
# assertion 'GDK_IS_DISPLAY (display)' failed" with Python 3
matplotlib.use('Agg')

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
from server import Server

targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id[one]: one for one in targetClass2id}

app = flask.Flask(__name__)
server = None


def get_args():
  """Parse arguments and intialize some hyper-params."""
  global targetClass2id, targetid2class
  parser = argparse.ArgumentParser()

  parser.add_argument("--video_dir", default=None)
  parser.add_argument("--video_lst_file", default=None,
                      help="video_file_path = os.path.join(video_dir, $line)")

  parser.add_argument("--out_dir", default=None,
                      help="out_dir/$basename/%%d.json, start from 0 index. "
                           "This is the object box output. Leave this blank "
                           "when use tracking to avoid saving the obj class "
                           "output to save IO time.")

  parser.add_argument("--frame_gap", default=8, type=int)

  parser.add_argument("--threshold_conf", default=0.0001, type=float)

  parser.add_argument("--is_load_from_pb", action="store_true",
                      help="load from a frozen graph")
  parser.add_argument("--log_time_and_gpu", action="store_true")
  parser.add_argument("--util_log_interval", type=float, default=5.)
  parser.add_argument("--save_util_log_to", default=None,
                      help="save to a json for generating figures")

  # ------ for box feature extraction
  parser.add_argument("--get_box_feat", action="store_true",
                      help="this will generate (num_box, 256, 7, 7) tensor for "
                           "each frame")
  parser.add_argument("--box_feat_path", default=None,
                      help="output will be out_dir/$basename/%%d.npy, start "
                      "from 0 index")

  parser.add_argument("--version", type=int, default=4, help="model version")
  parser.add_argument("--is_coco_model", action="store_true",
                      help="is coco model, will output coco classes instead")
  parser.add_argument("--use_gn", action="store_true",
                      help="it is group norm model")
  parser.add_argument("--use_conv_frcnn_head", action="store_true",
                      help="group norm model from tensorpack uses conv head")


  # ---- gpu params
  parser.add_argument("--gpu", default=1, type=int, help="number of gpu")
  parser.add_argument("--gpuid_start", default=0, type=int,
                      help="start of gpu id")
  parser.add_argument("--im_batch_size", type=int, default=1)
  parser.add_argument("--fix_gpuid_range", action="store_true",
                      help="for junweil.pc")
  parser.add_argument("--use_all_mem", action="store_true")

  # --- for internal visualization
  parser.add_argument("--visualize", action="store_true")
  parser.add_argument("--vis_path", default=None)
  parser.add_argument("--vis_thres", default=0.7, type=float)

  # ----------- model params
  parser.add_argument("--num_class", type=int, default=15,
                      help="num catagory + 1 background")

  parser.add_argument("--model_path", default="/app/object_detection_model")

  parser.add_argument("--rpn_batch_size", type=int, default=256,
                      help="num roi per image for RPN  training")
  parser.add_argument("--frcnn_batch_size", type=int, default=512,
                      help="num roi per image for fastRCNN training")

  parser.add_argument("--rpn_test_post_nms_topk", type=int, default=1000,
                      help="test post nms, input to fast rcnn")

  parser.add_argument("--max_size", type=int, default=1920,
                      help="num roi per image for RPN and fastRCNN training")
  parser.add_argument("--short_edge_size", type=int, default=1080,
                      help="num roi per image for RPN and fastRCNN training")

  # use lijun video loader, this should deal with avi videos
  # with duplicate frames
  parser.add_argument(
      "--use_lijun_video_loader", action="store_true",
      help="use video loader from https://github.com/Lijun-Yu/diva_io")
  parser.add_argument("--use_moviepy", action="store_true")


  # ----------- tracking params
  parser.add_argument("--get_tracking", action="store_true",
                      help="this will generate tracking results for each frame")
  parser.add_argument("--tracking_dir", default="/tmp",
                      help="output will be out_dir/$videoname.txt, start from 0"
                           " index")
  parser.add_argument("--tracking_objs", default="Person,Vehicle",
                      help="Objects to be tracked, default are Person and "
                           "Vehicle")
  parser.add_argument("--min_confidence", default=0.85, type=float,
                      help="Detection confidence threshold. Disregard all "
                           "detections that have a confidence lower than this "
                           "value.")
  parser.add_argument("--min_detection_height", default=0, type=int,
                      help="Threshold on the detection bounding box height. "
                           "Detections with height smaller than this value are "
                           "disregarded")
  # this does not make a big difference
  parser.add_argument("--nms_max_overlap", default=0.85, type=float,
                      help="Non-maxima suppression threshold: Maximum detection"
                           " overlap.")

  parser.add_argument("--max_iou_distance", type=float, default=0.5,
                      help="Iou distance for tracker.")
  parser.add_argument("--max_cosine_distance", type=float, default=0.5,
                      help="Gating threshold for cosine distance metric (object"
                           " appearance).")
  # nn_budget smaller more tracks
  parser.add_argument("--nn_budget", type=int, default=5,
                      help="Maximum size of the appearance descriptors gallery."
                           " If None, no budget is enforced.")

  parser.add_argument("--bupt_exp", action="store_true",
                      help="activity box experiemnt")

  # ---- tempory: for activity detection model
  parser.add_argument("--actasobj", action="store_true")
  parser.add_argument("--actmodel_path",
                      default="/app/activity_detection_model")

  parser.add_argument("--resnet152", action="store_true", help="")
  parser.add_argument("--resnet50", action="store_true", help="")
  parser.add_argument("--resnet34", action="store_true", help="")
  parser.add_argument("--resnet18", action="store_true", help="")
  parser.add_argument("--use_se", action="store_true",
                      help="use squeeze and excitation in backbone")
  parser.add_argument("--use_frcnn_class_agnostic", action="store_true",
                      help="use class agnostic fc head")
  parser.add_argument("--use_resnext", action="store_true", help="")
  parser.add_argument("--use_att_frcnn_head", action="store_true",
                      help="use attention to sum [K, 7, 7, C] feature into"
                           " [K, C]")

  # ------ 04/2020, efficientdet
  parser.add_argument("--is_efficientdet", action="store_true")
  parser.add_argument("--efficientdet_modelname", default="efficientdet-d0")
  parser.add_argument("--efficientdet_max_detection_topk", type=int,
                      default=5000, help="#topk boxes before NMS")
  parser.add_argument("--efficientdet_min_level", type=int, default=3)
  parser.add_argument("--efficientdet_max_level", type=int, default=7)

  # ---- COCO Mask-RCNN model
  parser.add_argument("--add_mask", action="store_true")

  # --------------- exp junk
  parser.add_argument("--use_dilations", action="store_true",
                      help="use dilations=2 in res5")
  parser.add_argument("--use_deformable", action="store_true",
                      help="use deformable conv")
  parser.add_argument("--add_act", action="store_true",
                      help="add activitiy model")
  parser.add_argument("--finer_resolution", action="store_true",
                      help="fpn use finer resolution conv")
  parser.add_argument("--fix_fpn_model", action="store_true",
                      help="for finetuneing a fpn model, whether to fix the"
                           " lateral and poshoc weights")
  parser.add_argument("--is_cascade_rcnn", action="store_true",
                      help="cascade rcnn on top of fpn")
  parser.add_argument("--add_relation_nn", action="store_true",
                      help="add relation network feature")

  parser.add_argument("--test_frame_extraction", action="store_true")
  parser.add_argument("--use_my_naming", action="store_true")


  
  # for efficient use of COCO model classes
  parser.add_argument("--use_partial_classes", action="store_true")

    # --------------- App name
  parser.add_argument("--app_name", default=None)
  # --------------- State store
  parser.add_argument("--state_store", default=None)
  # --------------- Pubsub config



  args = parser.parse_args()


  if args.use_partial_classes:
    args.is_coco_model = True
    args.partial_classes = [classname for classname in coco_obj_to_actev_obj]

  assert args.gpu == args.im_batch_size  # one gpu one image
  assert args.gpu == 1, "Currently only support single-gpu inference"

  if args.is_load_from_pb:
    args.load_from = args.model_path

  args.controller = "/cpu:0"  # parameter server

  targetid2class = targetid2class
  targetClass2id = targetClass2id

  if args.actasobj:
    from class_ids import targetAct2id
    targetClass2id = targetAct2id
    targetid2class = {targetAct2id[one]: one for one in targetAct2id}
  if args.bupt_exp:
    from class_ids import targetAct2id_bupt
    targetClass2id = targetAct2id_bupt
    targetid2class = {targetAct2id_bupt[one]: one for one in targetAct2id_bupt}

  assert len(targetClass2id) == args.num_class, (len(targetClass2id),
                                                 args.num_class)


  assert args.version in [2, 3, 4, 5, 6], \
         "Currently we only have version 2-6 model"

  if args.version == 2:
    pass
  elif args.version == 3:
    args.use_dilations = True
  elif args.version == 4:
    args.use_frcnn_class_agnostic = True
    args.use_dilations = True
  elif args.version == 5:
    args.use_frcnn_class_agnostic = True
    args.use_dilations = True
  elif args.version == 6:
    args.use_frcnn_class_agnostic = True
    args.use_se = True

  if args.is_coco_model:
    assert args.version == 2
    targetClass2id = coco_obj_class_to_id
    targetid2class = coco_obj_id_to_class
    args.num_class = 81
    if args.use_partial_classes:
      partial_classes = ["BG"] + args.partial_classes
      targetClass2id = {classname: i
                        for i, classname in enumerate(partial_classes)}
      targetid2class = {targetClass2id[o]: o for o in targetClass2id}

  # ---- 04/2020, efficientdet
  if args.is_efficientdet:
    targetClass2id = coco_obj_class_to_id
    targetid2class = coco_obj_id_to_class
    args.num_class = 81
    args.is_coco_model = True

  args.classname2id = targetClass2id
  args.classid2name = targetid2class
  # ---------------more defautls
  args.is_pack_model = False
  args.diva_class3 = True
  args.diva_class = False
  args.diva_class2 = False
  args.use_small_object_head = False
  args.use_so_score_thres = False
  args.use_so_association = False
  #args.use_gn = False
  #args.use_conv_frcnn_head = False
  args.so_person_topk = 10
  args.use_cpu_nms = False
  args.use_bg_score = False
  args.freeze_rpn = True
  args.freeze_fastrcnn = True
  args.freeze = 2
  args.small_objects = ["Prop", "Push_Pulled_Object",
                        "Prop_plus_Push_Pulled_Object", "Bike"]
  args.no_obj_detect = False
  #args.add_mask = False
  args.is_fpn = True
  # args.new_tensorpack_model = True
  args.mrcnn_head_dim = 256
  args.is_train = False

  args.rpn_min_size = 0
  args.rpn_proposal_nms_thres = 0.7
  args.anchor_strides = (4, 8, 16, 32, 64)

  # [3] is 32, since we build FPN with r2,3,4,5, so 2**5
  args.fpn_resolution_requirement = float(args.anchor_strides[3])

  #if args.is_efficientdet:
  #  args.fpn_resolution_requirement = 128.0  # 2 ** max_level
  #  args.short_edge_size = np.ceil(
  #      args.short_edge_size / args.fpn_resolution_requirement) * \
  #          args.fpn_resolution_requirement
  args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * \
                  args.fpn_resolution_requirement

  args.fpn_num_channel = 256

  args.fpn_frcnn_fc_head_dim = 1024

  # ---- all the mask rcnn config

  args.resnet_num_block = [3, 4, 23, 3]  # resnet 101
  args.use_basic_block = False  # for resnet-34 and resnet-18
  if args.resnet152:
    args.resnet_num_block = [3, 8, 36, 3]
  if args.resnet50:
    args.resnet_num_block = [3, 4, 6, 3]
  if args.resnet34:
    args.resnet_num_block = [3, 4, 6, 3]
    args.use_basic_block = True
  if args.resnet18:
    args.resnet_num_block = [2, 2, 2, 2]
    args.use_basic_block = True

  args.anchor_stride = 16  # has to be 16 to match the image feature
  args.anchor_sizes = (32, 64, 128, 256, 512)

  args.anchor_ratios = (0.5, 1, 2)

  args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
  # iou thres to determine anchor label
  # args.positive_anchor_thres = 0.7
  # args.negative_anchor_thres = 0.3

  # when getting region proposal, avoid getting too large boxes
  args.bbox_decode_clip = np.log(args.max_size / 16.0)

  # fastrcnn
  args.fastrcnn_batch_per_im = args.frcnn_batch_size
  args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype="float32")

  args.fastrcnn_fg_thres = 0.5  # iou thres
  # args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

  # testing
  args.rpn_test_pre_nms_topk = 6000

  args.fastrcnn_nms_iou_thres = 0.5

  args.result_score_thres = args.threshold_conf
  args.result_per_im = 100

  return args



def check_args():
  """Check the argument."""
  assert args.frame_gap >= 1
  if args.get_box_feat:
    assert args.box_feat_path is not None
    if not os.path.exists(args.box_feat_path):
      os.makedirs(args.box_feat_path)
  #print("cv2 version %s" % (cv2.__version__)




@app.route("/init", methods=["POST"])
def initialize_server():
    global server
    if not server:
        # logging.basicConfig(format="%(name)s -- %(levelname)s -- %(linen o)s -- %(message)s", level="INFO")
        t1 = time.time()
        server = Server(args)
        t2 = time.time()
        print("components initialization:", t2-t1, flush=True)
        img_path = 'VIRAT_S_000205_05_001092_001124_F_00000001.jpg'
        img = cv2.imread(img_path)
        print(img.shape, flush=True)
        
        server.run_detection(img, 'init')
        server.reset_tracker_state()
        t3 = time.time()
        print("pre-initialization:", t3-t2, flush=True)
        return "ok"
    else:
        server.reset_tracker_state()
        return "ok"


@app.route("/infer", methods=["POST"])
def infer():
    t1 = time.time()
    req = request.json
    tmp_str = ''
    tmp_dict = {}
    if type(req) == type(tmp_str):
      req = json.loads(req)
      
    # print(request.json['data'], flush=True)
    jpg_as_str = req['frame']
    cur_frame = req['hash']
    jpg_as_bytes = jpg_as_str.encode('ascii')
    jpg_original = base64.b64decode(jpg_as_bytes)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # if model is None:
    #   print("model is not initialized yet")
    # else:
    img = cv2.imdecode(jpg_as_np, flags=1)
    # print(img.shape, flush=True)

    # im = img.astype("float32")
    t2 = time.time()
    # print('111', t2-t1, flush=True)
    server.run_detection(img, cur_frame)

    t3 = time.time()
    # print('333', t3-t2, flush=True)
    return jsonify({"success": True, 'frame': cur_frame})





if __name__ == "__main__":
  global args
  args = get_args()
  check_args()
  t1 = time.time()

  app.run(host="0.0.0.0", port=5000)
