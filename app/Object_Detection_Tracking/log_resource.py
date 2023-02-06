# coding=utf-8
"""
  run object detection and tracking inference
"""

import argparse
import math
import json
import random
import sys
import time
import threading
import operator
import os
import pickle

# avoid the warning "gdk_cursor_new_for_display:
# assertion 'GDK_IS_DISPLAY (display)' failed" with Python 3



# detection stuff


from utils import get_op_tensor_name
from utils import parse_nvidia_smi
from utils import sec2time
from utils import PerformanceLogger




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
  parser.add_argument("--util_log_interval", type=float, default=1)
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
  parser.add_argument("--fps", type=int, default = 5)
  parser.add_argument("--frame_number", type=int, default = 100)
  parser.add_argument("--pid1", type=int)
  parser.add_argument("--pid2", type=int)
  parser.add_argument("--pid3", type=int)
  args = parser.parse_args()


  return args



if __name__ == "__main__":
  args = get_args()

  start_time = time.time()

  gpuid_range = (args.gpuid_start, args.gpu)
  if args.fix_gpuid_range:
    gpuid_range = (0, 1)

  performance_logger = PerformanceLogger(
      gpuid_range,
      args.util_log_interval,
      args.pid1, args.pid2, args.pid3)
  performance_logger.start()

  time.sleep(200)

  end_time = time.time()
  performance_logger.end()
  logs = performance_logger.logs
  print(logs, flush=True)
  with open("process_monitor.csv", "a+") as f:
    f.write("cpu1%,cpu2%,cpu3%,gpu%,mem%\n") # titles
    for i in range(len(logs["cpu_utilization1"])):
      line = str(logs["cpu_utilization1"][i])+","+str(logs["cpu_utilization2"][i])+","+ \
             str(logs["cpu_utilization3"][i])+","+str(logs["mem_utilization1"][i])+","+ \
             str(logs["mem_utilization2"][i])+","+str(logs["mem_utilization3"][i])+","+ \
             str(logs["gpu_utilization"][i])+","+ \
             str(logs["gpu_memory"][i])+","+str(logs["ram_used"][i])
      print (line)
      f.write(line + "\n")