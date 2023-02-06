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
# from utils import get_op_tensor_name
# from utils import parse_nvidia_smi
# from utils import sec2time
# from utils import PerformanceLogger

import pycocotools.mask as cocomask

# tracking stuff
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import create_obj_infos,linear_inter_bbox,filter_short_objs


# class ids stuff
from class_ids import targetClass2id_new_nopo
from class_ids import coco_obj_class_to_id
from class_ids import coco_obj_id_to_class
from class_ids import coco_obj_to_actev_obj
from class_ids import coco_id_mapping

from state import write_state
from state import read_state
from state import initialize_state

from dapr.clients import DaprClient
from dapr.clients.grpc._request import TransactionalStateOperation, TransactionOperationType
from dapr.clients.grpc._state import StateItem

targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id[one]: one for one in targetClass2id}



class Detector:
  def __init__(self, args):
    self.args = args
    self.logger = logging.getLogger('detector')
    handler = logging.NullHandler()
    self.logger.addHandler(handler)
    self.model = get_model(self.args, self.args.gpuid_start, controller=self.args.controller)
    
    self.tfconfig = tf.ConfigProto(allow_soft_placement=True)
    if not self.args.use_all_mem:
      self.tfconfig.gpu_options.allow_growth = True
    self.tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i
              for i in range(self.args.gpuid_start, self.args.gpuid_start + self.args.gpu)]))  
    self.sess = tf.Session(config=self.tfconfig)
    self.tracking_objs = self.args.tracking_objs.split(",")
      # print(tracking_objs)
    self.tracker_dict = {}
    self.tmp_tracking_results_dict = {}
    self.tracking_results_dict = {}
    for tracking_obj in self.tracking_objs:
        metric = metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.args.max_cosine_distance, self.args.nn_budget)
        self.tracker_dict[tracking_obj] = Tracker(metric, max_iou_distance=self.args.max_iou_distance)
        self.tracking_results_dict[tracking_obj] = []
        self.tmp_tracking_results_dict[tracking_obj] = {}
    videoname = "test"
    if self.args.out_dir is not None:  # not saving box json to save time
      self.video_out_path = os.path.join(self.args.out_dir, videoname)
      if not os.path.exists(self.video_out_path):
        os.makedirs(self.video_out_path)
    if self.args.get_box_feat:
      self.feat_out_path = os.path.join(self.args.box_feat_path, videoname)
      if not os.path.exists(self.feat_out_path):
        os.makedirs(self.feat_out_path)
    self.logger.info("Detector initialized")
    self.d = initialize_state()
  
  
   
  def save_as_json(self, final_boxes, final_probs, final_labels, cur_frame):
    # save as json

    pred = []
    videoname = "test"
    for j, (box, prob, label) in enumerate(zip(
        final_boxes, final_probs, final_labels)):
      box[2] -= box[0]
      box[3] -= box[1]  # produce x,y,w,h output

      cat_id = int(label)
      cat_name = targetid2class[cat_id]

      # encode mask
      rle = None
      if self.args.add_mask:
        final_mask = final_masks[j] # [14, 14]
        rle = cocomask.encode(np.array(
            final_mask[:, :, None], order="F"))[0]
        rle["counts"] = rle["counts"].decode("ascii")

      res = {
          "category_id": int(cat_id),
          "cat_name": cat_name,  # [0-80]
          "score": float(round(prob, 7)),
          #"bbox": list(map(lambda x: float(round(x, 2)), box)),
          "bbox": [float(round(x, 2)) for x in box],
          "segmentation": rle,
      }

      pred.append(res)
    if self.args.out_dir is not None:  
      if self.args.use_my_naming:
        predfile = os.path.join(self.video_out_path,
            "%s_F_%08d.json" % (os.path.splitext(videoname)[0], cur_frame))
      else:
        predfile = os.path.join(self.video_out_path, "%d.json" % (cur_frame))

      with open(predfile, "w") as f:
        json.dump(pred, f)


  
  def run_inference(self, image, cur_frame):
    tmp = []
    resized_image = resizeImage(image, self.args.short_edge_size, self.args.max_size)

    scale = (resized_image.shape[0] * 1.0 / image.shape[0] + resized_image.shape[1] * 1.0 / image.shape[1]) / 2.0
    feed_dict = self.model.get_feed_dict_forward(resized_image)


    if self.args.get_box_feat:
      sess_input = [self.model.final_boxes, self.model.final_labels,
                    self.model.final_probs, self.model.fpn_box_feat]
      final_boxes, final_labels, final_probs, box_feats = self.sess.run(
          sess_input, feed_dict=feed_dict)
      assert len(box_feats) == len(final_boxes)
      # save the box feature first

      featfile = os.path.join(self.feat_out_path, "%d.npy" % (cur_frame))
      np.save(featfile, box_feats)
    
    elif self.args.get_tracking:

      if self.args.add_mask:
        sess_input = [self.model.final_boxes, self.model.final_labels,
                      self.model.final_probs, self.model.fpn_box_feat,
                      self.model.final_masks]
        final_boxes, final_labels, final_probs, box_feats, final_masks = \
            self.sess.run(sess_input, feed_dict=feed_dict)
      else:
        sess_input = [self.model.final_boxes, self.model.final_labels,
                      self.model.final_probs, self.model.fpn_box_feat]
        final_boxes, final_labels, final_probs, box_feats = self.sess.run(
            sess_input, feed_dict=feed_dict)
        if self.args.is_efficientdet:
          # the output here is 1 - num_partial_classes
          if self.args.use_partial_classes:
            for i in range(len(final_labels)):
              final_labels[i] = coco_obj_class_to_id[
                  self.args.partial_classes[final_labels[i] - 1]]
          else:
            # 1-90 to 1-80
            for i in range(len(final_labels)):
              final_labels[i] = \
                  coco_obj_class_to_id[coco_id_mapping[final_labels[i]]]

      assert len(box_feats) == len(final_boxes)
      ## final_boxes, final_labels, final_probs, box_feats, current frame detection results
      t1 = time.time()
      tmp1 = sys.getsizeof([self.tracker_dict for i in range(20)])
      print("data shape:", box_feats.shape, flush=True)
      print("data size:", tmp1, flush=True)
      write_state(self.d, self.args.state_store, self.args.app_name, {"boxes": final_boxes, "labels": final_labels, "probs": final_probs, "tracker": self.tracker_dict, "current": cur_frame})
      t2 = time.time()
      print("write state time cost:", t2-t1, flush=True)
      time1 = time.time()
      final_boxes, final_labels, final_probs, self.tracker_dict, cur_frame = read_state(self.d, self.args.state_store, self.args.app_name)
      time2 = time.time()
      print("read state time cost:", time2-time1, flush=True)
      for tracking_obj in self.tracking_objs:
        target_tracking_obs = [tracking_obj]
        
        detections = create_obj_infos(
            cur_frame, final_boxes, final_probs, final_labels, box_feats,
            targetid2class, target_tracking_obs, self.args.min_confidence,
            self.args.min_detection_height, scale,
            is_coco_model=self.args.is_coco_model,              
            coco_to_actev_mapping=coco_obj_to_actev_obj)
          # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.args.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        self.tracker_dict[tracking_obj].predict()
        self.tracker_dict[tracking_obj].update(detections)
        # Store results
        for track in self.tracker_dict[tracking_obj].tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
            if (not track.is_confirmed()) and track.time_since_update == 0:
              bbox = track.to_tlwh()
              if track.track_id not in self.tmp_tracking_results_dict[tracking_obj]:
                self.tmp_tracking_results_dict[tracking_obj][track.track_id] = [[cur_frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]]]
              else:
                self.tmp_tracking_results_dict[tracking_obj][track.track_id].append([cur_frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            continue
          bbox = track.to_tlwh()
          if track.track_id in self.tmp_tracking_results_dict[tracking_obj]:
            pred_list = self.tmp_tracking_results_dict[tracking_obj][track.track_id]
            for pred_data in pred_list:
              self.tracking_results_dict[tracking_obj].append(pred_data)
            self.tmp_tracking_results_dict[tracking_obj].pop(track.track_id, None)
          self.tracking_results_dict[tracking_obj].append([cur_frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    else:
      if self.args.add_mask:
        sess_input = [self.model.final_boxes, self.model.final_labels,
                      self.model.final_probs, self.model.final_masks]
        final_boxes, final_labels, final_probs, final_masks = self.sess.run(sess_input, feed_dict=feed_dict)
      else:
        sess_input = [self.model.final_boxes, self.model.final_labels,
                      self.model.final_probs]
        final_boxes, final_labels, final_probs = self.sess.run(
            sess_input, feed_dict=feed_dict)


    # ---------------- get the json outputs for object detection

    # scale back the box to original image size
    final_boxes = final_boxes / scale

    if self.args.add_mask:
      final_masks = [fill_full_mask(box, mask, image.shape[:2])
                      for box, mask in zip(final_boxes, final_masks)]

    if self.args.out_dir is not None:  # not saving box json to save time
      self.save_as_json(final_boxes, final_probs, final_labels, cur_frame)
      
  def reset_tracker_state(self):
    self.tracker_dict = {}
    self.tmp_tracking_results_dict = {}
    self.tracking_results_dict = {}
    for tracking_obj in self.tracking_objs:
      metric = metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.args.max_cosine_distance, self.args.nn_budget)
      self.tracker_dict[tracking_obj] = Tracker(metric, max_iou_distance=self.args.max_iou_distance)
      self.tracking_results_dict[tracking_obj] = []
      self.tmp_tracking_results_dict[tracking_obj] = {}

    
  '''
  sess = tf.Session(config=tfconfig)
  img_path = 'VIRAT_S_000205_05_001092_001124_F_00000001.jpg'
  img = cv2.imread(img_path)
  im = img.astype("float32")
  resized_image = resizeImage(im, args.short_edge_size, args.max_size)

  scale = (resized_image.shape[0] * 1.0 / im.shape[0] + \
            resized_image.shape[1] * 1.0 / im.shape[1]) / 2.0

  feed_dict = model.get_feed_dict_forward(resized_image)

  sess_input = [model.final_boxes, model.final_labels,
                  model.final_probs, model.fpn_box_feat]

  final_boxes, final_labels, final_probs, box_feats = sess.run(
        sess_input, feed_dict=feed_dict)
  assert len(box_feats) == len(final_boxes)
  
  return model, sess
  '''
  