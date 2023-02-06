from ast import arg
import os
import queue
import argparse
import cv2
import time
import base64
import json
import hashlib
import multiprocessing
import requests

from multiprocessing import Queue, Pool
from kubernetes import client, config


from dapr.clients import DaprClient
from dapr.clients.grpc._request import TransactionalStateOperation, TransactionOperationType
from dapr.clients.grpc._state import StateItem


def get_args():
  parser = argparse.ArgumentParser()

  # --------------- RTMP server config
  parser.add_argument("--rtmp_server", default=None,
                      help="specify the rtmp server to recieve video frames")
  # --------------- dapr pubsub name
  parser.add_argument("--pubsub_name", default=None,
                      help="specify the pubsub name to publish video frames")
  parser.add_argument("--app_name", default=None,
                      help="sepcify the application name (pods mata.name)")
  args = parser.parse_args()
  return args


def check_args(args):
  assert args.rtmp_server is not None
  # assert args.pubsub_name is not None
  # assert args.app_name is not None


def open_rtmp_server(args):
  rtmp_url = 'rtmp://{}:1935/live/'.format(args.rtmp_server)
  cap = cv2.VideoCapture(rtmp_url)
  if not cap.isOpened():
    raise Exception("cannot open %s" % rtmp_url)
  return cap


def compute_hash(msg):
  m = hashlib.md5()
  m.update(msg)
  hash = m.hexdigest()
  return hash

def consume_frame(frame_queue, session, addr):
  while True:
    if not frame_queue.empty():
      t1 = time.time()
      tmp = frame_queue.get()
      data = json.dumps(tmp)
      url = addr + "/infer"
      response = session.post(url=url, json=json.dumps(tmp))
      # print(response, flush=True)
      t2 = time.time()
      print(t2-t1, ',')
      # print("consume frame {}, the curent queue length is {}".format(tmp['id']), frame_queue.qsize())

  
def sync_state(sync):
  t1 = time.time()
  session = requests.Session()
  config.load_incluster_config()
  v1 = client.CoreV1Api()
  print("Listing pods with their IPs:")
  ret = v1.list_pod_for_all_namespaces(watch=False)
  ctrl_ip = ''
  node_selector = {}
  for i in ret.items:
    if i.metadata.name == "migration_controller":
      ctrl_ip = i.status.pod_ip
      node_selector = i.spec.node_selector
  url = "http://{}:5000/sync_state".format(ctrl_ip)
  print(url)
  data = {}.update(node_selector)
  response = session.post(url=url, json=json.dumps(data))
  t2 = time.time()
  print(t2-t1)
  if response.status_code == 200:
    sync = True
    return True
  else:
    return False
  
  
def init_server(session, addr):
  url = addr+"/init"
  response = session.post(url=url)
  print(response)



def queue_frame(frame_queue, MAX_INTER_FRAMES, args):
  while True:
    try:
      vcap = open_rtmp_server(args)
      if vcap.isOpened():
        # print("rtmp server has been opened")
        break
    except Exception as e:
      print("warning, cannot open rtmp server")

  if vcap.isOpened():
      print("rtmp server has been opened")

  id = 0
 
  while vcap.isOpened():
    suc, frame = vcap.read()
    if suc:
      retval, frame = cv2.imencode('.jpg', frame)
      jpg_as_bytes = base64.b64encode(frame)
      jpg_as_str = jpg_as_bytes.decode('ascii')
      m = compute_hash(jpg_as_bytes[:10000])
      req_data = {
          'id': id % MAX_INTER_FRAMES,
          'hash': m,
          'frame': jpg_as_str
      }
      frame_queue.put(req_data)
      # print(frame_queue.qsize(), flush=True)
      id = id + 1

def delete_frame(queue_frame):
  pass



if __name__ == "__main__":
  args = get_args()
  check_args(args)
  MAX_INTER_FRAMES = 10000
  frame_queue = Queue(maxsize=MAX_INTER_FRAMES)
  init_t1 = time.time()
  addr = 'http://localhost:5000'
  session = requests.Session()
  init_server(session, addr)
  init_t2 = time.time()
  print("init time:", init_t2-init_t1)
  # sync_state()
  global sync 
  sync = False
  # pool_sync = Pool(1, sync_state, (sync))
  pool_queue = Pool(1, queue_frame, (frame_queue, MAX_INTER_FRAMES, args))
  
  # while True:
  #   if sync:
  #     delete_frame(queue_frame)
  #     break
  
  consume_frame(frame_queue, session, addr)
