from pycocotools.mask import encode
import redis 
import struct 
import numpy as np
import pickle
import json
import sys
import grpc
import time

from dapr.clients import DaprClient
from dapr.clients.grpc._request import TransactionalStateOperation, TransactionOperationType
from dapr.clients.grpc._state import StateItem

import hashlib


def cur_key(frame):
  tt = time.time()
  sha1obj = hashlib.sha1()
  sha1obj.update(frame)
  hash = sha1obj.hexdigest()
  ee = time.time()
  print(ee-tt)
  return hash



KEY = ["boxes", "labels", "probs", "tracker", "current"]

def write_state(d, storeName, tag, state_dict):

  # with DaprClient() as d:
  value_keys = state_dict.keys()
  state = []
  for k in value_keys:
    state.append(StateItem(key=tag+k, value=pickle.dumps(state_dict[k])))
  resp = d.save_bulk_state(store_name=storeName, states=state)
    
    
    # value = pickle.dumps(value)
    # d.save_state(store_name=storeName, key=key, value=value)
    # print(f"State store has successfully saved key {key}")
  # except grpc.RpcError as err:
    # print(f"Cannot save key {key}. ErrorCode={err.code()}")

  # packed_object = pickle.dumps(state)
  # packed_object = json.dumps(state)
  # print(sys.getsizeof(state))
  # r.set(name, packed_object)
  # assert len(state) == len(unpacked_object)
  # print(unpacked_object[-1].shape)
  return resp



def read_state(d, storeName, tag):
  # with DaprClient() as d:
  items = d.get_bulk_state(store_name=storeName, keys=[tag+KEY[i] for i in range(len(KEY))], states_metadata={"metakey": "metavalue"}).items
  tmp = []
  print(len(items), flush=True)
  for i in range(len(items)):
    tmp.append(pickle.loads(items[i].data))
  # print(type(tmp[0]), type(tmp[1]), type(tmp[2]), type(tmp[3]))
  [boxes, labels, probs, tracker, cur_frame] = tmp
  return boxes, labels, probs, tracker, cur_frame

def delete_state(d, storeName, key):
  with DaprClient() as d:
    d.delete_state(store_name=storeName, key=key, state_metadata={"metakey": "metavalue"})
  
def initialize_state():
  d = DaprClient()
  d.wait(5)
  # r.set()
  return d