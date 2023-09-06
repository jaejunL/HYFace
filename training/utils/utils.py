import os
import json
import glob
import argparse
from shutil import copyfile

import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def copy_DirStructure_and_Files(base_dir, include_dir, exclude_dir, include_ext, write_dir):
    # searching folder & code for saving
    filepath = []
    for dirpath, dirnames, filenames in os.walk(base_dir, topdown=True):
        if all(dir_ in dirpath for dir_ in include_dir) and not any(dir in dirpath for dir in exclude_dir):
            filtered_files = [name for name in filenames if os.path.splitext(name)[-1] in include_ext]
            filepath.append({'dir':dirpath, 'files':filtered_files})

    # make folder structure and copying code
    num_strip = len(os.getcwd())
    for path in filepath:
        dirname = path['dir'][num_strip+1:]
        dirpath2save = os.path.join(write_dir, 'codes', dirname)
        os.makedirs(dirpath2save, exist_ok=True)
        for filename in path['files']:
            file2copy = os.path.join(path['dir'], filename)
            filepath2save = os.path.join(dirpath2save, filename)
            copyfile(file2copy, filepath2save)

def load_checkpoint_hifigan(checkpoint_path, model, token):
  assert os.path.isfile(checkpoint_path) or os.path.isfile(checkpoint_path)
  try:
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    if "UNIVERSAL_V1" in checkpoint_path:
      saved_state_dict = checkpoint_dict[token]
    elif "16kHz" in checkpoint_path:
      for key in list(checkpoint_dict.keys()):
          if 'resblock' not in key:
              checkpoint_dict[key.replace('.conv.', '.')] = checkpoint_dict[key]
              checkpoint_dict.pop(key)      
      saved_state_dict = checkpoint_dict
  except:
    print("Load HiFi_GAN pretrained model is broken")
    stop  
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for i, (k, v) in enumerate(state_dict.items()):
    try:
      new_state_dict[k] = saved_state_dict[k] 
    except:
      print("Model {}, {} is not in the checkpoint".format(model, k))
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  return model

def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  # learning_rate = checkpoint_dict['learning_rate']
  learning_rate = checkpoint_dict['cntr_weight'] if 'cntr_weight' in checkpoint_dict.keys() else checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      print("Model {}, {} is not in the checkpoint".format(model, k))
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict, strict=False)
  else:
    model.load_state_dict(new_state_dict, strict=False)
  print("Loaded checkpoint '{}' (Epoch {})" .format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration
  
def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

def load_ecapa_checkpoint(checkpoint_path, models):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path)
    if hasattr(models, 'module'):
        self_state = models.module.state_dict()
    else:
        self_state = models.state_dict()    
    print("######### Pretrained model loading - ECAPA #########")
    for name, param in checkpoint_dict.items():
        origname = name
        if 'speaker_encoder' in name:
            name = name.replace('speaker_encoder.', '')        
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                print("%s is not in the model."%origname)
                continue
        if self_state[name].size() != checkpoint_dict[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), checkpoint_dict[origname].size()))
            continue
        self_state[name].copy_(param)
    print("#########*********************************#########\n")
    return models

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  return hparams


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
