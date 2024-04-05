import torch
from natsort import natsorted
import glob
import random
import os
import numpy as np
from skimage import io

class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self, config, data_path):
    self.window = config.window
    self.img_h = config.img_h
    self.img_w = config.img_w
    self.batch_size = config.batch_size
    self.seed = config.seed
    self.data_path = data_path
    self.img_list = glob.glob(os.path.join(self.data_path, "*/*/*.png"))
    self.img_list = natsorted(self.img_list)
    self.pos_list = []
    for idx, img in enumerate(self.img_list):
        img_info = img.split("/")
        temp = []
        for i in range(max(0, idx-self.window), min(idx+self.window+1, len(self.img_list))):
            if (i != idx) and (img.split("/")[1:3] == self.img_list[i].split("/")[1:3]):
                temp.append(self.img_list[i])
        self.pos_list.append(temp)
        
  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx): 
    anc_path = self.img_list[idx]
    pos_path = np.random.choice(self.pos_list[idx])
    while True:
        i = random.randrange(len(self.img_list))
        if (i != idx) and (self.img_list[i] not in self.pos_list[idx]):
            neg_path = self.img_list[i]
            break

    anc = io.imread(anc_path)
    pos = io.imread(pos_path)
    neg = io.imread(neg_path)
    return anc, pos, neg
           