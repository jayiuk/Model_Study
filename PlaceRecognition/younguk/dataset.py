import torch
from natsort import natsorted
import glob
import random
import os
import numpy as np
from skimage import io
from torchvision import transforms
from PIL import Image
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
    self.transform = transforms.Compose([
            transforms.Resize((config.img_h, config.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    for idx, img in enumerate(self.img_list):
        temp = []
        for i in range(max(0, idx-self.window), min(idx+self.window+1, len(self.img_list))):
            if (i != idx) and (img.split("/")[-2] == self.img_list[i].split("/")[-2]):
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
    anc = Image.open(anc_path).convert('RGB')
    pos = Image.open(pos_path).convert('RGB')
    neg = Image.open(neg_path).convert('RGB')

    if self.transform:
        anc = self.transform(anc)
        pos = self.transform(pos)
        neg = self.transform(neg)
    return anc, pos, neg