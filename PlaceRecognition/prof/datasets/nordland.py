import pandas as pd
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

        self.images = glob.glob(os.path.join(self.data_path, "*/*/*.png"))
        self.seasons = [(f.path, f.name) for f in os.scandir(data_path) if f.is_dir()]
        self.sections = [(f.path, f.name) for f in os.scandir(self.seasons[0][0]) if f.is_dir()]
        self.maxnum = {}
        for path, name in self.sections:
            files = natsorted(glob.glob(os.path.join(path, "*.png")))
            self.maxnum[name] = [int(os.path.splitext(os.path.basename(files[0]))[0]), int(os.path.splitext(os.path.basename(files[-1]))[0])]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anc_path = self.images[idx]
        pos_path = self.get_positive(anc_path)
        neg_path = self.get_negative(anc_path)
        anc = io.imread(anc_path)
        pos = io.imread(pos_path)
        neg = io.imread(neg_path)

        return anc, pos, neg

    def get_positive(self, path):
        season_src = path.split("/")[-3]
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0])
        ext = os.path.splitext(path.split("/")[-1])[1]

        while True:
            season_dst = random.choice(self.seasons)[1]
            number_dst = random.randrange(max(self.maxnum[section_src][0], number_src-self.window), min(number_src+self.window+1, self.maxnum[section_src][1]))
            if season_src != season_dst or number_src != number_dst:
                break

        path = path.replace(season_src, season_dst)
        path = path.replace(str(number_src)+ext, str(number_dst)+ext)

        return path

    def get_negative(self, path):
        season_src = path.split("/")[-3]
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0])
        ext = os.path.splitext(path.split("/")[-1])[1]

        while True:
            season_dst = random.choice(self.seasons)[1]
            section_dst = random.choice(self.sections)[1]
            number_dst = random.randrange(self.maxnum[section_dst][0], self.maxnum[section_dst][1]+1)
            if season_src != season_dst or section_src != section_dst or abs(number_src - number_dst) > self.window:
                break

        path = path.replace(season_src, season_dst)
        path = path.replace(section_src, section_dst)
        path = path.replace(str(number_src)+ext, str(number_dst)+ext)

        return path
