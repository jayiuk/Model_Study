import importlib
import torch
import torch.nn as nn
import torch.optim as optim
#import numpy as np
from setup import config
from NetVlad import NetVLAD
from NetVlad import EmbedNet
from NetVlad import TripletNet
from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights
from torchvision import transforms
import sys
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type = int, default = 10)
parser.add_argument('--lr', type = float, default = 1e-4)
args = parser.parse_args()
#Data 부분
#prof께서 작성해주신 Code에서 살짝 수정
train_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.train_data_path)
test_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.test_data_path)

train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size)


vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
vgg_layers = list(vgg_model.features.children())[:-1]
model = nn.Sequential(*vgg_layers)

net_vlad = NetVLAD(num_clusters=12, dim=512, alpha=7.0)#에러가 나서 이 부분 파라미터 조정
embednet = EmbedNet(model, net_vlad).cuda()
triplet = TripletNet(embednet).cuda()
criterion = nn.TripletMarginLoss(margin=0.1, p=2)#가까운건 더 가깝게 먼건 더 멀게 Triplet Loss
n_epoch = 0
lr = args.lr

def vlad_train(epochs, learning_rate):
    optimizer = optim.Adam(triplet.parameters(), lr=learning_rate)  # 최적화 알고리즘 설정
    
    train_losses = []
    test_losses = []
    for epoch in range(1, epochs + 1):  # epoch는 1부터 시작
        train_loss = train_epochs(optimizer)
        train_losses.append(train_loss)
        print(f'Epoch {epoch}, Training loss: {train_loss:.4f}')
        sys.stdout.flush()
        torch.save(triplet.state_dict(), f'C:/capstone_model/PR_model/checkpoint_{epoch}.pth')
    
        test_loss = test_epochs()
        test_losses.append(test_loss)
        print(f'Epoch {epoch}, Testing loss: {test_loss:.4f}')
        sys.stdout.flush()
    
    return train_losses, test_losses

def train_epochs(optimizer):
    running_loss = 0.0
    triplet.train()  # 모델을 학습 모드로 설정
    progress = tqdm(enumerate(train_loader), total = len(train_loader))
    
    for i, (images, pos, neg) in progress:
        images, pos, neg = images.cuda(), pos.cuda(), neg.cuda()
        optimizer.zero_grad()
        features, pos_fit, neg_fit = triplet(images, pos, neg)
        loss = criterion(features, pos_fit, neg_fit)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        progress.set_description(f'epoch {n_epoch}, training loss : {running_loss / (i + 1):.4f}')
    
    return running_loss / len(train_loader)  # 평균 훈련 손실 반환

def test_epochs():
    running_loss = 0.0
    triplet.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        progress_test = tqdm(enumerate(test_loader), total = len(test_loader))
        for i, (images, pos, neg) in progress_test:
            images, pos, neg = images.cuda(), pos.cuda(), neg.cuda()
            features, pos_fit, neg_fit = triplet(images, pos, neg)
            loss = criterion(features, pos_fit, neg_fit)
            running_loss += loss.item()
            progress_test.set_description(f'epoch {n_epoch}, testing loss : {running_loss / (i + 1):.4f}')
    
    return running_loss / len(test_loader)  # 평균 테스트 손실 반환

<<<<<<< HEAD
train_losses, test_losses = vlad_train(args.epoch, lr)
=======

train_losses, test_losses = vlad_train(10, lr)
>>>>>>> 8387b9b43565da76ba44f340cac595e2d417dadf
print("Train Losses : ", train_losses)
print("Test Losses : ", test_losses)

