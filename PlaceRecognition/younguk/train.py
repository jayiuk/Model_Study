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

#Data 부분
#prof께서 작성해주신 Code에서 살짝 수정
train_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.train_data_path)
test_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.test_data_path)

train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size)

model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
net_vlad = NetVLAD(num_clusters=12, dim=512, alpha=7.0)#에러가 나서 이 부분 파라미터 조정
embednet = EmbedNet(model, net_vlad).cuda()
triplet = TripletNet(embednet).cuda()
criterion = nn.TripletMarginLoss(margin=0.1, p=2)#가까운건 더 가깝게 먼건 더 멀게 Triplet Loss
n_epoch=0
lr = 1e-4

def vlad_train(epochs, learning_rate):
    optimizer = optim.Adam(triplet.parameters(), lr=learning_rate)  # 최적화 알고리즘 설정
    
    for epoch in range(1, epochs + 1):  # epoch는 1부터 시작
        train_loss = train_epochs(optimizer)
        print(f'Epoch {epoch}, Training loss: {train_loss:.4f}')
        sys.stdout.flush()
        torch.save(triplet.state_dict(), f'/home/student4/PR_code/checkpoint_{epoch}.pth')
    
    test_loss = test_epochs()
    print(f'Epoch {epoch}, Testing loss: {test_loss:.4f}')
    sys.stdout.flush()

def train_epochs(optimizer):
    running_loss = 0.0
    triplet.train()  # 모델을 학습 모드로 설정
    
    for i, (images, pos, neg) in enumerate(train_loader):
        images, pos, neg = images.cuda(), pos.cuda(), neg.cuda()
        optimizer.zero_grad()
        features, pos_fit, neg_fit = triplet(images, pos, neg)
        loss = criterion(features, pos_fit, neg_fit)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)  # 평균 훈련 손실 반환

def test_epochs():
    running_loss = 0.0
    triplet.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        for i, (images, pos, neg) in enumerate(test_loader):
            images, pos, neg = images.cuda(), pos.cuda(), neg.cuda()
            features, pos_fit, neg_fit = triplet(images, pos, neg)
            loss = criterion(features, pos_fit, neg_fit)
            running_loss += loss.item()
    
    return running_loss / len(test_loader)  # 평균 테스트 손실 반환

vlad_train(10, lr)
