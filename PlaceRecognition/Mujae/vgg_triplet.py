import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from setup import config
from netvlad import NetVLAD, EmbedNet, TripletNet
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

# Data loading
train_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.train_data_path)
test_dataset = getattr(importlib.import_module('dataset'), 'CustomDataset')(config, config.test_data_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model setup
base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove last layer
model = EmbedNet(base_model, NetVLAD(num_clusters=64, dim=512, alpha=100)).cuda()
triplet_net = TripletNet(model).cuda()

criterion = nn.TripletMarginLoss(margin=0.1)
optimizer = optim.Adam(triplet_net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train(epoch):
    triplet_net.train()
    total_loss = 0
    for i, (images, pos, neg) in enumerate(train_loader):
        images, pos, neg = images.cuda(), pos.cuda(), neg.cuda()
        optimizer.zero_grad()
        features, pos_fit, neg_fit = triplet_net(images, pos, neg)
        loss = criterion(features, pos_fit, neg_fit)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'Train Epoch: {epoch} [{i * len(images)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    return total_loss / len(train_loader)
#아직 수정중
def validate():
    triplet_net.eval()
    test_loss = 0
    total_correct_img_neg = 0
    total_correct_img_pos = 0
    total_correct_pos_neg = 0
    num_samples = 0

    with torch.no_grad():
        for i, (images, pos, neg) in enumerate(test_loader):
            images, pos, neg = images.cuda(), pos.cuda(), neg.cuda()
            features, pos_fit, neg_fit = triplet_net(images, pos, neg)
            loss = criterion(features, pos_fit, neg_fit)
            test_loss += loss.item()

            # Compute distances
            dist_img_pos = torch.norm(features - pos_fit, dim=1)
            dist_img_neg = torch.norm(features - neg_fit, dim=1)
            dist_pos_neg = torch.norm(pos_fit - neg_fit, dim=1)

            # 내가 생각한 평가 방식 -> 앵커이미지와 postive이미지의 거리가 negative보다 작으면 TP+=1, 반대면 
            total_correct_img_neg += (dist_img_pos < dist_img_neg).sum().item()
            total_correct_img_pos += (dist_img_neg > dist_img_pos).sum().item()
            total_correct_pos_neg += (dist_pos_neg > dist_img_pos).sum().item()

            num_samples += images.size(0)

    avg_loss = test_loss / num_samples
    recall_img_neg = total_correct_img_neg / num_samples
    recall_img_pos = total_correct_img_pos / num_samples
    recall_pos_neg = total_correct_pos_neg / num_samples

    print(f'Validation Loss: {avg_loss:.4f}')
    print(f'Recall (Image-Negative): {recall_img_neg:.4f}')
    print(f'Recall (Image-Positive): {recall_img_pos:.4f}')
    print(f'Recall (Positive-Negative): {recall_pos_neg:.4f}')

for epoch in range(1, 31):  # total_epoch is defined in your config file
    train_loss = train(epoch)
    print(f'Epoch {epoch}: Train loss {train_loss:.4f}')
    if epoch % 1 == 0:  # Validate every 5 epochs
        torch.save(triplet_net.state_dict(), f'vgg16_64_512_checkpoint_epoch_{epoch}.pth')


