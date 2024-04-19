import os
import importlib
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from setup import config
from utils.util_model import EmbedNet, TripletNet
import utils.util_path as PATH

from backbones import get_backbone
from models import get_model


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = config.gpu_ids[rank]
    torch.cuda.set_device(device_id)

    train_dataset = getattr(importlib.import_module('datasets.'+config.data), 'CustomDataset')(config, config.train_data_path)
    test_dataset = getattr(importlib.import_module('datasets.'+config.data), 'CustomDataset')(config, config.test_data_path)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, int(config.batch_size), num_workers=int(config.num_workers), sampler=train_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, int(config.batch_size), num_workers=int(config.num_workers), sampler=test_sampler, pin_memory=True)

    backbone = get_backbone(config.backbone)
    model = get_model(config.model)
    embed_net = EmbedNet(backbone, model)
    triplet_net = DDP(TripletNet(embed_net).to(device_id), device_ids=[device_id])

    criterion = torch.nn.TripletMarginLoss(margin=0.1)
    optimizer = torch.optim.Adam(triplet_net.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(PATH.CHECKPOINT, exist_ok=True)
    
    def train():
        triplet_net.train()
        total_loss = 0
        for i, (anc, pos, neg) in enumerate(train_loader):
            anc, pos, neg = anc.to(device_id), pos.to(device_id), neg.to(device_id)
            optimizer.zero_grad()
            anc_feat, pos_feat, neg_feat = triplet_net(anc, pos, neg)
            loss = criterion(anc_feat, pos_feat, neg_feat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f'Training... [{i * len(anc)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        return total_loss / len(train_loader)

    def validate():
        triplet_net.eval()
        total_loss = 0
        total_correct_img_neg = 0
        total_correct_img_pos = 0
        total_correct_pos_neg = 0
        num_samples = 0

        with torch.no_grad():
            for i, (anc, pos, neg) in enumerate(test_loader):
                anc, pos, neg = anc.to(device_id), pos.to(device_id), neg.to(device_id)
                anc_feat, pos_feat, net_feat = triplet_net(anc, pos, neg)
                loss = criterion(anc_feat, pos_feat, net_feat)
                total_loss += loss.item()

                dist_img_pos = nn.PairwiseDistance(anc_feat, pos_feat)
                dist_img_neg = nn.PairwiseDistance(anc_feat, net_feat)
                dist_pos_neg = nn.PairwiseDistance(pos_feat, net_feat)

                total_correct_img_neg += (dist_img_pos < dist_img_neg).sum().item()
                total_correct_img_pos += (dist_img_neg > dist_img_pos).sum().item()
                total_correct_pos_neg += (dist_pos_neg > dist_img_pos).sum().item()

                num_samples += anc.size(0)

        avg_loss = total_loss / num_samples
        recall_img_neg = total_correct_img_neg / num_samples
        recall_img_pos = total_correct_img_pos / num_samples
        recall_pos_neg = total_correct_pos_neg / num_samples

        print(f'Validation Loss: {avg_loss:.4f}')
        print(f'Recall (Image-Negative): {recall_img_neg:.4f}')
        print(f'Recall (Image-Positive): {recall_img_pos:.4f}')
        print(f'Recall (Positive-Negative): {recall_pos_neg:.4f}')

    for epoch in range(1, config.total_epoch):
        print(f'Epoch {epoch} Started')
        train_sampler.set_epoch(epoch)
        train_loss = train()
        print(f'Epoch {epoch} Finished: Train loss {train_loss:.4f}')
        validate()
        torch.save(triplet_net.state_dict(), os.path.join(PATH.CHECKPOINT, f'{config.backbone}_{config.model}_checkpoint_e{epoch}.pth'))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
