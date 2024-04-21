import os
import importlib
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from setup import config
from utils.util_model import EmbedNet, TripletNet
import utils.util_path as PATH

from backbones import get_backbone
from models import get_model

writer = SummaryWriter()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = config.gpu_ids[rank]
    torch.cuda.set_device(device_id)

    train_dataset = getattr(importlib.import_module('datasets.'+config.data), 'CustomDataset')(config, config.train_data_path)
    test_dataset = getattr(importlib.import_module('datasets.'+config.data), 'CustomDataset')(config, config.test_data_path)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, num_workers=config.num_workers, sampler=train_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, num_workers=config.num_workers, sampler=test_sampler, pin_memory=True)

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
        return total_loss / len(train_loader)

    def validate():
        triplet_net.eval()
        total_loss = 0
        total_dist_pos = 0
        total_dist_neg = 0
        num_samples = 0

        with torch.no_grad():
            for i, (anc, pos, neg) in enumerate(test_loader):
                anc, pos, neg = anc.to(device_id), pos.to(device_id), neg.to(device_id)
                anc_feat, pos_feat, neg_feat = triplet_net(anc, pos, neg)
                loss = criterion(anc_feat, pos_feat, neg_feat)
                total_loss += loss.item()

                total_dist_pos += nn.PairwiseDistance(anc_feat, pos_feat)
                total_dist_neg += nn.PairwiseDistance(anc_feat, neg_feat)

                num_samples += anc.size(0)

        avg_loss = total_loss / num_samples
        avg_dist_pos = total_dist_pos / num_samples
        avg_dist_neg = total_dist_neg / num_samples

        return avg_loss, avg_dist_pos, avg_dist_neg

    for epoch in range(1, config.total_epoch):
        print(f'Epoch {epoch} Started')
        train_sampler.set_epoch(epoch)
        train_loss = train()
        writer.add_scalar('train_loss', train_loss)
        print(f'Epoch {epoch} Finished: Train loss {train_loss:.4f}')

        print(f'Validation Started')
        avg_loss, avg_dist_pos, avg_dist_neg = validate()
        writer.add_scalar('avg loss', avg_loss)
        print(f'Validation Finished: Validation loss {avg_loss:.4f}')
        print(f'Average distance with positive sample: {avg_dist_pos:.4f}')
        print(f'Average distance with negative sample: {avg_dist_neg:.4f}')

        torch.save(triplet_net.state_dict(), os.path.join(PATH.CHECKPOINT, f'{config.backbone}_{config.model}_checkpoint_e{epoch}.pth'))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
