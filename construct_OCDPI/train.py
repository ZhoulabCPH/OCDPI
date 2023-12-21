import random
import pandas as pd
import numpy as np
import torch
import argparse
import os
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from warmup_scheduler import GradualWarmupScheduler
from pathlib import Path
from model import GDL
from utils.dataset import OV
from utils.util import make_batch, collate, CoxLoss, weight_init

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description='Graph-based deep learning')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--workspace', default='../dataset/clinical_data/TCGA_discovery_cohort.csv')
    parser.add_argument('--nodes', default='../dataset/graphs/TCGA_discovery_nodes.npy')
    parser.add_argument('--edges', default='../dataset/graphs/TCGA_discovery_edges.npy')
    parser.add_argument('--checkpoint-dir',
                        default='../checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=9e-5)
    parser.add_argument('--wd', default=5e-5)

    args = parser.parse_args()
    return args


def train():
    args = get_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    workspace = pd.read_csv(args.workspace)
    model = GDL().cuda()
    model.apply(weight_init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=200)
    print(f'Training beginning!')

    data_TCGA = OV(workspace, args.nodes, args.edges)
    data_TCGA_loader = DataLoader(data_TCGA, args.batch_size, shuffle=True, num_workers=0, drop_last=False, collate_fn=collate)

    for epoch in range(args.epochs):
        seed = 0
        setup_seed(seed)
        model.train()
        epoch_l_total = []
        epoch_l_cl = []
        epoch_l_pool = []
        for step, graphs in enumerate(data_TCGA_loader):
            batch_graphs, batch_OSs, batch_OS_times, batch_slides_name = make_batch(graphs)
            optimizer.zero_grad()
            OCDPI, l_pool = model.forward(batch_graphs)
            l_cl = CoxLoss(batch_OSs.cuda(), batch_OS_times.cuda(), OCDPI)

            l_total = l_cl

            l_total.backward()
            optimizer.step()

            epoch_l_total.append(l_total.item())
            epoch_l_cl.append(l_cl.item())
            epoch_l_pool.append(l_pool.item())

        epoch_l_total = np.sum(epoch_l_total)
        epoch_l_cl = np.sum(epoch_l_cl)
        epoch_l_pool = np.sum(epoch_l_pool)

        print(f'Epoch {epoch} total loss: {epoch_l_total}, CoxLoss: {epoch_l_cl}, PoolingLoss: {epoch_l_pool}')

        if epoch+1 == args.epochs:
            state = dict(epoch=epoch, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / f'checkpoint_GDL.pth')
        scheduler.step()


if __name__ == '__main__':
    train()