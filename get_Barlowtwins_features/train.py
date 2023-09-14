import json
import torch
import argparse
import os
import sys
from dataset import load_image, Transform, OV
from utils import get_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='512-512-512', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=1000, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--transform', default=Transform())
parser.add_argument('--loader', default=load_image)

parser.add_argument('--store_dir',
                    default='./patches.hdf5')
parser.add_argument('--checkpoint-dir',
                    default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--wd', default=1e-5)
parser.add_argument('--T_max', default=200)

def train():
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)
    model = get_model(args)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)

    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cuda:1')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0

    data = OV(store_dir=args.store_dir, transform=args.transform)
    data_loader = DataLoader(data, args.batch_size, shuffle=True, num_workers=8, drop_last=False)
    scaler = torch.cuda.amp.GradScaler()
    loss_ = 99999
    loss_epoch = 99999

    for epoch in range(start_epoch, args.epochs):
        loss_epoch_ = 0
        for step, ((x1, x2), _) in enumerate(data_loader, start=epoch * len(data_loader)):
            x1 = x1.cuda()
            x2 = x2.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x1, x2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_epoch_ = loss_epoch_ + loss
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             lr=optimizer.state_dict()['param_groups'][0]['lr'],
                             loss=loss.item())
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        if loss_epoch_ < loss_epoch:
            state = dict(epoch=epoch, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint_minloss.pth')
            loss_epoch = loss_epoch_
        print(f'Epoch{epoch} loss: {loss_epoch_}')
        print(f'Current{epoch} min loss: {loss_epoch}')
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
        scheduler.step()

    torch.save(model.state_dict(),
               args.checkpoint_dir / 'Barlow-Twins.pth')

if __name__ == '__main__':
    train()
