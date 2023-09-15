import argparse
from pathlib import Path
from dataset import load_image, Transform


def get_args():
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
    args = parser.parse_args()
    return args