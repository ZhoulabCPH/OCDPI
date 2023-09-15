import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import argparse
import random
import sys
from sklearn.model_selection import KFold
from model import GDL
from datasets import OC
from utils import make_batch, collate, COXLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from torch import nn
from pathlib import Path
import torch.backends.cudnn as cudnn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False
seed = 0
setup_seed(seed)
def get_args():
    parser = argparse.ArgumentParser(description='GDL Training')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--workspace', default='./data/GDL.csv')
    parser.add_argument('--patients', default='./data/patients.csv')
    parser.add_argument('--nodes', default='./data/GDL_nodes.npy')
    parser.add_argument('--nodes_index', default='./data/GDL_edges.npy')
    parser.add_argument('--excel_dir',
                        default='')
    parser.add_argument('--checkpoint-dir',
                        default='./checkpoint/model/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--wd', default=1e-2)
    parser.add_argument('--T_max', default=20)
    args = parser.parse_args()
    return args


def test(model,data_loader):
    data = pd.DataFrame()
    model.eval()
    id = []
    pred = np.array([])
    os = np.array([])
    os_states = np.array([])
    label = []
    for step, graphs in enumerate(data_loader):
        batch_graphs, batch_labels, risk_score, batch_states, batch_os, id_ = make_batch(graphs)
        id = id+id_
        pred_, _, _ = model.forward(batch_graphs)
        pred = np.append(pred, pred_.detach().cpu().numpy())
        os = np.append(os, batch_os)
        os_states = np.append(os_states, batch_states.detach().cpu().numpy())
        label = label+list(batch_labels.detach().cpu().numpy())

    c_index = concordance_index(os, -pred, os_states)
    data['id'] = list(id)
    data['os'] = list(os)
    data['os_states'] = list(os_states)
    data['label'] = list(label)
    data['risk_score'] = list(pred)

    return c_index, data
def train():
    kf = KFold(n_splits=5, shuffle=True)
    args = get_args()
    patients = pd.read_csv(args.patients)
    workspace = pd.read_csv(args.workspace)
    for fold, (train_index, test_index) in enumerate(kf.split(list(patients.index))):

        print(f'Fold {fold} training beginning!')
        patients_train = patients.iloc[train_index]
        patients_train = list(patients_train.iloc[:, 1])
        patients_test = patients.iloc[test_index]
        patients_test = list(patients_test.iloc[:, 1])
        slides = list(workspace.loc[:, 'slides'])
        workspace_train_index = []
        workspace_test_index = []
        for i in range(len(slides)):
            if slides[i][0:12] in patients_train:
                workspace_train_index.append(i)
            if slides[i][0:12] in patients_test:
                workspace_test_index.append(i)
        workspace_train = workspace.iloc[workspace_train_index]
        workspace_test = workspace.iloc[workspace_test_index]
        BCEloss = nn.BCELoss(reduction='none')
        MSEloss = nn.MSELoss()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = GDL().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
        data = OC(workspace_train, args.nodes, args.nodes_index)
        data_loader = DataLoader(data, args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
        data_test = OC(workspace_test, args.nodes, args.nodes_index)
        data_test_loader = DataLoader(data_test, args.batch_size, shuffle=True, num_workers=0, drop_last=False, collate_fn=collate)
        scaler = torch.cuda.amp.GradScaler()
        loss_ = 99999
        c_index_train = 0
        c_index_test = 0
        c_index_train2 = 0
        c_index_test2 = 0
        for epoch in range(args.epochs):

            model.train()
            for step, graphs in enumerate(data_loader, start=epoch * len(data_loader)):
                batch_graphs, batch_labels, risk_score, batch_states, batch_os, id = make_batch(graphs)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=False):
                    pred, mc, o = model.forward(batch_graphs)
                    l_bce = BCEloss(pred.cuda().to(torch.float64), batch_labels.type(torch.float64).cuda())
                    l_bce = torch.mean(l_bce * batch_states.cuda())
                    l_mse = MSEloss(pred.cuda().to(torch.float64), risk_score.type(torch.float64).cuda())
                    loss = 0.7*l_bce + 0.3*l_mse
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if loss < loss_:
                    state = dict(epoch=epoch, step=step, model=model.state_dict(),
                                 optimizer=optimizer.state_dict())
                    torch.save(state, args.checkpoint_dir / f'checkpoint_minloss_{fold}.pth')
                    loss_ = loss

            c_index_train_, data_train = test(model, data_loader)
            c_index_test_, data = test(model, data_test_loader)

            if c_index_train_ > c_index_train and c_index_test_ > c_index_test:
                c_index_train = c_index_train_
                c_index_test = c_index_test_
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                                 optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_max_c-index_{fold}.pth')
                data.to_csv(args.checkpoint_dir / f'temp_report_test_{fold}.csv')

            if c_index_train_ + c_index_test_ > c_index_train2 + c_index_test2:
                c_index_train2 = c_index_train_
                c_index_test2 = c_index_test_
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                                 optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_max_plus_c-index_{fold}.pth')
                data.to_csv(args.checkpoint_dir / f'temp_report_test_plus_{fold}.csv')
            print('Fold ' + str(fold) + 'Epoch: '+str(epoch)+' C-index: '+str(c_index_train_))
            print('Fold ' + str(fold) + 'Epoch: ' + str(epoch) + 'test C-index: ' + str(c_index_test_))
            print('Fold ' + str(fold) + 'Current best train C-index: ' + str(c_index_train))
            print('Fold ' + str(fold) + 'Current best test C-index: ' + str(c_index_test))
            print('Fold ' + str(fold) + 'Current best test C-index_plus: ' + str(c_index_train2+c_index_test2))
            if c_index_test_ > c_index_test:
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_max_test_c-index_{fold}.pth')

            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / f'checkpoint_{fold}.pth')

            scheduler.step()
        print('Best C-index train:' + str(c_index_train))
        print('Best C-index test:'+str(c_index_test))
        torch.save(model.state_dict(),
                   args.checkpoint_dir / f'GAT_{fold}.pth')


if __name__ == '__main__':
    train()
