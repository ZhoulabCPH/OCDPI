import pandas as pd
import torch
import numpy as np
from ..model import GDL
from dataset import OV
from util import CoxLoss, collate, make_batch_
from sklearn.preprocessing import scale
from torch_geometric.utils import to_dense_adj
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calculate_gradient_IG():
    model = GDL().cuda()
    ckpt = torch.load(
        f'../checkpoints/checkpoint_GDL.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    model.eval()
    workspace = pd.read_csv(f'../datasets/clinical_data/TCGA_discovery.csv')
    nodes_path = f'../datasets/graphs/TCGA_discovery_nodes.npy'
    edges_path = f'../datasets/graphs/TCGA_discovery_edges.npy'
    patches_path = f'../datasets/graphs/TCGA_discovery_patches_name.npy'
    patches = np.load(patches_path, allow_pickle=True)
    data_TCGA = OV(workspace, nodes_path, edges_path)
    data_loader_TCGA = DataLoader(data_TCGA, 8, shuffle=False, num_workers=0, drop_last=False,
                                  collate_fn=collate)
    workspace.index = workspace.iloc[:, 0]
    ig = IntegratedGradients(model)
    for step, graphs in enumerate(data_loader_TCGA):
        batch_graphs, batch_OSs, batch_OS_times, batch_slides_name = make_batch_(graphs)
        baseline_nodes = torch.zeros(batch_graphs[0].shape[0], batch_graphs[0].shape[1], batch_graphs[0].shape[2])
        addition_args = (batch_graphs[1], batch_graphs[2].cuda())
        attributions = ig.attribute(batch_graphs[0].requires_grad_(True).cuda(),
                                    baseline_nodes.requires_grad_(True).cuda(), additional_forward_args=addition_args)
        attributions = torch.sum(attributions, dim=2)[0]
        grads = list(attributions.detach().cpu().numpy())
        indexes = []
        for i in range(len(workspace)):
            sample = workspace.iloc[i]['SLIDES']
            if sample in batch_slides_name:
                indexes.append(i)
        patches_batch_ = patches[indexes]
        patches_batch = list(patches_batch_)[0]
        data = pd.DataFrame()
        data['patches'] = patches_batch
        data['grads'] = grads
        data.to_csv(f'../datasets/gradients/TCGA_discovery_patches_graident.csv',
                    mode='a', header=False, index=False)


