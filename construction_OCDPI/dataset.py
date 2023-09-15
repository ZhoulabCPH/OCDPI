import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset


class OC(Dataset):
    def __init__(self, workspace, nodes, nodes_index
                 ):
        super(OC, self).__init__()

        self.workspace = workspace
        self.graphs = list(self.workspace.index)
        self.nodes = np.load(nodes, allow_pickle=True)
        self.nodes_index = np.load(nodes_index, allow_pickle=True)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        label = self.workspace.iloc[item]['label']
        risk_score = self.workspace.iloc[item]['risk_score']
        os = self.workspace.iloc[item]['OS']
        state = self.workspace.iloc[item]['OS_STATE']
        id = self.workspace.iloc[item]['slides']
        node = self.nodes[item]
        node = torch.Tensor(node)
        edge_index = self.nodes_index[item]
        edge_index = sp.coo_matrix(edge_index)
        indices = torch.from_numpy(np.vstack((edge_index.row, edge_index.col)).astype(np.int64))
        values = torch.from_numpy(edge_index.data)

        edge_index = {'indices': indices.cuda(), 'values': values.cuda()}
        return {'node': node, 'edge_index': edge_index, 'label': label, 'risk_score': risk_score, 'state': state, 'os': os, 'id': id}



