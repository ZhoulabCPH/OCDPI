import numpy as np
import torch
import scipy.sparse as sp
import feather
from torch.utils.data import Dataset


class OV(Dataset):
    def __init__(self, workspace, nodes, edges
                 ):
        super(OV, self).__init__()

        self.workspace = workspace

        self.nodes = np.load(nodes, allow_pickle=True)
        self.edges = np.load(edges, allow_pickle=True)

    def __len__(self):
        return len(self.workspace)

    def __getitem__(self, item):

        OS_time = self.workspace.iloc[item]['OS.time']
        OS = self.workspace.iloc[item]['OS']
        slide_name = self.workspace.iloc[item]['SLIDES']
        node = self.nodes[item]
        node = torch.Tensor(node)
        edge = self.edges[item]
        edge = sp.coo_matrix(edge)
        indices = torch.from_numpy(np.vstack((edge.row, edge.col)).astype(np.int64))
        values = torch.from_numpy(edge.data)
        edge = {'indices': indices.cuda(), 'values': values.cuda()}
        return {'node': node, 'edge': edge, 'OS': OS, 'OS_time': OS_time, 'slide_name': slide_name}
