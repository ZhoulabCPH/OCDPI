import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import scipy.sparse as sp
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv, dense_mincut_pool
from torch import nn
from torch_geometric.nn import GATConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj


class GDL(torch.nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
        self.conv1 = GATConv(512, 256, heads=2, concat=False, return_attention_weights=True)
        self.conv2 = GATConv(256, 128, heads=2, concat=False, return_attention_weights=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.relu = nn.ReLU()
        self.mlp1 = nn.Linear(256, 100)
        self.mlp2 = nn.Linear(128, 50)
        self.classtoken = nn.Parameter(torch.randn(1, 1, 128))
        self.transformer = nn.Transformer(d_model=128, nhead=2, num_encoder_layers=2,
                                          num_decoder_layers=2, dim_feedforward=128,
                                          dropout=0.2, batch_first=True)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.mlp1.weight)
        nn.init.xavier_normal_(self.mlp2.weight)

    def get_edge_index(self,edge_index):
        edge_index_ = sp.coo_matrix(edge_index.cpu().detach().numpy())
        values = edge_index_.data
        indices = np.vstack((edge_index_.row, edge_index_.col))
        indices = torch.LongTensor(indices).cuda()
        values = torch.FloatTensor(values).cuda()
        return indices, values

    def forward(self, x):
        x_nodes = x['batch_nodes'].cuda()
        x_edge_index = x['batch_edges_index']
        x_mask = x['mask'].cuda()
        x_mask_ = x_mask.new_zeros(x_nodes.shape[0], max(x_mask))
        for i in range(x_mask_.shape[0]):
            x_mask_[i, 0:x_mask[i]] = True
            x_mask_[i, x_mask[i]:] = False
        x_mask_ = x_mask_.bool()

        #GATconv1
        x_edge_index_ = [i for i in range(len(x_mask))]
        x_nodes_ = x_nodes.new_zeros(x_nodes.shape[0], x_nodes.shape[1], 256)
        for graph in range(x_nodes.shape[0]):
            (x_nodes_[graph][0:x_mask[graph], :], x_edge_index_[graph]) = self.conv1(x_nodes[graph][0:x_mask[graph]],
                                                                x_edge_index[graph]['indices'],
                                                                return_attention_weights=True
                                                                )
        x_nodes_af_conv1 = self.relu(x_nodes_[:, :, 0:256])

        edge_index = x_nodes.new_zeros(x_nodes_.shape[0], max(x_mask), max(x_mask)).type(torch.float32)
        for i in range(len(x_edge_index_)):
            (indice, value) = x_edge_index_[i]
            value = torch.mean(value, dim=1)
            temp = to_dense_adj(indice, edge_attr=value)
            edge_index[i][0:temp.shape[1], 0:temp.shape[1]] = temp

        # mincutPool1
        s1 = self.mlp1(x_nodes_af_conv1[:, :, 0:256])
        x_pool1 = x_nodes_af_conv1[:, :, 0:256]
        x_pool1, edge_index, mc1, o1 = dense_mincut_pool(x_pool1, edge_index, s1, x_mask_)
        x_pool1 = self.bn2(x_pool1).squeeze(dim=1)

        #GATconv2
        x_pool2 = x_pool1.new_zeros(x_pool1.shape[0], x_pool1.shape[1], 128)
        edge_index_ = x_nodes.new_zeros(x_nodes_.shape[0], 100, 100).type(torch.float32)
        for graph in range(x_nodes_.shape[0]):
            indices, values = self.get_edge_index(edge_index[graph])
            (x_pool2[graph][:, :], temp) = self.conv2(x_pool1[graph], indices, return_attention_weights=True)
            (indice, value) = temp
            value = torch.mean(value, dim=1)
            temp = to_dense_adj(indice, edge_attr=value)
            edge_index_[graph] = temp

        #mincutPool2
        x_pool2 = self.bn2(x_pool2)
        x_pool2 = self.relu(x_pool2)
        s2 = self.mlp2(x_pool2)
        x_pool2, edge_index_, mc2, o2 = dense_mincut_pool(x_pool2, edge_index_, s2)

        #transformer
        classtoken = self.classtoken.repeat(x_pool2.size(0), 1, 1)
        x_pool2 = torch.cat([classtoken, x_pool2], dim=1)
        x_pool2 = self.transformer(x_pool2, x_pool2)
        x_af_tranformer = self.bn3(x_pool2[:, 0, :])

        #output
        pred = self.fc(x_af_tranformer).squeeze(dim=1)
        return pred, mc1 + mc2, o1 + o2