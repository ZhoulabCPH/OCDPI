from torch_geometric.nn import GATConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
import numpy as np
import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, adj_pos_embed):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        adj_pos_embed = adj_pos_embed.unsqueeze(1)
        attn[:, :, :-1, :-1] = attn[:, :, :-1, :-1] + adj_pos_embed
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, X):
        (x, adj_pos_embed) = X
        x = x + self.drop_path(self.attn(self.norm1(x), adj_pos_embed))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, adj_pos_embed)


class GraphAttentionLayer(nn.Module):
    """ Layer for progressively aggregating features of patches.
    """
    def __init__(self, in_features, out_features=None, heads=2, concat=False, return_attention_weights=True,dropout=0.,
                 return_dense=True):
        super().__init__()
        self.out_features = out_features
        self.return_dense = return_dense
        self.conv = GATConv(in_features, out_features, heads=heads, concat=concat, return_attention_weights=return_attention_weights)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_features)

    def covert_sparse_to_dense(self,x_edges, adj_matrix):
        for i in range(len(x_edges)):
            (indice, value) = x_edges[i]
            value = torch.mean(value, dim=1)
            temp = to_dense_adj(indice, edge_attr=value)
            adj_matrix[i][0:temp.shape[1], 0:temp.shape[1]] = temp
        return adj_matrix

    def reformat_edge(self, x_edges):
        for i in range(len(x_edges)):
            x_edges_= x_edges[i]
            x_edges[i] = {'indices': x_edges_[0], 'values': x_edges_[1]}
        return x_edges

    def forward(self, x_nodes, x_edges, x_masks):
        x_nodes_af_conv = x_nodes.new_zeros(x_nodes.shape[0], x_nodes.shape[1], self.out_features)
        x_edges_af_conv = [i for i in range(x_nodes.shape[0])]
        for graph in range(x_nodes.shape[0]):
            if x_masks == None:
                (x_nodes_af_conv[graph][0:, :], x_edges_af_conv[graph]) = self.conv(x_nodes[graph][0:],
                                                                                         x_edges[graph]['indices'],
                                                                                         x_edges[graph]['values'],
                                                                                         return_attention_weights=True
                                                                                   )
            else:
                (x_nodes_af_conv[graph][0:x_masks[graph], :], x_edges_af_conv[graph]) = self.conv(
                    x_nodes[graph][0:x_masks[graph]],
                    x_edges[graph]['indices'],
                    x_edges[graph]['values'],
                    return_attention_weights=True
                    )
        x_nodes_af_conv = self.relu(x_nodes_af_conv[:, :, 0:self.out_features])
        x_nodes_af_conv = self.ln(x_nodes_af_conv)
        x_nodes_af_conv = self.dropout(x_nodes_af_conv[:, :, 0:self.out_features])


        if self.return_dense:

            adj_matrix = x_nodes.new_zeros(x_nodes.shape[0], x_nodes.shape[1], x_nodes.shape[1]).type(torch.float32)
            adj_matrix = self.covert_sparse_to_dense(x_edges_af_conv, adj_matrix)
            return x_nodes_af_conv, adj_matrix
        else:
            x_edges_af_conv = self.reformat_edge(x_edges_af_conv)
            return x_nodes_af_conv, x_edges_af_conv



class MinCutPoolLayer(nn.Module):
    """
    """
    def __init__(self, in_features,pool_size=100, return_dense=False):
        super().__init__()
        self.in_features = in_features
        self.pool_size = pool_size
        self.mlp = nn.Linear(in_features, pool_size)
        self.bn = nn.BatchNorm1d(pool_size)
        self.return_dense = return_dense

    def get_edge_index(self,adj_matrix):
        adj_matrix_ = sp.coo_matrix(adj_matrix.cpu().detach().numpy())
        values = adj_matrix_.data
        indices = np.vstack((adj_matrix_.row, adj_matrix_.col))
        indices = torch.LongTensor(indices).cuda()
        values = torch.FloatTensor(values).cuda()
        return indices, values
    def convert_dense_to_sparse(self, x_edges, adj_matrix):
        for i in range(adj_matrix.shape[0]):
            x_edges_= self.get_edge_index(adj_matrix[i])
            x_edges[i] = {'indices':x_edges_[0], 'values':x_edges_[1]}
        return x_edges

    def forward(self, x_nodes, adj_matrix, x_masks):
        s = self.mlp(x_nodes[:, :, 0:self.in_features])
        x_bf_pool = x_nodes[:, :, 0:self.in_features]
        if x_masks ==None:
            x_af_pool, adj_matrix, mc, o = dense_mincut_pool(x_bf_pool, adj_matrix, s)
        else:
            x_af_pool, adj_matrix, mc, o = dense_mincut_pool(x_bf_pool, adj_matrix, s, x_masks)
        x_af_pool = self.bn(x_af_pool).squeeze(dim=1)
        if self.return_dense:
            return x_af_pool, adj_matrix, mc+o
        else:
            x_edges = [i for i in range(x_nodes.shape[0])]
            x_edges = self.convert_dense_to_sparse(x_edges, adj_matrix)
        return x_af_pool, x_edges, mc+o


class GDL(torch.nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
        self.conv1 = GraphAttentionLayer(768, 512, heads=2)
        self.conv2 = GraphAttentionLayer(512, 512, heads=2)
        self.pool1 = MinCutPoolLayer(512, pool_size=200, return_dense=False)
        self.pool2 = MinCutPoolLayer(512, pool_size=100, return_dense=True)

        self.pos_proj = nn.Sequential(nn.Linear(100, 100), nn.LayerNorm(100))
        self.head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.relu = nn.ReLU()
        self.pos_embed = nn.Parameter(torch.randn(1, 100 + 1, 512) * .02)
        def get_attention_block(depth, embed_dim=512, num_heads=8, mlp_ratio=4.,
                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            return nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i])
                for i in range(depth)])
        self.transformer = get_attention_block(depth=2)
        self.cls = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, 1, 512), 0., 0.2))

    def forward(self, x):
        x_nodes = x['batch_nodes'].cuda()
        x_edges = x['batch_edges']
        x_masks = x['batch_masks'].cuda()
        x_masks_bool = x_masks.new_zeros(x_nodes.shape[0], max(x_masks))
        for i in range(x_masks_bool.shape[0]):
            x_masks_bool[i, 0:x_masks[i]] = True
            x_masks_bool[i, x_masks[i]:] = False
        x_masks_bool = x_masks_bool.bool()

        #GATConv1
        x_nodes, adj_matrix = self.conv1(x_nodes, x_edges, x_masks)
        # MinCutPool1
        x_nodes, x_edges, l1 = self.pool1(x_nodes, adj_matrix, x_masks_bool)
        # GATConv2
        x_nodes, adj_matrix = self.conv2(x_nodes, x_edges, None)
        # MinCutPool2
        x_nodes, adj_matrix, l2 = self.pool2(x_nodes, adj_matrix, None)
        # Transformer
        adj_matrix_pos = self.pos_proj(adj_matrix)
        x_nodes = torch.cat([x_nodes, self.cls.repeat(x_nodes.shape[0], 1, 1)], dim=1)
        x_nodes = x_nodes + self.pos_embed
        x_nodes, _ = self.transformer((x_nodes, adj_matrix_pos))
        #Output
        pred = self.head(x_nodes[:, -1, :]).squeeze(dim=1)
        return pred, l1+l2
