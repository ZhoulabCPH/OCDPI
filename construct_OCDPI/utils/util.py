import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def make_batch(batch_graph):
    batch_masks = []
    nodes = batch_graph['nodes']
    bs = len(nodes)
    batch_edges = batch_graph['edges']
    batch_slide_names = batch_graph['slide_name']
    batch_OS_times = torch.tensor(batch_graph['OS_time'])
    batch_OSs = torch.tensor(batch_graph['OS'])
    max_nodes_num = 0
    for i in range(bs):
        max_nodes_num = max(max_nodes_num, nodes[i].shape[0])
        batch_masks.append(nodes[i].shape[0])
    batch_masks = torch.tensor(batch_masks)
    batch_nodes = torch.zeros(bs, max_nodes_num, nodes[0].shape[1])
    for i in range(bs):
        num = batch_masks[i]
        batch_nodes[i][0:num] = nodes[i]

    return {'batch_nodes': batch_nodes, 'batch_edges': batch_edges,
            'batch_masks': batch_masks}, batch_OSs, batch_OS_times, batch_slide_names

def make_batch_(batch_graph):
    batch_masks = []
    nodes = batch_graph['nodes']
    bs = len(nodes)
    batch_edges = batch_graph['edges']
    batch_slide_names = batch_graph['slide_name']
    batch_OS_times = torch.tensor(batch_graph['OS_time'])
    batch_OSs = torch.tensor(batch_graph['OS'])
    max_nodes_num = 0
    for i in range(bs):
        max_nodes_num = max(max_nodes_num, nodes[i].shape[0])
        batch_masks.append(nodes[i].shape[0])
    batch_masks = torch.tensor(batch_masks)
    batch_nodes = torch.zeros(bs, max_nodes_num, nodes[0].shape[1])
    for i in range(bs):
        num = batch_masks[i]
        batch_nodes[i][0:num] = nodes[i]

    return (batch_nodes, batch_edges, batch_masks), batch_OSs, batch_OS_times, batch_slide_names


def collate(batch):
    nodes = [b['node'] for b in batch]
    edges = [b['edge'] for b in batch]
    states = [b['state'] for b in batch]
    os = [b['os'] for b in batch]
    id = [b['id'] for b in batch]
    return {'nodes': nodes, 'edges': edges, 'states': states, 'os': os, 'id': id}


def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
