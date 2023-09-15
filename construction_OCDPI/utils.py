import pandas as pd
import scanpy as sc
import torch.nn.functional as F
from sklearn import preprocessing
import re
import openpyxl
import numpy as np
import anndata as ad
import torch

from torch.nn.modules.loss import _Loss


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def make_label_OV(patients_csv, patches_csv):
    patients = pd.read_csv(patients_csv)
    patients.index = patients.iloc[:,0]

    absent = []
    for i in range(patients.shape[0]):
        patient = patients.iloc[i, 0]
        os = str(patients.iloc[i, 2])
        os_state = str(patients.iloc[i, 3])
        if os == 'nan' or os_state == 'nan':
            absent.append(patient)
    for i in absent:
        patients = patients.drop(i)

    patients_os = np.array(patients.iloc[:,-2])
    t_max = np.max(patients_os)
    t_min = np.min(patients_os)
    patients_os_ = []
    for i in range(len(patients_os)):
        l = (t_min * (t_max - patients_os[i])) / (patients_os[i] * (t_max - t_min))
        # patients_os_.append(sigmoid(l))
        patients_os_.append(l)
    patients['risk_score'] = patients_os_
    slides = []
    slides_OS = []
    slides_OS_STATE = []
    slides_risk_score = []
    for i in range(patients.shape[0]):

        os = patients.iloc[i, 2]
        os_state = patients.iloc[i, 3]
        risk_score = patients.iloc[i, 4]
        slides_ = [slide for slide in patients.iloc[i,1].split("'") if slide != '[' if slide!= ']' if slide !=', ']
        for s in slides_:
            slides.append(s)
            slides_OS.append(os)
            slides_OS_STATE.append(os_state)
            slides_risk_score.append(risk_score)
    patches = pd.read_csv(patches_csv)
    patches = patches.drop('Unnamed: 0',axis=1)
    slides_patches = {}
    slides_patches_feature = {}
    for i in range(patches.shape[0]):
        slide = patches.iloc[i, 0].split('_')[0]
        if slide in slides:
            feature = np.array(list(patches.iloc[i, 1:]))
            if slide not in slides_patches.keys():
                slides_patches[slide] = [patches.iloc[i, 0]]
                slides_patches_feature[slide] = [feature]
            else:
                slides_patches[slide].append(patches.iloc[i, 0])
                slides_patches_feature[slide].append(feature)
    slides_edges = {}
    small_part = []
    slide_len = []
    for slide in slides_patches.keys():
        obs = pd.DataFrame()
        obs['patches'] = slides_patches[slide]
        var = [i for i in range(256)]
        var = pd.DataFrame(index=var)
        X = np.array(slides_patches_feature[slide])
        adata = ad.AnnData(X, obs=obs, var=var)
        slide_len.append(len(obs['patches']))
        if len(obs['patches']) < 10:
            small_part.append(slide)
            n_neighbors = 1
        else:
            n_neighbors = int(len(obs['patches'])/10)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, method='umap')
        connectivities = adata.obsp['connectivities']
        slides_edges[slide] = connectivities

    data = pd.DataFrame()
    data['slides'] = slides
    data['OS'] = slides_OS
    data['OS_STATE'] = slides_OS_STATE
    data['risk_score'] = slides_risk_score
    slides_patches_ = []
    slides_features_ = []
    slides_edges_ = []
    dp=99999
    for i in range(data.shape[0]):
        if data.iloc[i,0] in small_part:
            dp=i
    data=data.drop(dp)
    for i in data['slides']:
        slides_patches_.append(slides_patches[i])
        slides_features_.append(np.array(slides_patches_feature[i]))
        slides_edges_.append(slides_edges[i])
    data['patches'] = slides_patches_
    data.index = [i for i in range(data.shape[0])]

    np.save('./GDL_nodes.npy', np.array(slides_features_))
    np.save('./GDL_edges.npy', np.array(slides_edges_))
    data.to_csv('./GDL.csv')
    print('Finished!')


def make_batch(batch_graph):
    mask = []
    nodes = batch_graph['nodes']
    bs = len(nodes)
    edges_index = batch_graph['edges_index']
    labels = torch.tensor(batch_graph['labels'])
    risk_score = torch.tensor(batch_graph['risk_score'])
    states = torch.tensor(batch_graph['states'])
    os = torch.tensor(batch_graph['os'])
    id = batch_graph['id']
    max_nodes_num = 0
    for i in range(bs):
        max_nodes_num = max(max_nodes_num, nodes[i].shape[0])
        mask.append(nodes[i].shape[0])
    mask = torch.tensor(mask)
    batch_nodes = torch.zeros(bs, max_nodes_num, 512)
    for i in range(bs):
        num = mask[i]
        batch_nodes[i][0:num] = nodes[i]
    return {'batch_nodes': batch_nodes, 'batch_edges_index': edges_index, 'mask': mask}, labels, risk_score, states, os, id


def collate(batch):
    nodes = [b['node'] for b in batch]  # w, h
    edges_index = [b['edge_index'] for b in batch]
    labels = [b['label'] for b in batch]
    risk_score = [b['risk_score'] for b in batch]
    states = [b['state'] for b in batch]
    os = [b['os'] for b in batch]
    id = [b['id'] for b in batch]
    return {'nodes': nodes, 'edges_index': edges_index, 'labels': labels, 'risk_score': risk_score,'states': states, 'os': os, 'id': id}


def sort_by_indexes(lst, indexes, reverse=False):
  return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
          x[0], reverse=reverse)]



