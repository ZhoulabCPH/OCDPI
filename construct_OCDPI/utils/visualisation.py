import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openslide
import networkx as nx
import cv2
from matplotlib import cm


def slide_visualization(patches_gradient_path, slide_name):
    patch_size = 14
    patches_grad = pd.read_csv(patches_gradient_path, header=None)
    slide = openslide.OpenSlide(rf'../datasets/WSIs/TCGA_OV/{slide_name}.svs')
    indexes = []
    for i in range(len(patches_grad)):
        if patches_grad.iloc[i, 0].split('_')[0] == slide_name:
            indexes.append(i)
    slide_patches_gradient = patches_grad.iloc[indexes]
    patches_name = slide_patches_gradient.iloc[:, 0].to_list()
    patches_attribution = slide_patches_gradient.iloc[:, 2].to_list()
    patches_attribution = np.array(patches_attribution).astype(np.float32)
    width, height = slide.dimensions
    slide_thumbnail = slide.get_thumbnail((width / 128, height / 128))
    slide_thumbnail = np.asarray(slide_thumbnail)[:, :, ::-1].copy()
    slide_thumbnail = cv2.cvtColor(slide_thumbnail, cv2.COLOR_BGR2RGB)
    slide_thumbnail_gray = cv2.cvtColor(slide_thumbnail, cv2.COLOR_BGR2GRAY)
    slide_thumbnail_ = (slide_thumbnail - slide_thumbnail.min()) / (
            slide_thumbnail.max() - slide_thumbnail.min())
    mask = np.full_like(slide_thumbnail_gray, 0.).astype(np.float32)
    for i, filename in enumerate(patches_name):
        x, y = map(int, (filename.split('_')[-1].split('-')[0], filename.split('_')[-1].split('-')[1].split('.')[0]))
        mask[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size].fill(
            patches_attribution[i])
    color = cm.get_cmap('jet')
    heatmap = slide_thumbnail_
    heatmap = np.float32(heatmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    for i in range(slide_thumbnail.shape[0]):
        for j in range(slide_thumbnail.shape[1]):
            if mask[i][j] > 0:
                heatmap[i, j, :] = color(np.clip(0.5, 1, 0.5 + mask[i][j]))[0:-1]
            elif mask[i][j] < 0:
                heatmap[i, j, :] = color(np.clip(0, 0.5, 0.5 - abs(mask[i][j])))[0:-1]
    heatmap = np.clip(heatmap, 0, 1)

    plt.imshow(heatmap)

    # PCs visualisation
    patches_grad.index = patches_grad.iloc[:, 0]
    cluster_patches = patches_grad.loc[patches_name, :]
    patches_name = cluster_patches.iloc[:, 0].to_list()
    patches_cluster = cluster_patches.iloc[:, -1].to_list()
    G = nx.Graph()
    for (i, patch_name) in enumerate(patches_name):
        x, y = map(int,
                   (patch_name.split('_')[-1].split('-')[0], patch_name.split('_')[-1].split('-')[1].split('.')[0]))
        G.add_node(str((x, y)), label=patches_cluster[i])
    colors = ['#0a62a4', '#3a9cc7', '#75c8c7', '#b6e2bb', '#e0f2dc', '#f5fbef', '#f5f2f9', '#cca2cb', '#de2179',
              '#aa0649']
    label_colors = {i: colors[i] for i in range(len(colors))}
    node_colors = [label_colors[int(G.nodes[n]['label'])] for n in G.nodes()]
    pos = nx.spring_layout(G)
    for index in pos.keys():
        coordinate_ = index
        x = int(coordinate_.split(',')[0][1:])
        y = int(coordinate_.split(',')[1][1:][:-1])
        coordinate = np.array([x / 10, -y / 10])
        pos[index] = coordinate

    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_colors)
