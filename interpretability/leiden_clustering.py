import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from matplotlib import pyplot as plt

def leiden(workspace, nodes, resolution, k_neighbor, pca):
    workspace = workspace
    data_nodes = np.load(nodes, allow_pickle=True)
    obs = pd.DataFrame()
    PATCHES = []
    FEATURES = []
    for i in range(len(workspace)):
        patches = [slide for slide in workspace.iloc[i, 2].split("'") if slide != '[' if slide != ']' if
                   slide != ', ']
        for patch in patches:
            PATCHES.append(patch)
        FEATURES.append(data_nodes[i])
        if len(patches) != len(data_nodes[i]):
            print(i)

    obs['patch'] = PATCHES
    var_ = [i for i in range(512)]
    var = pd.DataFrame(index=var_)
    X = np.vstack(FEATURES)
    adata = ad.AnnData(X, obs=obs, var=var)
    sc.pp.neighbors(adata, n_neighbors=k_neighbor, method='umap', metric='euclidean')
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data', add_outline=False, legend_fontsize=10,
               legend_fontoutline=2, frameon=False, title='Leiden_train', show=False, palette='tab20')

    adata.write_h5ad(
        f'./adata_{str(resolution)}_{str(k_neighbor)}_{str(pca)}.h5ad')
    return adata
