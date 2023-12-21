import os
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import random
import feather
from scipy.sparse import csr_matrix
def setup_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

seed = 0
setup_seed(seed)

def are_neighbors(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) + abs(y1 - y2) == 1


def construction_graph(clinical_data_path, slides_feature_path, cohort):
    codebook = pd.read_csv(f'{clinical_data_path}')
    slides_patch = []
    slides_patch_feature = []
    slides_edge = []
    slides_name = codebook.loc[:, 'SLIDES'].to_list()
    count = 0
    for slide_name in slides_name:
        count = count + 1
        print(f'{count} {slide_name}')
        slide = feather.read_dataframe(f'{slides_feature_path}/{slide_name}.feather')
        slide_patches_features = slide.iloc[:,1:].values
        slide_patches = list(slide.iloc[:,0])
        patches_coordinates = [(int(patch.split('-')[-2].split('_')[-1]), int(patch.split('-')[-1].split('.')[0])) for patch in slide_patches]
        # Compute physics_edge
        physics_edge = np.zeros((len(slide_patches), len(slide_patches)))
        for i in range(len(physics_edge)):
            for j in range(len(physics_edge)):
                coord1 = patches_coordinates[i]
                coord2 = patches_coordinates[j]
                if are_neighbors(coord1, coord2):
                    physics_edge[i][j] = 1
        # Compute logical_edge
        obs = pd.DataFrame()
        obs['patches'] = slide_patches
        var = [i for i in range(2048)]
        var = pd.DataFrame(index=var)
        X = np.array(slide_patches_features)
        adata = ad.AnnData(X, obs=obs, var=var)
        n_neighbors = 9
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, method='umap', use_rep='X')
        logical_edge = adata.obsp['distances']
        logical_edge = logical_edge.toarray()
        logical_edge[logical_edge!=0] = 1
        adj_matrix = physics_edge + logical_edge
        adj_matrix[adj_matrix!=0] = 1
        adj_matrix = csr_matrix(adj_matrix)
        slides_patch.append(slide_patches)
        slides_patch_feature.append(slide_patches_features)
        slides_edge.append(adj_matrix)

    np.save(f'../datasets/{cohort}/{cohort}_patches_name.npy', np.array(slides_patch, dtype=object))
    np.save(f'../datasets/{cohort}/{cohort}_nodes.npy', np.array(slides_patch_feature, dtype=object))
    np.save(f'../datasets/{cohort}/{cohort}_edges.npy', np.array(slides_edge))


