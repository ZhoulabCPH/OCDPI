# OCDPI [![DOI](https://zenodo.org/badge/691128305.svg)](https://zenodo.org/doi/10.5281/zenodo.10405066)
Code for 'Predicting prognoses and therapy responses in ovarian cancer patients from histopathology images using graph deep learning: a multicenter retrospective study'

![flowchart](https://github.com/ZhoulabCPH/OCDPI/assets/143063392/e77dadbd-3da3-4e9d-9564-536ba309101b)

****
## Dataset 
- [TCGA](https://portal.gdc.cancer.gov/projects/TCGA-OV), we incorporate TCGA-OV cohort into our study, and its open access to all.
- [PLCO](https://cdas.cancer.gov/plco/#:~:text=PLCO%20has%20the%20following%20five%20ClinicalTrials.gov%20registration%20numbers%3A,the%20PLCO%20trial%20are%20available%20on%20this%20website.), the ovarian cancer of PLCO data was used in this study. If anyone wants to obtain PLCO data, please initiate an application on the official website.
- HMUCH, HMUCH is available from the corresponding author upon reasonable request.
****
## datasets
- clinical_data: Clinical information of each cohort, stored in csv format. At least three columns, id, event time and event state are required for training or obtaining evaluation results.
- WSIs: Store whole slide images of each cohort.
- patches: Store patches extracted from WSIs.
- graphs: Store graph representation of WSIs.
- gradients: Store gradients of patches in TCGA discovery cohort.
****
## checkpoints
- checkpoint_CTransPath: CTransPath model pretrained by [CTransPath](https://github.com/Xiyue-Wang/TransPath).
- checkpoint_GDL: Graph-based deep learning (GDL) pretrained on our TCGA discovery cohort.
****
## data_preprocessing
- <code>multi_thread_WSI_segmentation.py</code>: Used to segment and filter patches from WSIs. Implemented based on <code>histolab</code> package.
****
## get_patches_feature
- <code>ctran.py</code>: Implementation of CTransPath.
- <code>get_CTransPath_features.py</code>: Using pre-trained CTransPath to obtain histopathological features of patches.
  
  Part of the implementation here is based on [CTransPath](https://github.com/Xiyue-Wang/TransPath).
****
## construction_OCDPI
- <code>utils/conceptualize_WSI_to_graph.py</code>: Get the graph representation of WSIs and further used for the graph-based deep learning (GDL) model.
- <code>utils/dataset.py</code>: Generate datasets.
- <code>utils/util.py</code>: Tools and loss function used in training.
- <code>utils/calculate_gradient_of_patch.py</code>: Integrated Gradients (IG)-based gradient calculation for model interpretability.
- <code>utils/visualisation.py</code>: Gradient value visualization.
- <code>model</code>: Implementation of GDL model.
- <code>train</code>: Training the GDL model.
- <code>evaluation</code>: Evaluation of the GDL model in multi-center external cohorts.
## Usage
If you intend to utilize it for paper reproduction or your own WSI dataset, please adhere to the following workflow:
  1) Configuration Environment.
  2) Create a folder for your data in <code>datasets</code> and download or move the WSIs there.
  3) Use <code>data_preprocessing/multi_thread_WSI_segmentation.py</code> to segment WSIs into patches.
  4) Use <code>get_patches_feature/conceptualize_WSI_to_graph.py</code> to obtain representation vector of patches.
  5) Use <code>construct_OCDPI/utils/conceptualize_WSI_to_graph.py</code> to obtain graph representation for WSIs.
  6) Run <code>construct_OCDPI/train.py</code> to train the GDL model. When training is complete you can use this GDL model to calculate OCDPI (Ovarian Cancer Digital Pathology Index) from each WSI.
  7) Using <code>construct_OCDPI/evaluation.py</code> you can obtain the evaluation results of the survival prediction performance of the GDL model.

