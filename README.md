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
- graphs: Store Graph representation of WSIs.
- gradients: Store gradients of patches in TCGA discovery cohort.
****
## data_preprocessing
- <code>multi_thread_WSI_segmentation.py</code>: Used to segment and filter patches from WSIs. Implemented based on <code>histolab</code> package.
- <code>make_hdf5.py</code>: Save image to hdf5 format.
- <code>model.py</code>: Implementation of BarlowTwins.
- <code>train.py</code>: Training the BarlowTwins model on TCGA cohort.
- <code>utils.py</code>: Using pre-trained ResNet50 to obtain histopathological features of patches.
****
## get_patches_feature
- <code>ctran.py</code>: Implementation of CTransPath.
- <code>get_CTransPath_features.py</code>: Using pre-trained CTransPath to obtain histopathological features of patches.
Part of the implementation here is based on [CTransPath](https://github.com/Xiyue-Wang/TransPath).
****
## construction_OCDPI
- <code>args.py</code>: Get the training parameters of the graph-based deep learning (GDL) model.
- <code>model.py</code>: Implementation of GDL model.
- <code>train.py</code>: Training the GDL model on TCGA cohort.
- <code>utils.py</code>:WSI-based graph construction.

