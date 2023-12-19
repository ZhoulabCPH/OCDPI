# OCDPI [![DOI](https://zenodo.org/badge/691128305.svg)](https://zenodo.org/doi/10.5281/zenodo.10405066)
Code for 'Deep learning-enabled individualized prediction of prognoses and therapy responses in patients with ovarian cancerfrom histopathology images: a multicenter retrospective study'

![flowchart](https://github.com/ZhoulabCPH/OCDPI/assets/143063392/e77dadbd-3da3-4e9d-9564-536ba309101b)

****
## Dataset 
- [TCGA](https://portal.gdc.cancer.gov/projects/TCGA-OV), we incorporate TCGA-OV cohort into our study, and its open access to all.
- [PLCO](https://cdas.cancer.gov/plco/#:~:text=PLCO%20has%20the%20following%20five%20ClinicalTrials.gov%20registration%20numbers%3A,the%20PLCO%20trial%20are%20available%20on%20this%20website.), the ovarian cancer of PLCO data was used in this study. If anyone wants to obtain PLCO data, please initiate an application on the official website.
- HMUCH, HMUCH is available from the corresponding author upon reasonable request.
****
## self-supervised pretraining
- <code>args.py</code>: Get the training parameters of the BarlowTwins model.
- <code>make_hdf5.py</code>: Save image to hdf5 format.
- <code>model.py</code>: Implementation of BarlowTwins.
- <code>train.py</code>: Training the BarlowTwins model on TCGA cohort.
- <code>utils.py</code>: Using pre-trained ResNet50 to obtain histopathological features of patches.
****
## construction_OCDPI
- <code>args.py</code>: Get the training parameters of the graph-based deep learning (GDL) model.
- <code>model.py</code>: Implementation of GDL model.
- <code>train.py</code>: Training the GDL model on TCGA cohort.
- <code>utils.py</code>:WSI-based graph construction.

## interpretability
- <code>leiden_clustering.py</code>: Implement clustering of patches using Leiden method.
****
