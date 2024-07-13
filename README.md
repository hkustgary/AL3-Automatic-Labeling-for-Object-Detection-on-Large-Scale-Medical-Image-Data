# AL3-Automatic-Labeling-for-Object-Detection-on-Large-Scale-Medical-Image-Data
Annotating large-scale unlabeled medical image datasets for object detection is a costly process. Therefore, we proposed AL3, which is designed for automatic labeling for medical image object detection tasks.<br> This paper is under reviewed by VLDB 2024.
<img width="1182" alt="image" src="https://github.com/user-attachments/assets/3a31f1d4-d21c-4503-ae47-827e26442c2c">

# Overview
<img width="1182" alt="image" src="https://github.com/user-attachments/assets/80e3aee5-2642-4dfb-ac5b-3bdb92b24c54">

# Requirement
Install enviroment using: `conda env create -f environment.yml`<br>
Activate enviroment using: `conda activate od`
# Dataset & SAM
Download the BCCD dataset from: https://github.com/Shenggan/BCCD_Dataset<br>
Download SAM from https://github.com/facebookresearch/segment-anything
# Usage
We recommend using the script `./main_exp.sh` to generate pseudo labels and train a relative object detection model on BCCD dataset. However, if you prefer do it step by step, please follow Steps 1 to 4.
# Step One: Train Label Functions
We train LFs using the 5% development set, the output can be found in the 'lfs' directory.<br>
`python3 gem_lfs.py`
# Step Two: Generating features
In order to recover parameters, we first extract features of each label function output using a pretrained model. <br>
`python3 gen_features.py`
# Step Three: Parameter estimation and weak label generation
In this step, we first estimate the parameters of the label model, then using the learned label model to generate pseudo weak labels for the remaining 95% unlabeled data.<br>
`python3 parameter_estimation.py`
# Step Four: Training downstream model
We train a downstream object detection model using 5%labeled data + 95%pseudo labeled data.<br>
`python3 train_downstream.py`


