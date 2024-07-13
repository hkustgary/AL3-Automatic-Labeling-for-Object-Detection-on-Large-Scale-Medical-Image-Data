# AL3-Automatic-Labeling-for-Object-Detection-on-Large-Scale-Medical-Image-Data

# Requirement
install enviroment using: conda env create -f environment.yml
activate enviroment using: conda activate od
# Dataset & SAM
Download the BCCD dataset from: https://github.com/Shenggan/BCCD_Dataset
Download SAM from https://github.com/facebookresearch/segment-anything
# Usage
We recommend using the script main_exp.sh to generate pseudo labels and train a relative object detection model on BCCD dataset. However, if you prefer do it step by step, please follow Steps 1 to 4.
# Step One: Train Label Functions
We train LFs using the 5% development set, the output can be found in the 'lfs' directory.
Please run python3 main_exp.sh
# Step Two: Generating features
In order to recover parameters, we first extract features of each label function output using a pretrained model. 
python3 gen_features.py
# Step Three: Parameter estimation and weak label generation
In this step, we first estimate the parameters of the label model, then using the learned label model to generate pseudo weak labels for the remaining 95% unlabeled data.
python3 parameter_estimation.py
# Step Four: Training downstream model
We train a downstream object detection model using 5%labeled data + 95%pseudo labeled data
python3 train_downstream.py


