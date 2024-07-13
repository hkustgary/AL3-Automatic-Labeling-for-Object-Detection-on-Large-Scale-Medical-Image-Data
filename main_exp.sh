date
echo "#### Main experiment of training on pseudo labels on the BCCD data set ####"

date
echo "#### Step One: Generating LFs ####"
python3 gen_lfs.py

date
echo "#### Step Two: Generating Features ####"
python3 gen_features.py

date
echo "#### Step Three: Parameter Estimation & Weak Label Generation####"
python3 parameter_estimation.py

date
echo "#### Step Four: Downstream Object  Detection Model Training####"
python3 train_downstream.py


