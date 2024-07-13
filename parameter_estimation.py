##### STEP THREE: ESTIMATE PARAMETERS & GENERATE WEAK LABELS #####

from utils import process_and_combine_tensors
import torch
from sklearn.metrics import euclidean_distances
from lnmds import nmds_train
from gen_weak_label import generate_weak_label

def estimate_param():
    file_paths = [
            'faster_feature_list.csv',
            'fcos_feature_list.csv',
            'retina_feature_list.csv',
        ]
    features,feature_length  = process_and_combine_tensors(file_paths)
    features = torch.tensor(features)
    dissimilarities = torch.tensor(euclidean_distances(features), dtype=torch.float32)
    accs = nmds_train(features, feature_length, dissimilarities, mode = 'full' )
    return accs

if __name__ == "__main__":
    accs = estimate_param()
    generate_weak_label(accs)