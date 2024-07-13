import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import distance_mdf_mtx_calculate
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import euclidean_distances
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

batch_size_ab = 100
group_size_ab = 100
class FeatureMDSNet(nn.Module):
    def __init__(self, input_dim, n_components=2):
        super(FeatureMDSNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_components)
        )

    def forward(self, x):
        return self.network(x)

class GroupedMonotonicTransformNet(nn.Module):
    def __init__(self, n_points, n_groups):
        super(GroupedMonotonicTransformNet, self).__init__()
        self.n_groups = n_groups
        self.group_weights = nn.Parameter(torch.rand(n_groups, 1))
        self.activation = nn.ReLU()

    def forward(self, distance_matrix, groups):
        group_weight_matrix = torch.zeros_like(distance_matrix)
        for i in range(self.n_groups):
            group_mask = (groups == i)
            group_weight_matrix[group_mask] = self.group_weights[i]
        transformed = distance_matrix * group_weight_matrix
        monotonic_transformed = self.activation(transformed)
        return monotonic_transformed

def nmds_loss(embedded, original_dists, transformed_dists, print_loss = False):
    embedded_dists = torch.cdist(embedded, embedded, p=2)
    original_dists = original_dists[:, :embedded.size(0)]
    transformed_dists = transformed_dists[:, :embedded.size(0)]
    diff_original = original_dists - original_dists.t()
    diff_transformed = transformed_dists - transformed_dists.t()
    ordinal_loss = torch.mean((diff_original - diff_transformed) ** 2)
    stress_loss = ((embedded_dists - transformed_dists) ** 2).sum() / 2 *0.00000001
    return ordinal_loss + stress_loss
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.2)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
def train(model, transform_layer, features, original_dissimilarities, epochs, optimizer,groups, feature_length):
    for epoch in range(epochs):
        embedded = model(torch.tensor(features, dtype=torch.float32).to('cuda'))
        transformed_dissimilarities = transform_layer(original_dissimilarities, torch.tensor(groups))
        
        loss = nmds_loss(embedded, original_dissimilarities, transformed_dissimilarities, epoch % 20 == 0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            embedded = model(torch.tensor(features, dtype=torch.float32).to('cuda'))
            
            #res = distance_mdf_mtx_calculate(Tensor.cpu(embedded).detach().numpy(), feature_length)


class FidelitySampler:
    def __init__(self, error_matrix, batch_size=batch_size_ab):
        self.error_matrix = error_matrix
        self.batch_size = batch_size
        cleaned_error_matrix = np.nan_to_num(error_matrix, nan=0.0)
        self.probabilities = np.sum(cleaned_error_matrix, axis=1)
        self.probabilities /= np.sum(self.probabilities)

    def sample(self):
        total_prob = np.sum(self.probabilities)
        inverted_probabilities = total_prob - self.probabilities
        normalized_inverted_probabilities = inverted_probabilities / np.sum(inverted_probabilities)
        min_indices = normalized_inverted_probabilities.argsort()[:batch_size_ab]
        max_indices = normalized_inverted_probabilities.argsort()[-batch_size_ab:]
        sampled_indices = np.random.choice(len(self.probabilities), size=self.batch_size, replace=False, p=normalized_inverted_probabilities)
        return min_indices, max_indices

def compute_error_matrix(original_distances, embedded_distances):
    error_matrix = ((original_distances - embedded_distances) ** 8)/1000000
    return error_matrix


def batch_train(model, transform_layer, features, dissimilarities, epochs, optimizer, groups, feature_length, batch_size=batch_size_ab):
    embed_record = model(torch.tensor(features, dtype=torch.float32).to('cuda'))

    acc_1 = 0
    acc_2 = 0
    acc_3 = 0

    delta_count = 0
    for epoch in range(epochs):

            error_matrix = compute_error_matrix(Tensor.cpu(dissimilarities).detach().numpy(), euclidean_distances(Tensor.cpu(embed_record).detach().numpy(), Tensor.cpu(embed_record).detach().numpy()))
            fidelity_sampler = FidelitySampler(error_matrix, batch_size=batch_size_ab)
            batch_indices,_ = fidelity_sampler.sample()
            feature_batch = features[batch_indices].to('cuda')
            dissimilarity_batch = dissimilarities[batch_indices].to('cuda')
            group_batch = groups[batch_indices]
            embedded = model(feature_batch)
            transformed_dissimilarities = transform_layer(dissimilarity_batch, group_batch)
            loss = nmds_loss(embedded, dissimilarity_batch, transformed_dissimilarities)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if epoch % 100 == 0:

                embedded = model(torch.tensor(features, dtype=torch.float32).to('cuda'))
                embed_record = embedded
                accuracy_list = distance_mdf_mtx_calculate(Tensor.cpu(embedded).detach().numpy(), feature_length)

                err_faster = abs(29.2111 - accuracy_list[0])
                err_fcos = abs(29.2043 - accuracy_list[1])
                err_retina = abs(28.5093 - accuracy_list[2])
                res =  (err_faster+  err_fcos+ err_retina)/3
                delta = abs(acc_1-accuracy_list[0])+abs(acc_2-accuracy_list[1])+abs(acc_3-accuracy_list[2])
                if delta < 2:
                    delta_count+=1
                    if delta_count == 2:
                        print('parameter estimation finish, with recovery error = '+ str(res))
                        break
                else:
                    delta_count = 0
                acc_1 = accuracy_list[0]
                acc_2 = accuracy_list[1]
                acc_3 = accuracy_list[2]

def nmds_train(features, feature_length, dissimilarities, mode = 'full'):
    torch.manual_seed(5)
    n_points = features.shape[0]
    original_dissimilarity_matrix = dissimilarities.to('cuda')

    n_groups = group_size_ab
    epochs = 3000000
    groups = np.random.randint(0, n_groups, size=n_points) # more strategy?

    transform_layer = GroupedMonotonicTransformNet(n_points, n_groups).to('cuda')
    model = FeatureMDSNet(input_dim=features.shape[1], n_components=100).to('cuda')  # Adjust input dimension for flattened matrix
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    batch_train(model, transform_layer, features, original_dissimilarity_matrix,  epochs, optimizer, groups, feature_length )
    embedded = model(torch.tensor(features, dtype=torch.float32).to('cuda'))
    accuracy_list = distance_mdf_mtx_calculate(Tensor.cpu(embedded).detach().numpy(), feature_length)
    return accuracy_list

