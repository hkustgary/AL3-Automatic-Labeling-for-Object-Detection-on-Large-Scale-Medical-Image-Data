from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import numpy as np
import math
import scipy.stats as stats

norm_mean, norm_std = (0.5,), (0.5,)  # 0,1 to -1,1
de_norm_mean = -torch.tensor(norm_mean)/torch.tensor(norm_std)
de_norm_std = 1.0/torch.tensor(norm_std)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std),])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def triplet(a_b, b_c, a_c):
    # a = math.sqrt(a_b * a_c / b_c)
    # b = math.sqrt(a_b * b_c / a_c)
    # c = math.sqrt(a_c * b_c / a_b)
    # print(a_b)
    # print(b_c)
    # print(a_c)
    # a = math.sqrt((a_b*a_b + a_c*a_c - b_c*b_c)/2)
    # b = math.sqrt(a_b*a_b-a*a)
    # c = math.sqrt(a_c*a_c-a*a)

    a = (a_b + a_c - b_c)/2
    b = (a_b + b_c - a_c)/2
    c = (a_c + b_c - a_b)/2


    return a,b,c
    
def filter_predict(data,topk):
    extracted_items = []

    for entry in data:
        boxes = entry['boxes']
        scores = entry['scores']
        labels = entry['labels']
        
        # 找到labels等于3的索引
        indices = torch.where(labels == 1)[0]

        filtered_boxes = boxes[indices]
        filtered_scores = scores[indices]
        filtered_labels = labels[indices]
        
        if len(filtered_scores) > topk:
            top_scores_indices = torch.argsort(filtered_scores, descending=True)[:topk]
        else:
            top_scores_indices = torch.arange(len(filtered_scores))

        # 提取分数最高的三个检测结果
        extracted_boxes = filtered_boxes[top_scores_indices]
        extracted_scores = filtered_scores[top_scores_indices]
        extracted_labels = filtered_labels[top_scores_indices]

        # 将这些结果添加到最终列表中
        extracted_items.append({
            'boxes': extracted_boxes.cpu().detach().numpy(),
            'scores': extracted_scores.cpu().detach().numpy(),
            'labels': extracted_labels.cpu().detach().numpy()
        })
    return extracted_items

def model_inference(model, path, topk):
    model.eval()
    image = Image.open(path).convert("RGB")
    imagess = [transform(image)]
    imagess = list(image.to(device) for image in imagess)   
    pred = filter_predict(model(imagess),topk)
    return pred

def random_inference(image_path):
    """
    Generates random detections for an image.

    Parameters:
    - image_path (str): The path to the image for which to generate detections.

    Returns:
    - list[dict]: A list containing a single dictionary with random detection results.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    # Set the random seed for reproducibility
    random.seed(582153221)
    
    # Define the number of detections to generate
    num_detections = 1
    
    boxes = []

    for _ in range(num_detections):
        x1 = random.uniform(0, width * 0.9)
        y1 = random.uniform(0, height * 0.9)
        b_w = 88
        b_h = 100
        boxes.append(np.array([x1, y1, x1+b_w, y1+b_h] ))


   
    
    # Generate random scores
    scores = np.array([random.random() for _ in range(num_detections)], dtype=np.float32)
    
    # Generate random labels (assuming COCO dataset where label ids range from 1 to 80)
    labels = np.array([random.randint(1, 80) for _ in range(num_detections)])
    
    # Return the random detection result
    return [{
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }]
def parse_and_stack_tensor_strings(data_frame):
    # Initialize an empty list to collect tensor data
    tensor_list = []
    
    # Iterate through each tensor string in the data frame
    for tensor_string in data_frame['0']:
        # Parse the tensor string to extract numerical data
        tensor_data = eval(tensor_string.replace("tensor", ""))
        # Convert the list to a PyTorch tensor and append to list
        tensor_list.append(torch.tensor(tensor_data))
    
    # Stack all tensors into a single tensor
    stacked_tensor = torch.cat(tensor_list, dim=0)
    
    return stacked_tensor
def process_and_combine_tensors(file_paths):
    # Initialize an empty list to hold tensors from all files
    combined_tensors = []
    # Process each file
    feature_length = []
    for path in file_paths:
        data = pd.read_csv(path)
        
        # Parse and stack tensors from the current file
        tensor = parse_and_stack_tensor_strings(data)
        # num_rows = tensor.shape[0] // 10
        # tensor = tensor[:num_rows]

        combined_tensors.append(tensor)
        feature_length.append(len(data))

    # Concatenate all tensors along dimension 0 (stacking them vertically)
    final_tensor = torch.cat(combined_tensors, dim=0)
    return final_tensor,feature_length


def distance_mdf_mtx_calculate(embedding_location, feature_length):
    ## input: 每个bbox对应的坐标 (原始feature或嵌入位置都可以)
    ##    --形如：
#     [[  1.4580685   -4.306163    -1.0388721  ...  -4.1970096   -0.640073
#    -3.2454953 ]
#  [ -5.1402926   -5.074091    -6.9581437  ...   1.4398277    4.8056707
#     2.4358826 ]
#  [  1.3766929   -9.166671    -6.00928    ...  -2.7132804    0.08671889
#     6.6810718 ]
#  ...
#  [ -6.467313    -3.045171     3.3683686  ...  10.888274    -2.404644
#   -11.306165  ]
#  [ 10.859201    -3.5110848   -7.277192   ...  10.492323    -0.42190728
#   -12.311717  ]
#  [ -5.0806546   -5.2556973    1.7816641  ...  11.303874    14.179235
#   -12.038748  ]]
# output：LF之间距离
    #dist_matrix = Tensor.cpu(embedding_location).detach().numpy()

    dist_matrix = distance_matrix(embedding_location, embedding_location)
    dist_matrix_tensor = torch.tensor(dist_matrix)
    #print(feature_length)
    ind = [0]
    for length in feature_length:
        ind.append(length)
    
    indices = np.cumsum(ind)
    
    accuracy_list = np.zeros(len(feature_length))
    
    total_time = 0
    # 为了遍历每三个数据集的组合，我们使用三层循环
    for i in range(len(feature_length)):
        for j in range(i + 1, len(feature_length)):
            for k in range(j + 1, len(feature_length)):
                start_i = indices[i]
                end_i = indices[i + 1]
                start_j = indices[j]
                end_j = indices[j + 1]
                start_k = indices[k]
                end_k = indices[k + 1]
           
                # 提取三个数据集之间的子矩阵，并计算平均距离
                submatrix_ij = dist_matrix_tensor[start_i:end_i, start_j:end_j]
                submatrix_ik = dist_matrix_tensor[start_i:end_i, start_k:end_k]
                submatrix_jk = dist_matrix_tensor[start_j:end_j, start_k:end_k]
               
                avg_dist_ij = np.nanmean(torch.diag(submatrix_ij))
                avg_dist_ik = np.nanmean(torch.diag(submatrix_ik))
                avg_dist_jk = np.nanmean(torch.diag(submatrix_jk))

                print(f"组合 ({i+1}, {j+1}, {k+1}):")
                print(f"    数据集 {i+1} 和数据集 {j+1} 之间的平均距离: {avg_dist_ij:.4f}")
                print(f"    数据集 {i+1} 和数据集 {k+1} 之间的平均距离: {avg_dist_ik:.4f}")
                print(f"    数据集 {j+1} 和数据集 {k+1} 之间的平均距离: {avg_dist_jk:.4f}")

                a_i,a_j,a_k = triplet(avg_dist_ij, avg_dist_jk,avg_dist_ik)
                accuracy_list[i]+=a_i
                accuracy_list[j]+=a_j
                accuracy_list[k]+=a_k
                total_time += 1
    #pearson_metric(res[0],res[1],res[2])
    
    return accuracy_list/total_time