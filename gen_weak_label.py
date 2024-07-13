import torch
import sys
import os
import torchvision
from PIL import Image
from sklearn.metrics import euclidean_distances
from torch import Tensor
import numpy as np
import json
from lfs_accuracy_estimation import single_img_feature
from dataset import BCCDDataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import model_inference
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  'segment-anything/'))
from single_image_sam import sam_predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


norm_mean, norm_std = (0.5,), (0.5,)  # 0,1 to -1,1
de_norm_mean = -torch.tensor(norm_mean)/torch.tensor(norm_std)
de_norm_std = 1.0/torch.tensor(norm_std)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std),])
def filter_top_3(data):
    data = data[0]
    _, indices = torch.sort(data['scores'], descending=True)
    top_indices = indices[:3]
    top_boxes = data['boxes'][top_indices]
    return [top_boxes]
def single_image_weak_label(image,path,model1,model2,model3,accs):
    image = Image.open(path).convert("RGB")
    imagess = [transform(image)]
    imagess = list(image.to(device) for image in imagess)   
    pred1 = filter_top_3(model1(imagess))[0]
    pred2 = filter_top_3(model2(imagess))[0]
    pred3 = filter_top_3(model3(imagess))[0]
    combined_list=sam_predict(path)   
    milde_lf_distances = [accs[0],accs[1],accs[2]]
    milde_bbox_weight = []

    min_distance = float('inf')
    max_distance = 0
    distances = []

    for i, bbox in enumerate(combined_list):
        distance1 = 0
        for model1_bbox in pred1:
            feature1 = single_img_feature(path, bbox)
            feature2 = single_img_feature(path, model1_bbox.cpu().detach().numpy())
            distance = euclidean_distances(Tensor.cpu(feature1).detach().numpy(), Tensor.cpu(feature2).detach().numpy())
            distance1 += distance

        distance2 = 0
        for model2_bbox in pred2:
            feature1 = single_img_feature(path, bbox)
            feature2 = single_img_feature(path, model2_bbox.cpu().detach().numpy())
            distance = euclidean_distances(Tensor.cpu(feature1).detach().numpy(), Tensor.cpu(feature2).detach().numpy())
            distance2 += distance

        distance3 = 0
        for model3_bbox in pred3:
            feature1 = single_img_feature(path, bbox)
            feature2 = single_img_feature(path, model3_bbox.cpu().detach().numpy())
            distance = euclidean_distances(Tensor.cpu(feature1).detach().numpy(), Tensor.cpu(feature2).detach().numpy())
            distance3 += distance

        distances.append((distance1, distance2, distance3))

        min_distance = min(min_distance, distance1, distance2, distance3)
        max_distance = max(max_distance, distance1, distance2, distance3)

    if max_distance == min_distance:
        max_distance = min_distance + 1

    milde_bbox_weight = []
    for (distance1, distance2, distance3) in distances:
        scaled_distance1 = (distance1 - min_distance) / (max_distance - min_distance)
        scaled_distance2 = (distance2 - min_distance) / (max_distance - min_distance)
        scaled_distance3 = (distance3 - min_distance) / (max_distance - min_distance)
        dist = milde_lf_distances[0]*scaled_distance1 + milde_lf_distances[1]*scaled_distance2 + milde_lf_distances[2]*scaled_distance3
        milde_bbox_weight.append(dist)
        print(str(dist) + '=' + str(milde_lf_distances[0])+'*'+str(scaled_distance1)+str(milde_lf_distances[1])+'*'+str(scaled_distance2)+str(milde_lf_distances[2])+'*'+str(scaled_distance3))

    flattened_values = [v.item() for v in milde_bbox_weight]
    indices_of_smallest = np.argsort(flattened_values)[:min(1, len(flattened_values))]
    milde_selected_boxes = [combined_list[i] for i in indices_of_smallest]
    milde_score = [flattened_values[i]/250 for i in indices_of_smallest]

    return milde_selected_boxes,milde_score



def generate_weak_label(accs):
    print('start generating pseudo labels')
    model1 = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model1.roi_heads.box_predictor.cls_score.in_features
    model1.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2) #background +1
    model1.load_state_dict(torch.load('lfs/faster.pth'))
    model1.to(device)
    model1.eval()

    model2 = torchvision.models.detection.fcos_resnet50_fpn().to(device)
    model2.load_state_dict(torch.load('lfs/fcos.pth'))
    model2.eval()

    model3 = torchvision.models.detection.retinanet_resnet50_fpn().to(device)
    model3.load_state_dict(torch.load('lfs/retina.pth'))
    model3.eval()

    dataset = BCCDDataset(root_dir='BCCD')
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    milde_hash_table = {}

    total_len = len(dataset)
    for images, targets,path in dataset_loader:
        path = path[0]
        milde_selected_boxes,_ = single_image_weak_label(images,path,model1,model2,model3,accs)
        milde_hash_table[path] = milde_selected_boxes

    milde_json_data = json.dumps(milde_hash_table, indent = 4)

    with open('al3_pseudo_labels.json','w') as file:
        file.write(milde_json_data)
    print('pseudo label generation success')
