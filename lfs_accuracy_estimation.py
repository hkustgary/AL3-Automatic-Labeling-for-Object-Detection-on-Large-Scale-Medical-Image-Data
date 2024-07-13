import torch
import sys
import os
from dataset import BCCDDataset
import torchvision
from utils import model_inference, random_inference,process_and_combine_tensors,distance_mdf_mtx_calculate,triplet
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50
import torchvision.transforms as transforms
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import torch.nn as nn
import cv2
from sklearn.metrics import euclidean_distances
from torchvision.models import convnext_base
from lnmds import nmds_train
pd.set_option('display.max_columns', None)  # 或者设定一个具体的较大数字
pd.set_option('display.max_rows', None)  # 或者设定一个具体的较大数字

def single_img_feature(path, bbox):
    return extract_features(path, bbox, back_model, transform)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_fn(batch):
    return tuple(zip(*batch))


# 创建模型实例
back_model = resnet50(pretrained=True).to('cuda')
back_model.eval()  # 设置为评估模式
back_model.to(device)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path, bbox, model, transform):
    """
    提取给定边界框内图像的特征。
    :param image_path: 图像的路径
    :param bbox: 边界框，格式为(x_min, y_min, x_max, y_max)
    :param model: 预训练的模型用于特征提取
    :param transform: 对图像进行预处理的转换
    :return: 提取的特征
    """
    image = cv2.imread(image_path)
    # 根据边界框裁剪图像
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[round(y_min):round(y_max), round(x_min):round(x_max)]


    # 将图像转换为模型需要的格式
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    processed_image = transform(cropped_image)
    # 添加一个批次维度，因为模型期望批次维度
    processed_image = processed_image.unsqueeze(0).to('cuda')
    with torch.no_grad():  # 在不计算梯度的情况下进行前向传播
        features = model(processed_image)
    
    return features.view(1,-1)[:,:1000]

dataset = BCCDDataset(root_dir='BCCD')
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
def gen_feature():

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
    model1_feature = []
    for images, targets,path in dataset_loader:

        model1_bboxes = model_inference(model1, path[0], targets['boxes'].size()[0])

        for model1_bbox in model1_bboxes[0]['boxes']:
            features1 = extract_features(path[0], model1_bbox, back_model, transform)
            model1_feature.append(features1.to('cpu'))
    df = pd.DataFrame(model1_feature)
    df.to_csv('faster_feature_list.csv', index=False)

    model2_feature = []
    for images, targets,path in dataset_loader:
        model2_bboxes = model_inference(model2, path[0], targets['boxes'].size()[0])
        for model2_bbox in model2_bboxes[0]['boxes']:
            features2 = extract_features(path[0], model2_bbox, back_model, transform)
            model2_feature.append(features2.to('cpu'))
           
    df = pd.DataFrame(model2_feature)
    df.to_csv('fcos_feature_list.csv', index=False)

    model3_feature = []
    for images, targets,path in dataset_loader:
        model3_bboxes = model_inference(model3, path[0], targets['boxes'].size()[0])
        for model3_bbox in model3_bboxes[0]['boxes']:
            features3 = extract_features(path[0], model3_bbox, back_model, transform)
            model3_feature.append(features3.to('cpu'))
           
    df = pd.DataFrame(model3_feature)
    df.to_csv('retina_feature_list.csv', index=False)

    

def gen_gt_feature():
        gt_feature = []
        i = 0
    
        for images, targets,path in dataset_loader:
            for detect_res in targets['boxes'][0]:

                    model1_bbox = detect_res
                    features1 = extract_features(path[0], [round(model1_bbox[0].item()),round(model1_bbox[1].item()), round(model1_bbox[0].item()) + round(model1_bbox[2].item()),round(model1_bbox[1].item()) + round(model1_bbox[3].item())], back_model, transform)           
                    gt_feature.append(features1.to('cpu'))
             
    


        df = pd.DataFrame(gt_feature)
        df.to_csv('gt_feature_list.csv', index=False)
### 生成feature
#gen_gt_feature()
#gen_feature()
#gen_intensity_feature()

### 计算feature GT真实距离
# file_paths = [
#         'faster_intensity_feature_list.csv',
#         'fcos_intensity_feature_list.csv',
#         'retina_intensity_feature_list.csv',
#         #'gt_feature_list.csv'
#     ]
# features,feature_length  = process_and_combine_tensors(file_paths)
# features = torch.tensor(features)
# distance_mdf_mtx_calculate(features,feature_length)
#dissimilarities = torch.tensor(euclidean_distances(features), dtype=torch.float32)
# nmds_train(features, feature_length, dissimilarities, mode = 'full' )
# print(triplet(22.2012,18.7173,22.4873))
