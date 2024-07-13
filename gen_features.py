##### STEP TWO: GENERATE FEATURES #####

import torch
import os
from dataset import BCCDDataset
import torchvision
from utils import model_inference
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50
import torchvision.transforms as transforms
import pandas as pd
import cv2
from sklearn.metrics import euclidean_distances
from torchvision.models import convnext_base
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 

def single_img_feature(path, bbox):
    return extract_features(path, bbox, back_model, transform)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_fn(batch):
    return tuple(zip(*batch))

back_model = resnet50(pretrained=True).to('cuda')
back_model.eval()
back_model.to(device)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((244)),
    transforms.ToTensor(),
])

def extract_features(image_path, bbox, model, transform):
    image = cv2.imread(image_path)
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[round(y_min):round(y_max), round(x_min):round(x_max)]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    processed_image = transform(cropped_image)
    processed_image = processed_image.unsqueeze(0).to('cuda')
    with torch.no_grad():
        features = model(processed_image)
    
    return features.view(1,-1)[:,:1000]


dataset = BCCDDataset(root_dir='BCCD')
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
def gen_feature(model1, model2, model3):
    model1_feature = []
    for _, targets,path in dataset_loader:
        model1_bboxes = model_inference(model1, path[0], targets['boxes'].size()[0])
        for model1_bbox in model1_bboxes[0]['boxes']:
            features1 = extract_features(path[0], model1_bbox, back_model, transform)
            model1_feature.append(features1.to('cpu'))
    df = pd.DataFrame(model1_feature)
    df.to_csv('faster_feature_list.csv', index=False)

    model2_feature = []
    for _, targets,path in dataset_loader:
        model2_bboxes = model_inference(model2, path[0], targets['boxes'].size()[0])
        for model2_bbox in model2_bboxes[0]['boxes']:
            features2 = extract_features(path[0], model2_bbox, back_model, transform)
            model2_feature.append(features2.to('cpu'))
           
    df = pd.DataFrame(model2_feature)
    df.to_csv('fcos_feature_list.csv', index=False)

    model3_feature = []
    for _, targets,path in dataset_loader:
        model3_bboxes = model_inference(model3, path[0], targets['boxes'].size()[0])
        for model3_bbox in model3_bboxes[0]['boxes']:
            features3 = extract_features(path[0], model3_bbox, back_model, transform)
            model3_feature.append(features3.to('cpu'))
           
    df = pd.DataFrame(model3_feature)
    df.to_csv('retina_feature_list.csv', index=False)

def gen_gt_feature():
    gt_feature = []    
    for _, targets,path in dataset_loader:
        for detect_res in targets['boxes'][0]:

            model1_bbox = detect_res
            features1 = extract_features(path[0], [round(model1_bbox[0].item()),round(model1_bbox[1].item()), round(model1_bbox[0].item()) + round(model1_bbox[2].item()),round(model1_bbox[1].item()) + round(model1_bbox[3].item())], back_model, transform)           
            gt_feature.append(features1.to('cpu'))
             
    


    df = pd.DataFrame(gt_feature)
    df.to_csv('gt_feature_list.csv', index=False)

def gen_features():
    if os.path.exists('faster_feature_list.csv'):
        if os.path.exists('fcos_feature_list.csv'):
            if os.path.exists('retina_feature_list.csv'):
                print('Feature list already exists.')
                return
            
    print('start generating features')
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
    gen_feature(model1,model2,model3)
    print('feature generation finish')

if __name__ == "__main__":
    gen_feature()