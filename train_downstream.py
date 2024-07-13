import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torchvision.models.detection import fasterrcnn_resnet50_fpn,fcos_resnet50_fpn,retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from dataset import BCCDDataset

from sklearn.model_selection import train_test_split


def get_pseudo_label(new_boxes, data):
    res = {}
    res['boxes'] =  torch.tensor(new_boxes, device='cuda:0', dtype=torch.float32)
    res['labels'] = torch.tensor([1,1,1,1,1,1,1,1,1], device='cuda:0')
    res['image_id'] = torch.tensor(data[0]['image_id'], device='cuda:0')
    res['area'] = torch.tensor([0,0,0,0,0,0,0,0,0], device='cuda:0')
    res['iscrowd'] = torch.tensor([0,0,0,0,0,0,0,0,0], device='cuda:0')
    return res
def create_datasets(root_dir, train_size=0.05):

    dataset = BCCDDataset(root_dir=root_dir)
    
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        train_size = 0.8,
        test_size=0.2,
        random_state=42 
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset

train_set, val_set = create_datasets(root_dir='BCCD')
train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
best_performance_map = 0
def evaluate_model(model_name, model, save_index=0):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=False)
    with torch.no_grad():
        for images, targets, _ in test_loader:
            try:
                images = [img.to(device) for img in images]
                if isinstance(targets, dict):
                    targets = [{k: v.squeeze(0).to(device) for k, v in targets.items()}]
                else:
                    targets = [{k: v.squeeze(0).to(device) for k, v in t.items()} for t in targets]
                pred = model(images)
                metric.update(pred, targets)
            except Exception as e:
                print(e)
            
        result = metric.compute()
         
        model.train()
        if not os.path.exists(model_name):
            os.makedirs(model_name)
        return result
       
def get_model(type):
    if type == 'faster-rcnn':
        model = fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2) #background +1
        return model
    elif type == 'fcos':
        model = fcos_resnet50_fpn()
        return model
    elif type == 'retinanet':
        model = retinanet_resnet50_fpn()
        return model

def downstream_training(type ='retinanet'):
    with open('al3_pseudo_labels.json', 'r') as file:
        hash_table = json.load(file)
    model = get_model(type)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.000001)
    num_epochs = 30
    best_performance_map = 0
    for epoch in range(num_epochs):
                model.train()
                i = 0
                for images, targets,path in train_loader:
                    if i < 17:
                        if isinstance(targets, dict):
                            pseudo_label = [{k: v.squeeze(0).to(device) for k, v in targets.items()}]
                        else:
                            pseudo_label = [{k: v.squeeze(0).to(device) for k, v in t.items()} for t in targets]
                    else:
                        pseudo_label = get_pseudo_label(hash_table.get(path[0]),[targets])
                        pseudo_label = [pseudo_label]
                    

                    images = [img.to(device) for img in images] 
                    loss_dict = model(images, pseudo_label)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                result = evaluate_model(type,model,epoch)
                if result['map'] > best_performance_map:
                    best_performance_map = result['map']


    print('downstream best performance: ' + str(best_performance_map))

if __name__ == "__main__":
    downstream_training()