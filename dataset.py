import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import xml.etree.ElementTree as ET
from PIL import Image

class BCCDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else ToTensor()
        self.valid_imgs = []
        self.valid_annots = []

        imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        annots = list(sorted(os.listdir(os.path.join(root_dir, "annotations"))))

        # 筛选出包含至少一个RBC的图像和对应的标注文件
        for img, annot in zip(imgs, annots):
            annot_path = os.path.join(root_dir, "annotations", annot)
            tree = ET.parse(annot_path)
            root = tree.getroot()
            for member in root.findall('object'):
                if member[0].text == 'RBC':
                    self.valid_imgs.append(img)
                    self.valid_annots.append(annot)
                    break

    def __len__(self):
        return len(self.valid_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.valid_imgs[idx])
        annot_path = os.path.join(self.root_dir, "annotations", self.valid_annots[idx])
        
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        tree = ET.parse(annot_path)
        root = tree.getroot()
        
        boxes, labels = [], []
        for member in root.findall('object'):
            if member[0].text == 'RBC':
                xmin, ymin, xmax, ymax = [int(member[4][i].text) for i in range(4)]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # RBC class label
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        return img, target, img_path