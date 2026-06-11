import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ایمپورت تابع کمکی از ماژول utils
from src.utils.utils import parse_xml_annotation

class FaceMaskDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root_dir, "annotations"))))

        # دیکشنری تبدیل متن به عدد
        self.class_map = {
            'with_mask': 1,
            'without_mask': 2,
            'mask_weared_incorrect': 3
        }

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 1. مسیر فایل عکس و XML
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        ann_path = os.path.join(self.root_dir, "annotations", self.annotations[idx])

        # 2. خواندن تصویر
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # تبدیل به RGB

        # نرمال‌سازی ساده (تبدیل پیکسل‌ها به بازه 0 تا 1)
        img = img.astype(np.float32) / 255.0

        # 3. خواندن باکس‌ها و لیبل‌ها 
        labels_text, boxes_list = parse_xml_annotation(ann_path)

        boxes = []
        labels = []

        for i in range(len(boxes_list)):
            label_text = labels_text[i]
            label_idx = self.class_map.get(label_text, 0) # اگر پیدا نکرد 0 (بک‌گراند)

            xmin, ymin, xmax, ymax = boxes_list[i]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_idx)

        # 4. تبدیل به Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # محاسبه مساحت باکس‌ها
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # هدف (Target)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        # تبدیل فرمت عکس به (Channel, Height, Width)
        img = torch.tensor(img).permute(2, 0, 1)

        if self.transforms:
            # اعمال Augmentation های احتمالی در آینده
            pass

        return img, target