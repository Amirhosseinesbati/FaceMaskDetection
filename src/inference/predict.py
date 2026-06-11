import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# ایمپورت ماژول‌های خودمون
from src.training.dataset import FaceMaskDataset
from src.training.model import get_model_instance_segmentation

# دیکشنری برای تبدیل عدد به متن
ID_TO_CLASS = {
    1: 'with_mask',
    2: 'without_mask',
    3: 'mask_weared_incorrect'
}

def load_trained_model(model_path, num_classes, device):
    """
    لود کردن مدل و وزن‌های ذخیره شده
    """
    print(f"[*] Loading model from {model_path}...")
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # حتما باید روی حالت eval باشه
    return model

def predict_and_plot(model, img_tensor, device, threshold=0.5):
    """
    دریافت تنسور تصویر، پیش‌بینی و رسم خروجی‌ها روی عکس
    """
    img_tensor = img_tensor.to(device)

    # پیش‌بینی
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # تبدیل تصویر از تنسور (C, H, W) به آرایه نامپای (H, W, C)
    img_display = img_tensor.cpu().permute(1, 2, 0).numpy().copy()
    
    # تبدیل رنگ برای OpenCV (چون با BGR کار میکنه ولی عکس ما RGB هست)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

    # استخراج خروجی‌ها
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # کشیدن باکس‌ها
    for i, score in enumerate(scores):
        if score > threshold:
            box = boxes[i].astype(int)
            label_id = labels[i]
            label_text = ID_TO_CLASS.get(label_id, 'Unknown')

            # تنظیم رنگ (سبز: با ماسک، قرمز: بی ماسک، زرد: غلط)
            if label_id == 1: 
                color = (0, 255, 0)
            elif label_id == 2: 
                color = (0, 0, 255) # BGR
            else: 
                color = (0, 255, 255)

            # رسم مستطیل
            cv2.rectangle(img_display, (box[0], box[1]), (box[2], box[3]), color, 2)

            # نوشتن متن + درصد اطمینان
            text = f"{label_text}: {score:.2f}"
            cv2.putText(img_display, text, (box[0], max(0, box[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # تبدیل مجدد به RGB برای نمایش درست در Matplotlib
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB) 
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_display)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # مسیرها
    processed_data_dir = os.path.join("data", "processed")
    model_path = os.path.join("models", "mask_detector_model.pth")
    
    if not os.path.exists(model_path):
        print("❌ Error: Trained model not found! Please run 'train.py' first.")
        exit()

    # 1. لود کردن مدل
    num_classes = 4
    model = load_trained_model(model_path, num_classes, device)

    # 2. لود کردن دیتاست برای تست (میتونید یک عکس دلخواه رو هم مستقیما به تنسور تبدیل کنید)
    print("[*] Loading dataset for testing...")
    dataset = FaceMaskDataset(processed_data_dir)
    
    # 3. انتخاب یک عکس دلخواه (مثلا عکس شماره 10)
    test_idx = 10
    print(f"[*] Making prediction on image index {test_idx}...")
    img_tensor, _ = dataset[test_idx]
    
    # 4. پیش‌بینی و نمایش
    predict_and_plot(model, img_tensor, device, threshold=0.6)