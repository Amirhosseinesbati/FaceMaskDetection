import os
import time
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset

import mlflow
import mlflow.pytorch

from src.training.dataset import FaceMaskDataset
from src.training.model import get_model_instance_segmentation
from src.utils.utils import collate_fn

def run_training(data_dir: str, hyperparams: dict):
    """
    این تابع دقیقاً همان منطق آموزش شماست که از ماژول‌های دیگر استفاده می‌کند.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[*] Training on device: {device}")

    # 1. فراخوانی ماژول دیتاست
    dataset = FaceMaskDataset(data_dir)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = Subset(dataset, indices[:-50])

    train_data_loader = DataLoader(
        dataset_train,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2. فراخوانی ماژول مدل
    num_classes = 4 
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=hyperparams["learning_rate"], 
                          momentum=hyperparams["momentum"], 
                          weight_decay=hyperparams["weight_decay"])
    
    use_cuda = device.type == 'cuda'
    # 🌟 تغییر اول: اسکِیلر را با توجه به نسخه جدید و فقط برای GPU می‌سازیم
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    print("[*] Starting training loop with MLflow tracking...")
    
    # لاگ کردن پارامترها در MLflow
    mlflow.log_params(hyperparams)
    
    # حلقه آموزش (همان کدهای قبلی شما)
    for epoch in range(hyperparams["num_epochs"]):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # 🌟 تغییر دوم: استفاده از سینتکس جدید PyTorch 2.x
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

            if i % 10 == 0:
                step = (epoch * len(train_data_loader)) + i
                mlflow.log_metric("batch_loss", losses.item(), step=step)

        avg_loss = epoch_loss / len(train_data_loader)
        mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)
        # 🌟 با اضافه کردن flush=True مطمئن می‌شویم پرینت بلافاصله روی صفحه می‌آید
        print(f"[+] Epoch {epoch+1} done | Average Loss: {avg_loss:.4f}", flush=True)

    print("[*] Training Finished! 🎉")

    # ذخیره در MLflow (بدون تگ Production، برای بررسی دستی در UI)
    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")
    
    # ذخیره فایل فیزیکی
    save_path = os.path.join("models", "latest_model.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    return save_path