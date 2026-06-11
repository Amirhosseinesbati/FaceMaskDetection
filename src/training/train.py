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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[*] Training on device: {device}", flush=True)

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
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    print("[*] Starting training loop with MLflow tracking...", flush=True)
    
    mlflow.log_params(hyperparams)
    
    # 🌟 اضافه کردن فلگ تست سریع
    is_fast_run = hyperparams.get("fast_dev_run", False)
    
    for epoch in range(hyperparams["num_epochs"]):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

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
                
            # چاپ وضعیت هر بچ (برای اینکه در لاگ گیت‌هاب ببینیم سیستم هنگ نکرده)
            print(f"   [+] Processed batch {i+1}/{len(train_data_loader)}", flush=True)

            # 🌟 جادوی Smoke Test: خروج از حلقه بعد از 2 بچ
            if is_fast_run and i >= 1:
                print("⚠️ [Smoke Test] Processed 2 batches. Stopping batch loop early!", flush=True)
                break

        # محاسبه میانگین لاس (برای لاگ) - در حالت CI عدد دقیقی نیست ولی مهم نیست
        avg_loss = epoch_loss / (i + 1)
        mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)
        print(f"[+] Epoch {epoch+1} done | Average Loss: {avg_loss:.4f}", flush=True)

        # 🌟 اطمینان از اینکه در حالت CI وارد اپوک‌های بعدی نمی‌شود
        if is_fast_run:
            print("⚠️ [Smoke Test] Stopping epoch loop early!", flush=True)
            break

    print("[*] Training Finished! 🎉", flush=True)

    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")
    
    save_path = os.path.join("models", "latest_model.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    return save_path