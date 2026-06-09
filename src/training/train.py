import os
import time
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset

# اضافه شدن MLflow و DagsHub
import mlflow
import mlflow.pytorch
import dagshub

from dataset import FaceMaskDataset
from model import get_model_instance_segmentation
from utils import collate_fn

# ==========================================
# تنظیمات DagsHub و MLflow
# نام کاربری و نام مخزن خود در DagsHub را اینجا وارد کنید
REPO_OWNER = "YOUR_DAGSHUB_USERNAME" 
REPO_NAME = "YOUR_DAGSHUB_REPO_NAME"

# اتصال خودکار به DagsHub
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
# ==========================================

def train_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[*] Training on device: {device}")

    processed_data_dir = os.path.join("..", "data", "processed")
    dataset = FaceMaskDataset(processed_data_dir)
    
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = Subset(dataset, indices[:-50])

    # تعریف هایپرپارامترها در یک دیکشنری برای ثبت راحت‌تر در MLflow
    hyperparams = {
        "batch_size": 2,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "num_epochs": 15,
        "optimizer": "SGD",
        "model_architecture": "Faster R-CNN ResNet50 FPN V2"
    }

    train_data_loader = DataLoader(
        dataset_train,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    num_classes = 4 
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=hyperparams["learning_rate"], 
                          momentum=hyperparams["momentum"], 
                          weight_decay=hyperparams["weight_decay"])
    scaler = amp.GradScaler()

    # ایجاد یک آزمایش (Experiment) جدید در MLflow
    mlflow.set_experiment("Face_Mask_Detection_Experiment")

    # شروع بلاک MLflow
    print("[*] Starting training loop with MLflow tracking...")
    with mlflow.start_run(run_name="Faster_RCNN_Run") as run:
        
        # ۱. ثبت هایپرپارامترها
        mlflow.log_params(hyperparams)
        
        for epoch in range(hyperparams["num_epochs"]):
            print(f"--- Epoch {epoch+1}/{hyperparams['num_epochs']} ---")
            start_time = time.time()
            
            model.train()
            epoch_loss = 0
            
            for i, (images, targets) in enumerate(train_data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                with amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += losses.item()

                # ۲. ثبت زنده (لحظه‌ای) خطای هر بچ در DagsHub
                if i % 10 == 0:
                    step = (epoch * len(train_data_loader)) + i
                    mlflow.log_metric("batch_loss", losses.item(), step=step)
                    print(f"  Batch {i}: Loss = {losses.item():.4f}")

            end_time = time.time()
            avg_loss = epoch_loss / len(train_data_loader)
            
            # ۳. ثبت خطای میانگین هر دوره (Epoch)
            mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)
            mlflow.log_metric("epoch_duration_sec", end_time - start_time, step=epoch)
            
            print(f"[+] Epoch {epoch+1} done | Average Loss: {avg_loss:.4f}\n")

        print("[*] Training Finished! 🎉")

        # ۴. ذخیره و ثبت مدل نهایی در Model Registry سرور DagsHub
        print("[*] Uploading model artifacts to DagsHub...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            #registered_model_name="Face_Mask_Production_Model" # نامی که در تب Model Registry قرار می‌گیرد
        )
        print("[*] Model successfully logged to MLflow and DagsHub!")

        # (اختیاری) ذخیره لوکال برای اطمینان
        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(models_dir, 'mask_detector_model.pth'))

if __name__ == "__main__":
    train_model()