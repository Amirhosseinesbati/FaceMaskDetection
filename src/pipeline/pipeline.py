import os
from zenml import step, pipeline
from src.training.train import run_training

# ==========================================
# تعریف استپ آموزش
# ==========================================
# با enable_cache=False به زن‌ام‌ال می‌گوییم هر بار که دکمه را زدیم از نو آموزش بده
@step(experiment_tracker="dagshub_mlflow_tracker", enable_cache=False)
def train_model_step(data_dir: str, hyperparams: dict) -> str:
    print("⏳ ZenML Step Started: Executing PyTorch Training...")
    
    # صدا زدن منطق آموزش
    model_path = run_training(data_dir=data_dir, hyperparams=hyperparams)
    
    return model_path

# ==========================================
# تعریف پایپ‌لاین
# ==========================================
@pipeline
def mask_detection_training_pipeline(data_dir: str, hyperparams: dict):
    train_model_step(data_dir=data_dir, hyperparams=hyperparams)

# ==========================================
# نقطه شروع اجرای اسکریپت
# ==========================================
if __name__ == "__main__":
    print("🚀 Python script started successfully!")
    
    # تنظیم مسیرها و پارامترها
    DATA_DIR = os.path.join("data", "processed")
    
    HYPERPARAMS = {
        "batch_size": 2,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "num_epochs": 15,  # برای تست میتونی موقتاً بذاری روی 1 یا 2
        "optimizer": "SGD",
        "model_architecture": "Faster R-CNN ResNet50 FPN V2"
    }

    # راه‌اندازی پایپ‌لاین
    print("🔥 Handing over execution to ZenML Orchestrator...")
    mask_detection_training_pipeline(data_dir=DATA_DIR, hyperparams=HYPERPARAMS)
    print("✅ Pipeline execution command finished.")