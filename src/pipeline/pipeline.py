import os
from zenml import step, pipeline

# ایمپورت کردن ماژول‌های باکیفیت خودتان
from src.preprocessing.download_data import download_and_zip_kaggle_dataset
from src.preprocessing.make_dataset import extract_dataset, verify_extracted_data
from src.training.train import run_training

# ==========================================
# استپ ۱: دانلود داده‌ها
# ==========================================
@step(enable_cache=True)
def download_data_step(dataset_handle: str) -> str:
    print("📥 Starting Data Download Step...")
    raw_dir = os.path.join("data", "raw")
    zip_filename = "archive"
    
    # فراخوانی تابع خودتان
    download_and_zip_kaggle_dataset(dataset_handle=dataset_handle, raw_dir=raw_dir, zip_filename=zip_filename)
    
    # خروجی این استپ، مسیر فایل زیپ است تا به استپ بعدی برود
    zip_path = os.path.join(raw_dir, f"{zip_filename}.zip")
    return zip_path

# ==========================================
# استپ ۲: استخراج و آماده‌سازی داده‌ها
# ==========================================
@step(enable_cache=True)
def prepare_dataset_step(zip_filepath: str) -> str:
    print("📦 Starting Data Extraction Step...")
    processed_dir = os.path.join("data", "processed")
    
    # فراخوانی توابع خودتان
    extract_dataset(zip_filepath=zip_filepath, processed_dir=processed_dir)
    verify_extracted_data(processed_dir=processed_dir)
    
    # خروجی این استپ، مسیر پوشه داده‌های آماده برای آموزش است
    return processed_dir

# ==========================================
# استپ ۳: آموزش شبکه عصبی
# ==========================================
@step(experiment_tracker="dagshub_mlflow_tracker", enable_cache=False)
def train_model_step(data_dir: str, hyperparams: dict) -> str:
    print("⏳ Starting Model Training Step...")
    model_path = run_training(data_dir=data_dir, hyperparams=hyperparams)
    return model_path

# ==========================================
# پایپ‌لاین نهایی (ارتباط ماژول‌ها)
# ==========================================
@pipeline
def mask_detection_training_pipeline(dataset_handle: str, hyperparams: dict):
    # 1. دانلود
    zip_path = download_data_step(dataset_handle=dataset_handle)
    
    # 2. اکسترکت (ورودی: خروجی مرحله ۱)
    processed_path = prepare_dataset_step(zip_filepath=zip_path)
    
    # 3. آموزش (ورودی: خروجی مرحله ۲)
    train_model_step(data_dir=processed_path, hyperparams=hyperparams)


if __name__ == "__main__":
    print("🚀 Python script started successfully!")
    
    DATASET_HANDLE = "andrewmvd/face-mask-detection"
    
    HYPERPARAMS = {
        "batch_size": 16, # روی سرور ابری می‌توانید بالاتر ببرید
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "num_epochs": 15,
        "optimizer": "SGD",
        "model_architecture": "Faster R-CNN"
    }

    print("🔥 Handing over execution to ZenML Orchestrator...")
    mask_detection_training_pipeline(dataset_handle=DATASET_HANDLE, hyperparams=HYPERPARAMS)