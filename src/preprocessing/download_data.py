import os
import shutil
import kagglehub

def setup_kaggle_credentials():
    """
    تنظیم API Token کگل به عنوان متغیرهای محیطی.
    در پروژه‌های واقعی بهتر است این مقادیر از یک فایل .env خوانده شوند.
    """
    # مقادیر زیر را با اطلاعات داخل فایل kaggle.json خود جایگزین کنید
    # یا قبل از اجرای برنامه، این متغیرها را در ترمینال ست کنید
    
    os.environ['KAGGLE_USERNAME'] = "YOUR_KAGGLE_USERNAME"
    os.environ['KAGGLE_KEY'] = "YOUR_KAGGLE_API_TOKEN"
    
    # بررسی می‌کنیم که آیا توکن‌ها تنظیم شده‌اند یا خیر
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("⚠️ Warning: Kaggle API Tokens are not set in environment variables.")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY.")
    else:
        print("✅ Kaggle API credentials loaded via Environment Variables.")

def download_and_zip_kaggle_dataset(dataset_handle: str, raw_dir: str, zip_filename: str = "face-mask-dataset"):
    # تنظیم اعتبارنامه قبل از دانلود
    setup_kaggle_credentials()
    
    print(f"Downloading dataset '{dataset_handle}' from Kaggle...")
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        # کگل‌هاب به صورت خودکار از متغیرهای محیطی KAGGLE_USERNAME و KAGGLE_KEY استفاده می‌کند
        dataset_path = kagglehub.dataset_download(dataset_handle)
        print(f"Dataset downloaded successfully to cache: {dataset_path}")
        
        zip_path_without_ext = os.path.join(raw_dir, zip_filename)
        print(f"Zipping the dataset to {raw_dir}...")
        
        shutil.make_archive(zip_path_without_ext, 'zip', dataset_path)
        print(f"Dataset successfully zipped and saved to: {zip_path_without_ext}.zip")
        
    except Exception as e:
        print(f"An error occurred during download or zipping: {e}")

if __name__ == "__main__":
    DATASET_HANDLE = "andrewmvd/face-mask-detection"
    RAW_DATA_DIR = os.path.join("data", "raw")
    
    download_and_zip_kaggle_dataset(DATASET_HANDLE, RAW_DATA_DIR)