import os
import zipfile
import shutil

def extract_dataset(zip_filepath: str, processed_dir: str):
    """
    استخراج فایل زیپ از مسیر raw به مسیر processed
    """
    if not os.path.exists(zip_filepath):
        raise FileNotFoundError(f"Zip file not found at: {zip_filepath}")
    
    # اطمینان از وجود پوشه processed
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Extracting '{zip_filepath}' to '{processed_dir}'...")
    
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(processed_dir)
        print("Extraction completed successfully!")
        
    except zipfile.BadZipFile:
        print("Error: The file is not a valid zip archive.")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")

def verify_extracted_data(processed_dir: str):
    """
    بررسی اینکه آیا پوشه‌های تصاویر و XML ها به درستی اکسترکت شده‌اند
    """
    images_dir = os.path.join(processed_dir, "images")
    annotations_dir = os.path.join(processed_dir, "annotations")
    
    if os.path.exists(images_dir) and os.path.exists(annotations_dir):
        num_images = len(os.listdir(images_dir))
        num_xmls = len(os.listdir(annotations_dir))
        print("-" * 30)
        print("Data Verification Status: OK")
        print(f"Found {num_images} images and {num_xmls} XML annotations.")
        print("-" * 30)
    else:
        print("Warning: 'images' or 'annotations' directories not found in processed folder. Please check the zip structure.")

if __name__ == "__main__":
    # مسیرها
    ZIP_FILE_PATH = os.path.join("data", "raw", "archive.zip")
    PROCESSED_DATA_DIR = os.path.join("data", "processed")
    
    # 1. اکسترکت فایل
    extract_dataset(ZIP_FILE_PATH, PROCESSED_DATA_DIR)
    
    # 2. تایید سلامت فایل‌های اکسترکت شده
    verify_extracted_data(PROCESSED_DATA_DIR)