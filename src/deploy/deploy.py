import subprocess
import json
import sys
import dotenv

# ==========================================
# 🛑 متغیرهای خود را اینجا وارد کنید 🛑
# ==========================================
dotenv.load_dotenv()
VAST_API_KEY = dotenv.get("VAST_API_KEY")
DAGSHUB_TOKEN = dotenv.get("DAGSHUB_USER_TOKEN")
DAGSHUB_USERNAME = dotenv.get("DAGSHUB_REPO_OWNER")
DAGSHUB_TRACKING_URI = dotenv.get("DAGSHUB_TRACKING_URI")
GIT_REPO_URL = dotenv.get("GIT_REPO_URL")
GPU_TARGET = dotenv.get("GPU_TARGET")
GIT_BRANCH = dotenv.get("GIT_BRANCH") # اگر GIT_BRANCH تعریف نشده بود، از main استفاده کن 
# ==========================================

def run_command(command, return_output=False):
    """تابع کمکی برای اجرای دستورات ترمینال در پایتون"""
    result = subprocess.run(command, shell=True, capture_output=return_output, text=True)
    if result.returncode != 0 and return_output:
        print(f"Error executing: {command}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip() if return_output else None

if __name__ == "__main__":
    print(f"🔍 Searching for the cheapest {GPU_TARGET} on Vast.ai...")
    
    # 1. لاگین به سیستم vast.ai در کامپیوتر شما
    run_command(f"vastai set api-key {VAST_API_KEY}")

    # 2. جستجوی ارزان‌ترین سرور (گرفتن خروجی به صورت JSON خام)
    search_cmd = f"vastai search offers \"gpu_name={GPU_TARGET} num_gpus=1\" -o dph --raw"
    raw_json = run_command(search_cmd, return_output=True)
    
    try:
        offers = json.loads(raw_json)
        if not offers:
            print(f"❌ No {GPU_TARGET} found! Try a different GPU.")
            sys.exit(1)
            
        instance_id = str(offers[0]['id'])
        price = offers[0]['dph_total']
        print(f"✅ Found cheapest instance! ID: {instance_id} | Price: ${price:.3f}/hour")
    except Exception as e:
        print(f"❌ Failed to parse Vast.ai output: {e}")
        sys.exit(1)

    print(f"🚀 Renting instance {instance_id} and injecting setup script...")

    # 3. ایجاد سرور ابری و ارسال متغیرهای محیطی + فایل setup_vast.sh
    env_vars = (
        f"-e VAST_API_KEY={VAST_API_KEY} "
        f"-e INSTANCE_ID={instance_id} "
        f"-e DAGSHUB_TOKEN={DAGSHUB_TOKEN} "
        f"-e DAGSHUB_USERNAME={DAGSHUB_USERNAME} "
        f"-e DAGSHUB_TRACKING_URI={DAGSHUB_TRACKING_URI} "
        f"-e GIT_REPO_URL={GIT_REPO_URL}"
        f"-e GIT_BRANCH={GIT_BRANCH}"
    )

    create_cmd = (
        f"vastai create instance {instance_id} "
        f"--image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime "
        f"--disk 10 "
        f"--env \"{env_vars}\" "
        f"--onstart setup_vast.sh"
    )

    run_command(create_cmd)
    
    print("🎉 Magic initiated! The server is rented.")
    print("It will setup, train the model, save logs to DagsHub, and DESTROY itself automatically.")