import subprocess
import json
import os
import sys
from dotenv import load_dotenv

# ==========================================
# 🛠️ بخش ۱: بارگذاری و اعتبارسنجی تنظیمات
# ==========================================
def load_environment():
    """خواندن متغیرهای محیطی از فایل .env و بررسی صحت آن‌ها"""
    load_dotenv()
    
    config = {
        "VAST_API_KEY": os.getenv("VAST_API_KEY"),
        "DAGSHUB_TOKEN": os.getenv("DAGSHUB_USER_TOKEN"),
        "DAGSHUB_USERNAME": os.getenv("DAGSHUB_REPO_OWNER"),
        "DAGSHUB_TRACKING_URI": os.getenv("DAGSHUB_TRACKING_URI"),
        "GIT_REPO_URL": os.getenv("GIT_REPO_URL"),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", "main"), # اگر برنچ تنظیم نبود، main در نظر می‌گیرد
        "GPU_TARGET": os.getenv("GPU_TARGET", "RTX_3060"),
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME"),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY")
    }
    
    # بررسی اینکه آیا متغیری خالی مانده است یا خیر
    missing_vars = [key for key, value in config.items() if not value]
    if missing_vars:
        print(f"❌ Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all variables are set.")
        sys.exit(1)
        
    return config

# ==========================================
# 🚀 بخش ۲: توابع کمکی (اجرای دستورات)
# ==========================================
def run_command(command, return_output=False, silent_error=False):
    """Execute shell commands safely, capturing raw bytes and decoding as UTF-8.
    Decoding uses 'replace' for invalid sequences to avoid 'charmap' codec errors.
    Also injects UTF-8 environment variables for subprocesses.
    """
    try:
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env.setdefault('LC_ALL', 'C.UTF-8')
        env.setdefault('LANG', 'C.UTF-8')

        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env
        )

        output_bytes = result.stdout or b''
        # Try UTF-8 decode, fall back to replacement to avoid decode errors
        try:
            output = output_bytes.decode('utf-8')
        except Exception:
            output = output_bytes.decode('utf-8', errors='replace')

        output = output.strip()

        if result.returncode != 0:
            if not silent_error:
                print(f"\n🛑 COMMAND FAILED: {command}")
                print(f"--- Error Details ---\n{output}\n---------------------")
            sys.exit(1)

        return output if return_output else None
    except Exception as e:
        print(f"\n❌ Subprocess execution failed: {e}")
        sys.exit(1)

# ==========================================
# 🎯 بخش ۳: منطق اصلی برنامه (دیپلوی)
# ==========================================
def main():
    # ۱. بررسی وجود فایل اسکریپت لینوکس
    if not os.path.exists("setup_vast.sh"):
        print("❌ Error: 'setup_vast.sh' not found in the current directory!")
        sys.exit(1)

    # ۲. دریافت تنظیمات
    config = load_environment()
    
    print(f"🔍 Searching for the cheapest {config['GPU_TARGET']} on Vast.ai...")
    
    # ۳. لاگین به اکانت (بدون چاپ شدن کلید API در صورت بروز خطا)
    run_command(f"vastai set api-key {config['VAST_API_KEY']}", silent_error=True)

    # ۴. جستجوی سرور مناسب
    search_cmd = f"vastai search offers \"gpu_name={config['GPU_TARGET']} num_gpus=1\" -o dph --raw"
    raw_json = run_command(search_cmd, return_output=True)
    
    try:
        offers = json.loads(raw_json)
        if not offers:
            print(f"❌ No {config['GPU_TARGET']} found! Try a different GPU.")
            sys.exit(1)
            
        instance_id = str(offers[0]['id'])
        price = offers[0]['dph_total']
        print(f"✅ Found cheapest instance! ID: {instance_id} | Price: ${price:.3f}/hour")
    except Exception as e:
        print(f"❌ Failed to parse Vast.ai output: {e}")
        sys.exit(1)

    print(f"🚀 Renting instance {instance_id} and injecting setup script...")

    # ۵. ایجاد لیست متغیرهای محیطی سرور
    env_vars_string = (
        f"-e VAST_API_KEY={config['VAST_API_KEY']} "
        f"-e INSTANCE_ID={instance_id} "
        f"-e DAGSHUB_TOKEN={config['DAGSHUB_TOKEN']} "
        f"-e DAGSHUB_USERNAME={config['DAGSHUB_USERNAME']} "
        f"-e DAGSHUB_TRACKING_URI={config['DAGSHUB_TRACKING_URI']} "
        f"-e GIT_REPO_URL={config['GIT_REPO_URL']} "
        f"-e GIT_BRANCH={config['GIT_BRANCH']} "
        f"-e KAGGLE_USERNAME={config['KAGGLE_USERNAME']} "
        f"-e KAGGLE_KEY={config['KAGGLE_KEY']}"
    )

    # ۶. اجرای دستور ساخت سرور
    create_cmd = (
        f"vastai create instance {instance_id} "
        f"--image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime "
        f"--disk 20 "
        f"--env \"{env_vars_string}\" "
        f"--onstart setup_vast.sh "
        f"--raw"
    )

    create_output = run_command(create_cmd, return_output=True)
    
    # ۷. بررسی امنیتی پاسخ ساخت سرور (تشخیص خطاهای پنهان)
    try:
        response_json = json.loads(create_output)
        if response_json.get("error"):
            print(f"\n❌ Vast.ai Failed: {response_json.get('msg')}")
            sys.exit(1)
    except json.JSONDecodeError:
        pass # در صورتی که پاسخ، JSON معتبری نبود نادیده می‌گیریم
        
    if not create_output.strip():
        print("\n⚠️ WARNING: Vast.ai returned an EMPTY response.")
        sys.exit(1)
    
    # ۸. پایان موفقیت‌آمیز
    print("\n🎉 Magic initiated! The server is rented successfully.")
    print("It will setup, train the model, save logs to DagsHub, and DESTROY itself automatically.")

if __name__ == "__main__":
    main()