import os
import mlflow
import mlflow.pytorch
import torch

# نام مدل ثبت‌شده در MLflow Model Registry
MODEL_NAME = os.getenv("MODEL_NAME", "face-mask-detector")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# آدرس Tracking سرور داگزهاب
DAGSHUB_USER = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")


def _configure_mlflow():
    """تنظیم اتصال به MLflow روی DagsHub."""
    if DAGSHUB_USER and DAGSHUB_REPO:
        tracking_uri = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        # احراز هویت با توکن
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        print(f"✅ MLflow tracking URI set to: {tracking_uri}")
    else:
        print("⚠️ DagsHub credentials not found. Using local MLflow.")


def load_production_model():
    """
    مدل با تگ Production را از MLflow Registry لود می‌کند.
    اگر در دسترس نبود، به فایل لوکال fallback می‌کند.
    """
    _configure_mlflow()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"📥 Loading model from registry: {model_uri}")
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        print("✅ Production model loaded successfully from MLflow Registry.")
    except Exception as e:
        print(f"⚠️ Could not load from registry ({e}). Trying local fallback...")
        local_path = os.path.join("models", "latest_model.pth")
        from src.training.model import get_model_instance_segmentation
        model = get_model_instance_segmentation(num_classes=4)
        model.load_state_dict(torch.load(local_path, map_location=device))
        print(f"✅ Loaded local fallback model from: {local_path}")

    model.to(device)
    model.eval()
    return model, device


# نگاشت اندیس به نام کلاس (مطابق dataset.py شما)
CLASS_NAMES = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect",
}

CLASS_COLORS = {
    1: (0, 200, 0),     # سبز: ماسک دارد
    2: (220, 0, 0),     # قرمز: ماسک ندارد
    3: (255, 165, 0),   # نارنجی: ماسک نادرست
}