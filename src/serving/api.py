import io
import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.serving.load_model import load_production_model, CLASS_NAMES

# نگهداری مدل در حافظه (Global State)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """در زمان استارت، مدل یک‌بار لود می‌شود و تا پایان زنده می‌ماند."""
    print("🚀 Loading model into memory...")
    model, device = load_production_model()
    ml_models["model"] = model
    ml_models["device"] = device
    yield
    ml_models.clear()
    print("🧹 Model cleared from memory.")


app = FastAPI(
    title="😷 Face Mask Detection API",
    description="Detect mask status using a Faster R-CNN model served from MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)

# اجازه دسترسی از Streamlit / فرانت‌اند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== مدل‌های Pydantic برای خروجی تمیز =====
class Detection(BaseModel):
    label: str
    label_id: int
    confidence: float
    box: list[float]  # [xmin, ymin, xmax, ymax]


class PredictionResponse(BaseModel):
    filename: str
    detections: list[Detection]
    count: int


# ===== توابع کمکی =====
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """تبدیل بایت‌های تصویر به تنسور آماده مدل (مطابق dataset.py)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1)
    return tensor


@torch.no_grad()
def run_inference(tensor: torch.Tensor, threshold: float = 0.5) -> list[Detection]:
    """اجرای مدل و فیلتر خروجی بر اساس آستانه اطمینان."""
    model = ml_models["model"]
    device = ml_models["device"]

    prediction = model([tensor.to(device)])[0]

    detections = []
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        score = float(score)
        if score < threshold:
            continue
        label_id = int(label)
        detections.append(
            Detection(
                label=CLASS_NAMES.get(label_id, "unknown"),
                label_id=label_id,
                confidence=round(score, 3),
                box=[round(float(c), 2) for c in box.tolist()],
            )
        )
    return detections


# ===== Endpoints =====
@app.get("/")
def root():
    return {"message": "Face Mask Detection API is running 🚀", "docs": "/docs"}


@app.get("/health")
def health_check():
    """برای Health Check در Docker / Kubernetes."""
    status = "ok" if "model" in ml_models else "model_not_loaded"
    return {"status": status, "device": str(ml_models.get("device", "n/a"))}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    """دریافت تصویر و برگرداندن نتایج تشخیص به صورت JSON."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes)
    detections = run_inference(tensor, threshold=threshold)

    return PredictionResponse(
        filename=file.filename,
        detections=detections,
        count=len(detections),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)