import io
import os
import time

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# آدرس API (در داکر کامپوز با نام سرویس صدا زده می‌شود)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ===== تنظیمات صفحه =====
st.set_page_config(
    page_title="Face Mask Detector | AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== استایل سفارشی (CSS) =====
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1c1f26; border-radius: 12px; padding: 15px; }
    h1 { background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button {
        background: linear-gradient(45deg, #0072ff, #00c6ff);
        color: white; border-radius: 10px; border: none;
        padding: 0.6rem 1.2rem; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# رنگ‌ها برای رسم باکس‌ها (BGR در OpenCV)
LABEL_STYLE = {
    "with_mask": {"color": (0, 200, 0), "emoji": "✅", "fa": "ماسک دارد"},
    "without_mask": {"color": (0, 0, 220), "emoji": "❌", "fa": "بدون ماسک"},
    "mask_weared_incorrect": {"color": (0, 165, 255), "emoji": "⚠️", "fa": "ماسک نادرست"},
}


# ===== توابع کمکی =====
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json().get("status") == "ok", r.json().get("device", "n/a")
    except Exception:
        return False, "n/a"


def draw_detections(image: Image.Image, detections: list) -> np.ndarray:
    """رسم باکس‌ها روی تصویر."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        style = LABEL_STYLE.get(det["label"], {"color": (200, 200, 200), "emoji": "❓"})
        color = style["color"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{style['emoji']} {det['label']} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===== Sidebar =====
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/protection-mask.png", width=80)
    st.title("⚙️ تنظیمات")

    threshold = st.slider("🎯 آستانه اطمینان (Confidence)", 0.1, 1.0, 0.5, 0.05)

    st.divider()
    st.subheader("📡 وضعیت سرویس")
    healthy, device = check_api_health()
    if healthy:
        st.success(f"API آنلاین است • دستگاه: `{device}`")
    else:
        st.error("API در دسترس نیست ❌")

    st.divider()
    st.caption("ساخته‌شده با ❤️ — Faster R-CNN + MLflow")


# ===== صفحه اصلی =====
st.title("😷 سامانه هوشمند تشخیص ماسک")
st.markdown("تصویر خود را آپلود کنید تا مدل **Faster R-CNN** وضعیت ماسک افراد را تشخیص دهد.")

uploaded_file = st.file_uploader(
    "📤 تصویر را اینجا بکشید یا انتخاب کنید",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ تصویر اصلی")
        st.image(image, use_container_width=True)

    if st.button("🔍 شروع تشخیص", use_container_width=True):
        with st.spinner("🧠 مدل در حال تحلیل تصویر است..."):
            start = time.time()
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    params={"threshold": threshold},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                elapsed = time.time() - start

                detections = result["detections"]
                annotated = draw_detections(image, detections)

                with col2:
                    st.subheader("🎯 نتیجه تشخیص")
                    st.image(annotated, use_container_width=True)

                # ===== متریک‌ها =====
                st.divider()
                with_mask = sum(d["label"] == "with_mask" for d in detections)
                without_mask = sum(d["label"] == "without_mask" for d in detections)
                incorrect = sum(d["label"] == "mask_weared_incorrect" for d in detections)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("👥 کل چهره‌ها", result["count"])
                m2.metric("✅ ماسک‌دار", with_mask)
                m3.metric("❌ بدون ماسک", without_mask)
                m4.metric("⚠️ نادرست", incorrect)

                st.caption(f"⏱️ زمان پردازش: {elapsed:.2f} ثانیه")

                # جدول جزئیات
                if detections:
                    with st.expander("📋 مشاهده جزئیات کامل"):
                        st.dataframe(
                            [
                                {
                                    "کلاس": d["label"],
                                    "اطمینان": f"{d['confidence']:.1%}",
                                    "Box": str([round(c) for c in d["box"]]),
                                }
                                for d in detections
                            ],
                            use_container_width=True,
                        )
                else:
                    st.warning("هیچ چهره‌ای با این آستانه تشخیص داده نشد. آستانه را کاهش دهید.")

            except requests.exceptions.RequestException as e:
                st.error(f"خطا در ارتباط با API: {e}")
else:
    st.info("👆 برای شروع، یک تصویر آپلود کنید.")