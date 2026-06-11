import torch
from src.training.model import get_model_instance_segmentation

def test_model_architecture():
    # 1. آیا مدل با تعداد کلاس درست ساخته می‌شود؟
    num_classes = 4 # Background, with_mask, without_mask, mask_weared_incorrect
    model = get_model_instance_segmentation(num_classes)
    
    # بررسی می‌کنیم که آیا لایه آخر به درستی تغییر کرده است
    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes

def test_model_forward_pass_cpu():
    # 2. آیا مدل می‌تواند یک داده فرضی (Dummy) را روی CPU پردازش کند و کرش نکند؟
    num_classes = 4
    model = get_model_instance_segmentation(num_classes)
    model.eval() # برای تست، مدل را در حالت eval قرار می‌دهیم
    
    # ساخت یک عکس فرضی 3 کاناله (RGB) با سایز 256x256
    dummy_image = [torch.rand(3, 256, 256)]
    
    try:
        with torch.no_grad():
            output = model(dummy_image)
        # خروجی Faster R-CNN یک لیست از دیکشنری‌هاست
        assert isinstance(output, list)
        assert "boxes" in output[0]
        assert "labels" in output[0]
        assert "scores" in output[0]
    except Exception as e:
        assert False, f"Model forward pass failed: {e}"