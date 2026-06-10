import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes: int):
    """
    دانلود مدل از پیش آموزش دیده Faster R-CNN و تغییر لایه آخر برای دیتاست ماسک
    """
    # لود کردن مدل Faster R-CNN روی ResNet50
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # تعداد ویژگی‌های ورودی لایه آخر (Box Predictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # جایگزین کردن لایه آخر با توجه به تعداد کلاس‌های پروژه ما
    # (کلاس‌ها: بک‌گراند + با ماسک + بدون ماسک + ماسک غلط)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model