import xml.etree.ElementTree as ET

def parse_xml_annotation(xml_dir: str):
    """
    تجزیه فایل XML و استخراج لیبل‌ها و مختصات باکس‌ها
    """
    tree = ET.parse(xml_dir)
    root = tree.getroot()

    labels = []
    bndboxes = []

    for obj in root.findall("object"):
        labels.append(obj.findtext("name"))
        x_min = int(obj.find("bndbox").findtext("xmin"))
        y_min = int(obj.find("bndbox").findtext("ymin"))
        x_max = int(obj.find("bndbox").findtext("xmax"))
        y_max = int(obj.find("bndbox").findtext("ymax"))
        bndboxes.append([x_min, y_min, x_max, y_max])

    return labels, bndboxes

def collate_fn(batch):
    """
    تابعی برای دیتالودر پایتورچ تا لیست‌ها را به درستی در یک Batch کنار هم قرار دهد
    """
    return tuple(zip(*batch))