
import yaml
import os  
import xml.etree.ElementTree as ET  
  
def convert(size, box):  
    dw = 1. / size[0]  
    dh = 1. / size[1]  
    x = (box[0] + box[1]) / 2.0 - 1  
    y = (box[2] + box[3]) / 2.0 - 1  
    w = box[1] - box[0]  
    h = box[3] - box[2]  
    x = x * dw  
    w = w * dw  
    y = y * dh  
    h = h * dh  
    return [x, y, w, h]  
  
def yaml2yolo(yaml_path, class_names):  
    with open(yaml_path, 'r', encoding='gb18030') as f:
        yaml_data = yaml.safe_load(f)
    img_h, img_w = yaml_data["img_h"], yaml_data["img_w"]
    boxes = []
    for anno in yaml_data["annotations"]:  
        cls = anno["class_name"] 
        if cls not in class_names:  
            continue  
        cls_id = class_names[cls] if isinstance(class_names, dict) else class_names.index(cls)
        b = (anno['xmin'], anno['xmax'],  
             anno['ymin'], anno['ymax'])  
        bb = convert((img_w, img_h), b)  
        boxes.append([cls_id]+bb)
    return boxes
  
def xml2yolo(xml_file, class_names):  
    tree = ET.parse(xml_file)  
    root = tree.getroot()  
    size = root.find('size')  
    w = int(size.find('width').text)  
    h = int(size.find('height').text)  
    boxes = []
    for obj in root.iter('object'):  
        difficult = obj.find('difficult').text  
        cls = obj.find('name').text  
        if cls not in class_names:  
            print(cls)
            continue  
        cls_id = class_names[cls] if isinstance(class_names, dict) else class_names.index(cls)
        xmlbox = obj.find('bndbox')  
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),  
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))  
        bb = convert((w, h), b)  
        boxes.append([cls_id]+bb)
    return boxes

def readYolo(yolo_txt, class_names):
    num_classes = len(class_names)
    boxes = []
    with open(yolo_txt) as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 5:
                continue
            cls_id = int(line[0])
            if cls_id >= num_classes:
                continue
            box = [cls_id]+[float(value) for value in line[1:5]]
            boxes.append(box)
    return boxes

def readAnno(anno_file, class_names):
    if anno_file.endswith('.yaml'):
        return yaml2yolo(anno_file, class_names)
    elif anno_file.endswith('.xml'):
        return xml2yolo(anno_file, class_names)
    elif anno_file.endswith('.txt'):
        return readYolo(anno_file, class_names)
    return []

if __name__ == "__main__":
    class_names = [  
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',  
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',  
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',  
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',  
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',  
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',  
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',  
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',  
        'toothbrush'  
    ]
    
    # 调用函数，转换一个XML文件  
    yaml_file = '/home/mqr/Desktop/datasets/coco128/xml/train2017/000000000009.yaml'  
    yaml2yolo(yaml_file, class_names)


