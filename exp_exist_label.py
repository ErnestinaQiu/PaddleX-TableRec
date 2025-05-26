"""
date: 2025/4/16
author: ErnestinaQiu
des: explore the label by plot table structure
"""
import os
import cv2
import math
import json
import logging
import random
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


def draw_tables(img: np.ndarray, boxes: list):
    """_summary_

    Args:
        img (np.ndarray): img matrix
        boxes (list): four points

    Returns:
        _type_: _description_
    """
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = image.height, image.width
    img_top = image.copy()
    img_bottom = cv2.cvtColor(np.ones((h, w, 3), dtype=np.uint8) * 255, cv2.COLOR_BGR2RGB)
    random.seed(0)

    draw_top = ImageDraw.Draw(img_top)
    for box in boxes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_top.polygon(box, fill=color)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bottom, [pts], True, color, 1)
    img_top = Image.blend(image, img_top, 0.5)
    img_show = Image.new("RGB", (w, h * 2), (255, 255, 255))
    img_show.paste(img_top, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_bottom), (0, h, w, h  * 2))
    return np.array(img_show)

def check_and_read(img_path):
    assert os.path.exists(img_path), "file is not exists"
    img = cv2.imread(img_path, 1)
    return img

def visualize_datasets(data_dir, save_dir, mode='train'):
    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_train.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'
    os.makedirs(save_dir, exist_ok=True)
    imgs_info = val['images']
    anns = val['annotations']
    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        img_path = os.path.join(imgs_dir, img_name)
        img_id = image_info['id']
        img = check_and_read(img_path=img_path)

        boxes = []
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            origin_x, origin_y, w, h = ann['bbox']
            box = [(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)]
            boxes.append(box)
        img_show = draw_tables(img, boxes)
        img_save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_save_path, img_show[:, :, ::-1])
        logging.info(f'out into {img_save_path}')


def scale(img, text_boxes, target_width, target_height):
    h, w, c = img.shape
    ratio_h = target_height / h
    ratio_w = target_width / w
    if ratio_h > ratio_w:
        ratio = ratio_w
    else:
        ratio = ratio_h

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    new_img = cv2.resize(img, dsize=(new_w, new_h))

    new_text_boxes = []
    for box in text_boxes:
        origin_x, origin_y, w, h = box
        new_x = origin_x * ratio
        new_y = origin_y * ratio
        new_w = w * ratio
        new_h = h * ratio
        new_poly_box = [(new_x, new_y), (new_x + new_w, new_y), (new_x + new_w, new_y + new_h), (new_x, new_y + new_h)]
        new_text_boxes.append(new_poly_box)
    return new_img, new_text_boxes


def test_scale(data_dir, save_dir, mode='train'):
    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_train.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'
    os.makedirs(save_dir, exist_ok=True)
    imgs_info = val['images']
    anns = val['annotations']
    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        img_path = os.path.join(imgs_dir, img_name)
        img_id = image_info['id']

        img = check_and_read(img_path=img_path)
        print(type(img))
        
        pts_boxes = []
        boxes = []
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            boxes.append(ann['bbox'])
            origin_x, origin_y, w, h = ann['bbox']
            box = [(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)]
            pts_boxes.append(box)

        img_show = draw_tables(img, pts_boxes)

        new_img, new_text_boxes = scale(img=img, text_boxes=boxes, target_height=255, target_width=255)
        new_img_show = draw_tables(new_img, new_text_boxes)

        new_img_name = 'scaled_img.png'
        new_img_path = os.path.join(save_dir, new_img_name)
        cv2.imwrite(new_img_path,new_img)

        new_table_show = 'table_img.png'
        new_table_path = os.path.join(save_dir, new_table_show)
        cv2.imwrite(new_table_path, new_img_show)

        break

if __name__ == "__main__":
    data_dir = "D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets"

    save_dir = os.path.join(os.getcwd(), 'output', 'tmp')
    os.makedirs(save_dir, exist_ok=True)
    # visualize_datasets(data_dir=data_dir, save_dir=save_dir)
    test_scale(data_dir=data_dir, save_dir=save_dir, mode='train')
