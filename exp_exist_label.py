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


if __name__ == "__main__":
    data_dir = "D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets"

    save_dir = os.path.join(data_dir, 'visual_table_structure')
    visualize_datasets(data_dir=data_dir, save_dir=save_dir)

