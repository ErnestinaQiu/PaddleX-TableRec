"""
generate the labels and get data for ties
author: ErnestinaQiu
"""
import os
import cv2
import json
import random
import numpy as np
import paddle
from paddle.io import IterableDataset


class TiesDataSet(IterableDataset):
    def __init__(self, config, mode, logger, seed=None):
        super(TiesDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        self.num_samples = config['num_samples']
        dataset_dir = config['data_dir']
        self.imgs_dir = os.path.join(dataset_dir, 'images')
        if self.mode == 'train':
            self.anns_path = os.path.join(dataset_dir, "annotations", "instance_train.json")
        else:
            self.anns_path = os.path.join(dataset_dir, "annotations", "instance_val.json")
        self.normalized_height = config['normalized_height']
        self.normalized_width = config['normalized_width']
        self.seed = config['seed']

    def get_info(self):
        with open(self.anns_path, 'r', encoding='utf8') as f:
            val = json.load(f)
        cat = val['categories']
        assert len(cat) == 1, 'number of category is more than one, please check dataset'
        imgs_info = val['images']
        anns = val['annotations']
        return imgs_info, anns

    def __iter__(self):
        """For model training
        Returns:
            dict: {"images": paddle.Tensor|[b, c, h, w], "text_boxes": list|[[text boxes in one image], [...]]},
                    images with shape as [batch, channel, width, height],
                    text_box with shape [x1, y1, x2, y2]
        """
        random.seed(self.seed)
        imgs_info, anns = self.get_info()
        images = []
        text_boxes = []
        for i in range(self.num_samples):
            chosen_img_info = imgs_info[random.choice(range(len(imgs_info)))]
            img_id = chosen_img_info['id']
            file_name = chosen_img_info['file_name']
            img_path = os.path.join(self.imgs_dir, file_name)
            img = self.check_and_read(img_path=img_path)

            boxes = []
            for j in range(len(anns)):
                ann = anns[j]
                if ann['image_id'] != img_id:
                    continue
                boxes.append(ann['bbox'])

            new_img, new_boxes = self.scale(img=img, text_boxes=boxes, target_width=self.normalized_width, target_height=self.normalized_height)
            new_img = np.transpose(new_img, (2, 0, 1))
            new_img_tensor = paddle.ones(shape=(img.shape[2], self.normalized_height, self.normalized_width), dtype=paddle.float32) * 255
            new_img_tensor[:, :new_img.shape[0], :new_img.shape[1]] = new_img
            images.append(new_img_tensor)
            text_boxes.append(new_boxes)

        new_img_tensor = paddle.to_tensor(new_img_tensor, dtype=paddle.float32)

        return {'images': new_img_tensor, 'text_boxes': text_boxes}

    def check_and_read(self, img_path):
        assert os.path.exists(img_path), "file is not exists"
        img = cv2.imread(img_path, 1)
        return img

    def scale(self, img, text_boxes, target_width, target_height):
        """_summary_

        Args:
            img (numpy.ndarray): with shape (h, w, c)
            text_boxes (list): with boxes as [x, y, w, h] 
            target_width (int): for normalization
            target_height (int): for normalization

        Returns:
            new_img (paddle.tensor): img with max edge 
            new_text_boxes (paddle.tensor): 
        """
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


    def __getitem__(self, idx):

        return 

