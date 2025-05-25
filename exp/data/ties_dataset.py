"""
generate the labels and get data for ties
author: ErnestinaQiu
"""
import os
import cv2
import json
import random
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
        """_summary_
        """
        random.seed(self.seed)
        imgs_info, anns = self.get_info()
        for i in range(self.num_samples):
            chosen_img_info = imgs_info[random.choice(range(len(imgs_info)))]
            img_id = chosen_img_info['id']
            file_name = chosen_img_info['file_name']
            img_path = os.path.join(self.imgs_dir, file_name)
            img = self.check_and_read(img_path=img_path)
            norm_img = paddle.vision.transforms.resize(img, size=(self.normalized_height, self.normalized_width))

            text_boxes = []
            for j in range(len(anns)):
                ann = anns[j]
                if ann['image_id'] != img_id:
                    continue
                origin_x, origin_y, w, h = ann['bbox']
                box = [(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)]
                text_boxes.append(box)
            

            

        return 
        

    def check_and_read(self, img_path):
        assert os.path.exists(img_path), "file is not exists"
        img = cv2.imread(img_path, 1)
        return img

    def __getitem__(self, idx):

        return 
        
