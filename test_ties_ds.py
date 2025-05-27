import os
import cv2
import yaml
from exp.data.ties_dataset import TiesDataSet
from exp_exist_label import draw_tables

from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


class TestDs:
    def __init__(self):
        config_path = os.path.join(os.getcwd(), 'exp/configs/ties.yml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.logger = get_logger(name='test_ds', log_file='./output/tmp/test_ds.log')

    def test_ties_ds(self):
        ties_ds = TiesDataSet(config=self.config, logger=self.logger, mode='train', seed=123)
        for x in ties_ds:
            assert type(x) == dict, f'type(x): {type(x)}'
            assert 'images' in x.keys(), f'x.keys(): {x.keys()}'
            assert 'text_boxes' in x.keys(), f'x.keys(): {x.keys()}'
            images = x['images']
            assert images.shape[0] == self.config['num_samples'], f'images.shape: {images.shape}, self.config["num_samples"]: {self.config["num_samples"]}'
            assert images.shape[2] == self.config['normalized_height']
            assert images.shape[3] == self.config['normalized_width']
            text_boxes = x['text_boxes']
            assert len(text_boxes) == self.config['num_samples']

    def test_idx(self):
        ties_ds = TiesDataSet(config=self.config, logger=self.logger, mode='train', seed=123)
        img, boxes = ties_ds[1]
        pts = []
        for b in boxes:
            origin_x, origin_y, w, h = b
            pts.append([(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)])

        img_show = draw_tables(img, pts)
        cv2.imwrite(filename='./output/tmp/test_ds_idx.png', img=img_show)


if __name__ == "__main__":
    test_ds = TestDs()
    test_ds.test_ties_ds()
    test_ds.test_idx()
