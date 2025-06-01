import os
import cv2
import yaml
import logging
from exp.data.ties_dataset import TiesDataSet
from exp_exist_label import draw_tables

from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


class TestDs:
    def __init__(self):
        config_path = os.path.join(os.getcwd(), 'exp/configs/ties.yml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.logger = get_logger(name='test_ds', log_file='./output/tmp/test_ds.log', log_level=logging.ERROR)

    def test_ties_ds(self):
        ties_ds = TiesDataSet(config=self.config, logger=self.logger, mode='train', seed=123)
        for x in ties_ds:
            assert type(x) == dict, f'type(x): {type(x)}'
            assert 'images' in x.keys(), f'x.keys(): {x.keys()}'
            assert 'cell_boxes' in x.keys(), f'x.keys(): {x.keys()}'
            assert 'ocr_res_boxes' in x.keys(), f'x.keys(): {x.keys()}'
            assert 'cell_adj_mats' in x.keys(), f'x.keys(): {x.keys()}'
            assert 'row_adj_mats' in x.keys(), f'x.keys(): {x.keys()}'
            assert 'col_adj_mats' in x.keys(), f'x.keys(): {x.keys()}'
            images = x['images']
            assert images.shape[0] == self.config['num_samples'], f'images.shape: {images.shape}, self.config["num_samples"]: {self.config["num_samples"]}'
            assert images.shape[1] == 3
            assert images.shape[2] == self.config['normalized_height']
            assert images.shape[3] == self.config['normalized_width']
            text_boxes = x['cell_boxes']
            assert len(text_boxes) == self.config['num_samples']
            # print(text_boxes)
            break

    def test_idx(self):
        ties_ds = TiesDataSet(config=self.config, logger=self.logger, mode='train', seed=123)
        img, boxes = ties_ds[1]
        pts = []
        for b in boxes:
            origin_x, origin_y, w, h = b
            pts.append([(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)])

        img_show = draw_tables(img, pts)
        cv2.imwrite(filename='./output/tmp/test_ds_idx.png', img=img_show)

    def test_cells_relations(self):
        ties_ds = TiesDataSet(config=self.config, logger=self.logger, mode='train', seed=123)
        save_dir = os.path.join(os.getcwd(), 'output', 'tmp')
        ties_ds.check_cells_relations(save_dir=save_dir)

    def test_ocr_res_box_relations(self):
        ties_ds = TiesDataSet(config=self.config, logger=self.logger, mode='train', seed=123)
        save_dir = os.path.join(os.getcwd(), 'output', 'tmp')
        ties_ds.check_res_box_relations(save_dir=save_dir)

if __name__ == "__main__":
    test_ds = TestDs()
    test_ds.test_ties_ds()
    # test_ds.test_idx()
    # test_ds.test_cells_relations()
    # test_ds.test_ocr_res_box_relations()
