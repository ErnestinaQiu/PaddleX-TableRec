"""
generate the labels and get data for ties
author: ErnestinaQiu
"""
import os
import gc
import cv2
import json
import random
import logging
import numpy as np
import paddle
from paddle.io import IterableDataset
from exp.ocr import TableOCR
from exp_exist_label import draw_tables


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
        elif self.mode == 'val':
            self.anns_path = os.path.join(dataset_dir, "annotations", "instance_val.json")
        else:
            raise ValueError(f'Not support {mode} for TiesDataSet mode.')
        self.normalized_height = config['normalized_height']
        self.normalized_width = config['normalized_width']
        self.max_vertices = config['max_vertices']
        self.seed = config['seed']
        imgs_info, anns = self.get_info()
        self.imgs_num = len(imgs_info)
        del imgs_info
        del anns
        gc.collect()

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
            dict: {"images": paddle.Tensor|(b, c, h, w),
                    "text_boxes": list|[[text boxes in one image], [...]],
                    "ocr_res_boxes": list|[[ocr result boxes in one image]],
                    "cell_adj_mat": paddle.Tensor|(b, max_vertices, max_vertices),
                    "row_adj_mat": paddle.Tensor|(b, max_vertices, max_vertices),
                    "col_adj_mat": paddle.Tensor|(b, max_vertices, max_vertices),
                    }
                    images with shape as [batch, channel, width, height],
                    text_box with shape [x1, y1, x2, y2]
        """
        random.seed(self.seed)

        table_ocr = TableOCR(log_level=logging.INFO, platform='pc')

        imgs_info, anns = self.get_info()
        images = []
        cell_boxes = []
        ocr_res_boxes = []
        cell_adj_mats = []
        row_adj_mats = []
        col_adj_mats = []
        for i in range(self.num_samples):
            chosen_img_info = imgs_info[random.choice(range(len(imgs_info)))]
            img_id = chosen_img_info['id']
            file_name = chosen_img_info['file_name']
            img_path = os.path.join(self.imgs_dir, file_name)

            img = self.check_and_read(img_path=img_path)
            boxes = []                     # [x, y, w, h]
            new_format_cell_boxes = []          # [x1, y1, x2, y2]
            for j in range(len(anns)):
                ann = anns[j]
                if ann['image_id'] != img_id:
                    continue
                boxes.append(ann['bbox'])
                x, y, w, h = ann['bbox']
                new_format_cell_boxes.append([x, y, x + w, y + h])

            # get norm image tensor
            ratio, norm_img, norm_cell_boxes = self.scale(img=img, text_boxes=boxes, target_width=self.normalized_width, target_height=self.normalized_height)
            norm_img = np.transpose(norm_img, (2, 0, 1))
            norm_img_tensor = np.ones(shape=(img.shape[2], self.normalized_height, self.normalized_width)) * 255
            norm_img_tensor[:, :norm_img.shape[1], :norm_img.shape[2]] = norm_img[:, :, :]

            # get relation matrix of ocr result boxes
            cells_rel = self.get_cells_relations(boxes=new_format_cell_boxes)

            res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path)

            # get relations among ocr result boxes
            res_box_rel = self.get_boxes_rels_according_to_cells_rels(res_boxes=res_boxes, cells_rel=cells_rel)

            # scale ocr res boxes
            norm_res_boxes = self.scale_by_ratio(boxes=res_boxes, ratio=ratio)

            images.append(norm_img_tensor)
            cell_boxes.append(norm_cell_boxes)
            ocr_res_boxes.append(norm_res_boxes)

            cell_adj_mat = np.zeros(shape=(self.max_vertices, self.max_vertices), dtype=np.int8)
            row_adj_mat = np.zeros(shape=(self.max_vertices, self.max_vertices), dtype=np.int8)
            col_adj_mat = np.zeros(shape=(self.max_vertices, self.max_vertices), dtype=np.int8)
            for k in res_box_rel.keys():
                same_cell_idxs = res_box_rel[k]['same_cell']
                for m in same_cell_idxs:
                    cell_adj_mat[int(k), int(m)] = 1
                cell_adj_mats.append(cell_adj_mat)

                same_row_idxs = res_box_rel[k]['same_row']
                for n in same_row_idxs:
                    row_adj_mat[int(k), int(n)] = 1
                row_adj_mats.append(row_adj_mat)

                same_col_idxs = res_box_rel[k]['same_col']
                for l in same_col_idxs:
                    col_adj_mat[int(k), int(l)] = 1
                col_adj_mats.append(col_adj_mat)

        print(f'self.num_samples: {self.num_samples}, len(cell_adj_mats): {len(cell_adj_mats)}')

        images = paddle.to_tensor(images, dtype=paddle.float32)
        cell_adj_mats = paddle.to_tensor(data=cell_adj_mats, dtype=paddle.float32)
        row_adj_mats = paddle.to_tensor(data=row_adj_mats, dtype=paddle.float32)
        col_adj_mats = paddle.to_tensor(data=col_adj_mats, dtype=paddle.float32)

        yield {'images': images, 'cell_boxes': cell_boxes, "ocr_res_boxes": ocr_res_boxes, "cell_adj_mats": cell_adj_mats, "row_adj_mats": row_adj_mats, "col_adj_mats": col_adj_mats}

    def __getitem__(self, idx: int):
        """return origin image and cell boxes belong to the index

        Args:
            idx (int): index of the train or val ds

        Returns:
            dict: 'image':  images(np.ndarray) with shape (h, w, c)
                  'cell_boxes': cell_boxes(list), (x1, y1, x2, y2)
                  'img_path': str, absolute path
        """
        imgs_info, anns = self.get_info()
        img_info = imgs_info[idx]
        file_name = img_info['file_name']
        img_id = img_info['id']
        img_path = os.path.join(self.imgs_dir, file_name)
        img = self.check_and_read(img_path=img_path)

        boxes = []
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            x, y, w, h = ann['bbox']
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            boxes.append([x1, y1, x2, y2])

        return {'image': img, 'cell_boxes': boxes, 'img_path': img_path}

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
            new_text_boxes (paddle.tensor): [x1, y1, x2, y2]
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
            new_poly_box = [int(new_x), int(new_y), int(new_x + new_w), int(new_y + new_h)]
            new_text_boxes.append(new_poly_box)
        return ratio, new_img, new_text_boxes

    def scale_by_ratio(self, boxes, ratio):
        """scale coordinates of boxes according to ratio

        Args:
            boxes (list): box with (x, y, w, h)
            ratio (float32)
        Returns:
            scaled_boxes (list): new box with (x, y, w, h)
        """
        new_boxes = []
        for box in boxes:
            origin_x, origin_y, origin_w, origin_h = box
            x = origin_x * ratio
            y = origin_y * ratio
            w = origin_w * ratio
            h = origin_h * ratio
            new_box = [int(x), int(y), int(x + w), int(y + h)]
            new_boxes.append(new_box)
        return new_boxes

    def get_cells_relations(self, boxes):
        """split cell boxes into different rows and columns

        Args:
            boxes (list): [x1, y1, x2, y2]
        Returns:
            boxes_rel (dict): {'index of box': {'same_row': list, 'same_col': list, 'box': [x1, y1, x2, y2]}}
                            'same_row': indexes of boxes which belong to the same row
                            'same_col': indexes of boxes which belong to the same col
        """
        boxes_rel = {}
        for i in range(len(boxes)):
            box = boxes[i]
            boxes_rel[str(i)] = {'same_row': [], 'same_col': [], 'box': box}
            for j in range(len(boxes)):
                tmp_box = boxes[j]
                # row, y
                if (box[1] <= tmp_box[1] and box[3] >= tmp_box[3]) or (box[1] >= tmp_box[1] and box[3] <= tmp_box[3]):
                    self.logger.debug(f'same row, box: {box}, tmp_box: {tmp_box}')
                    boxes_rel[str(i)]['same_row'].append(j)
                # col, x
                if (box[0] <= tmp_box[0] and box[2] >= tmp_box[2]) or (box[0] >= tmp_box[0] and box[2] <= tmp_box[2]):
                    self.logger.debug(f'same col, box: {box}, tmp_box: {tmp_box}')
                    boxes_rel[str(i)]['same_col'].append(j)
        return boxes_rel

    def check_cells_relations(self, save_dir, seed=123):
        imgs_info, anns = self.get_info()
        random.seed(seed)
        chosen_img_idx = random.choice(range(len(imgs_info)))
        save_dir = os.path.join(save_dir, str(chosen_img_idx), 'ocr_cell_box')
        os.makedirs(save_dir, exist_ok=True)
        info_dict = self.__getitem__(chosen_img_idx)
        img = info_dict['image']
        cell_boxes = info_dict['cell_boxes']
        cell_boxes_pts = []
        for box in cell_boxes:
            x1, y1, x2, y2 = box
            cell_boxes_pts.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        img_show = draw_tables(img, cell_boxes_pts)
        table_img_path = os.path.join(save_dir, 'origin_table_img.png')
        cv2.imwrite(table_img_path, img_show)

        boxes_rel = self.get_cells_relations(boxes=cell_boxes)
        self.logger.info(f'total cell boxes num is {len(cell_boxes)}')
        for i in boxes_rel.keys():
            same_rows = boxes_rel[i]['same_row']
            same_rows_pts = []
            for j in same_rows:
                x1, y1, x2, y2 = cell_boxes[j]
                same_rows_pts.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            row_img_show = draw_tables(img, same_rows_pts)
            sp = os.path.join(save_dir, '.'.join([f"{i}_same_row", "png"]))
            cv2.imwrite(sp, row_img_show)

            same_cols = boxes_rel[i]['same_col']
            same_cols_pts = []
            for k in same_cols:
                x1, y1, x2, y2 = cell_boxes[k]
                same_cols_pts.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            self.logger.info(f'cell box index: {i}\nsame rows: {same_rows}\nsame cols: {same_cols}')
            col_img_show = draw_tables(img, same_cols_pts)
            sp = os.path.join(save_dir, '.'.join([f"{i}_same_col", "png"]))
            cv2.imwrite(sp, col_img_show)

    def get_boxes_rels_according_to_cells_rels(self, res_boxes, cells_rel):
        """get relations dict of res_boxes according to cells relations

        Args:
            res_boxes (list): the ocr result of table img, box inside is [x, y, w, h]
            cells_rel (dict): the annotations of cells in table img, box inside is [x1, y1, x2, y2]
        Returns:
            res_box_rel (dict): {"index of res box": {"box": [x1, y1, x2, y2], "same_cell": [index of res box], "same_row": [index of res box], "same_col": [index of res box]}]}
        """
        cell_to_res_box = {k: {'box': cells_rel[k]['box'], 'res_box_idxs': []} for k in cells_rel.keys()}
        self.logger.debug(f'def get_boxes_rels_according_to_cells_rels: cells_rel: {cells_rel}')
        for i in cells_rel.keys():
            c_x1, c_y1, c_x2, c_y2 = cells_rel[i]['box']
            for j in range(len(res_boxes)):
                x, y, w, h = res_boxes[j]
                x1 = x
                x2 = x + w
                y1 = y
                y2 = y + h
                self.logger.debug(f'def get_boxes_rels_according_to_cells_rels: cell{i} {cells_rel[i]["box"]}, res box{j} {[x1, y1, x2, y2]}')
                if x1 >= c_x1 and x2 <= c_x2 and y1 >= c_y1 and y2 <= c_y2:
                    cell_to_res_box[str(i)]['res_box_idxs'].append(j)
        self.logger.info(f'cell_to_res_box: {cell_to_res_box}')

        i = None
        j = None
        res_box_rel = {str(k): {'box': [res_boxes[k][0], res_boxes[k][1], res_boxes[k][0] + res_boxes[k][2], res_boxes[k][1] + res_boxes[k][3]], 'same_cell': [], 'same_row': [], 'same_col': []} for k in range(len(res_boxes))}

        for k in cell_to_res_box.keys():
            res_box_idxs = cell_to_res_box[k]['res_box_idxs']
            # same cell
            if len(res_box_idxs) > 1:
                for l in res_box_idxs:
                    for m in res_box_idxs:
                        res_box_rel[str(l)]['same_cell'].append(m)

            cell_rel = cells_rel[k]
            # same row
            same_row_cell_idxs = cell_rel['same_row']
            same_row_res_idxs = []
            for i in same_row_cell_idxs:
                same_row_res_idxs.extend(cell_to_res_box[str(i)]['res_box_idxs'])
            for j in res_box_idxs:
                res_box_rel[str(j)]['same_row'].extend(same_row_res_idxs)
            i = None
            j = None

            # same col
            same_col_cell_idxs = cell_rel['same_col']
            same_col_res_idxs = []
            for i in same_col_cell_idxs:
                same_col_res_idxs.extend(cell_to_res_box[str(i)]['res_box_idxs'])
            for j in res_box_idxs:
                res_box_rel[str(j)]['same_col'].extend(same_col_res_idxs)
            i = None
            j = None

        return res_box_rel

    def check_res_box_relations(self, save_dir, seed=123):
        table_ocr = TableOCR(log_level=logging.INFO, platform='pc')
        imgs_info, anns = self.get_info()
        random.seed(seed)
        chosen_img_idx = random.choice(range(len(imgs_info)))
        save_dir = os.path.join(save_dir, str(chosen_img_idx), 'ocr_res_box')
        os.makedirs(save_dir, exist_ok=True)
        info_dict = self.__getitem__(chosen_img_idx)
        img = info_dict['image']
        cell_boxes = info_dict['cell_boxes']

        cell_boxes_pts = []
        for box in cell_boxes:
            x1, y1, x2, y2 = box
            cell_boxes_pts.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        cell_boxes_img = draw_tables(img, cell_boxes_pts)
        cell_boxes_path = os.path.join(save_dir, 'cell_boxes.png')
        cv2.imwrite(cell_boxes_path, cell_boxes_img)

        img_path = info_dict['img_path']

        cells_rel = self.get_cells_relations(boxes=cell_boxes)
        res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path)

        res_boxes_pts = []
        for box in res_boxes:
            x, y, w, h = box
            res_boxes_pts.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        res_boxes_img = draw_tables(img, res_boxes_pts)
        res_boxes_path = os.path.join(save_dir, 'res_boxes.png')
        cv2.imwrite(res_boxes_path, res_boxes_img)

        res_boxes_rel = self.get_boxes_rels_according_to_cells_rels(res_boxes=res_boxes, cells_rel=cells_rel)

        self.logger.info(f'total res boxes num is {len(res_boxes)}\nres_boxes_rel: {res_boxes_rel}')

        for i in res_boxes_rel.keys():
            same_rows = res_boxes_rel[i]['same_row']
            same_rows_pts = []
            for j in same_rows:
                x1, y1, x2, y2 = res_boxes_rel[str(j)]['box']
                same_rows_pts.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            row_img_show = draw_tables(img, same_rows_pts)
            sp = os.path.join(save_dir, '.'.join([f"{i}_same_row", "png"]))
            cv2.imwrite(sp, row_img_show)

            same_cols = res_boxes_rel[i]['same_col']
            same_cols_pts = []
            for k in same_cols:
                x1, y1, x2, y2 = res_boxes_rel[str(k)]['box']
                same_cols_pts.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            self.logger.info(f'res box index: {i}\nsame rows: {same_rows}\nsame cols: {same_cols}')
            col_img_show = draw_tables(img, same_cols_pts)
            sp = os.path.join(save_dir, '.'.join([f"{i}_same_col", "png"]))
            cv2.imwrite(sp, col_img_show)

