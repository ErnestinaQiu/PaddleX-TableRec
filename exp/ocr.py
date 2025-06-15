"""
author: ErnestinaQiu
description: get ocr result from PaddleOCR repos
"""
import os
import gc
import cv2
import random
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
from logging import NOTSET, DEBUG, INFO, ERROR
from typing import Any, List, Optional, Tuple
from paddlex import create_pipeline
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger
from paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2 import sort_table_cells_boxes


class TableOCR:
    def __init__(
        self,
        log_level=DEBUG,
        log_file='./output/exp/logs/debug.log',
        platform='aistudio',
        save_dir=None,
        md_path='./output/exp/XGB/20250610115334/xgb.pickle',
    ) -> None:
        """an ocr and its postprocess for table
        Args:
            log_level (int, optional):
            log_file (str, optional): 
            platform (str, optional): ['aistudio', 'pc']
            save_dir (str, optional): default None

        Returns:
            _type_: _description_
        """
        self.logger_flag = log_level
        self.platform = platform
        self.logger = get_logger(name='ocrtable', log_file=log_file, log_level=log_level)
        self.pipeline = create_pipeline(pipeline="OCR")
        # self.logger.info(dir(self.pipeline))
        # self.logger.info(dir(self.pipeline.text_det_model))
        # self.logger.info(dir(self.pipeline.text_rec_model))
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.model = pickle.load(open(md_path, 'rb'))

    def split_into_cell(self, ):
        return

    def get_ocr_text_boxes(self, img_path: str = None, save_dir: str = None):
        """get the text boxes of the ocr result of the img

        Args:
            img_path (str, optional): image path. Defaults to None.
            save_dir (str, optional): directory to save. Defaults to None.

        Returns:
            shrink_boxes (list): box [x, y, w, h]
        """
        if img_path:
            img_name = os.path.basename(img_path).split('.')[0]
            if save_dir is not None:
                save_dir = os.path.join(save_dir, 'ocr_text_box', img_name)
                os.makedirs(save_dir, exist_ok=True)
        ocr_res = self.get_img_ocr_result(img_path=img_path, save_dir=save_dir)
        ocr_res_json = ocr_res._to_json()['res']
        rec_boxes = ocr_res_json['rec_boxes']
        img = self.check_and_read_img(img_path=img_path)
        shrink_boxes = []
        for i in range(len(rec_boxes)):
            if self.logger_flag == DEBUG:
                self.logger.debug(" ".join(['-'*10, str(i), '-'*10]))
            box = rec_boxes[i]
            box_img = self.get_box_img(box=box, img=img)
            assert box_img.shape[0] != 0 and box_img.shape[1] != 0, f'box_img is empty, img_path: {img_path}'
            shrink_box_img, shrink_box = self.shrink_text_box(box_img=box_img, origin_box=box)

            if self.logger_flag == NOTSET:
                shrink_box_img = img[shrink_box[1]: shrink_box[1] + shrink_box[3], shrink_box[0]: shrink_box[0] + shrink_box[2]]
                if self.save_dir:
                    box_sp = os.path.join(self.save_dir, 'text_box.png')
                    shrink_box_sp = os.path.join(self.save_dir, 'shrink_box.png')
                else:
                    box_sp = None
                    shrink_box_sp = None
                self.show_img(img=box_img, sp=box_sp)
                self.show_img(img=shrink_box_img, sp=shrink_box_sp)
            if save_dir or self.save_dir and self.logger_flag <= DEBUG:
                shrink_box_img = img[shrink_box[1]: shrink_box[1] + shrink_box[3], shrink_box[0]: shrink_box[0] + shrink_box[2]]
                assert shrink_box_img.shape[0] != 0 and shrink_box_img.shape[1] != 0, f'shrink_box_img is empty, img_path: {img_path}'

                if save_dir is None:
                    save_dir = self.save_dir
                box_img_name = '.'.join(['_'.join([img_name, str(i)]), 'png'])
                box_img_path = os.path.join(save_dir, box_img_name)
                if os.path.exists(box_img_path):
                    pass
                else:
                    cv2.imwrite(box_img_path, shrink_box_img)

            # shrink_boxes.append(shrink_box)

            new_shrink_boxes, new_box_imgs = self.modify_text_boxes(text_box=shrink_box, box_img=shrink_box_img)

            for k in range(len(new_shrink_boxes)):
                tmp_box = new_shrink_boxes[k]
                shrink_boxes.append(tmp_box)
                if save_dir or self.save_dir and self.logger_flag <= DEBUG:
                    tmp_box_img = img[tmp_box[1]: tmp_box[1] + tmp_box[3], tmp_box[0]: tmp_box[0] + tmp_box[2]]
                    # assert tmp_box_img != [], f'rec_boxes {i}, new_shrink_boxes {k}, tmp_box_img: {tmp_box_img}, tmp_box: {tmp_box}, img.shape: {img.shape}'
                    if save_dir is None:
                        save_dir = self.save_dir
                    box_img_name = '.'.join(['_'.join([img_name, str(i), 'modified', str(k)]), 'png'])
                    box_img_path = os.path.join(save_dir, box_img_name)
                    if os.path.exists(box_img_path):
                        pass
                    else:
                        try:
                            cv2.imwrite(box_img_path, tmp_box_img)
                        except Exception as e:
                            self.logger.debug(f'tmp_box_img: {tmp_box_img}, tmp_box: {tmp_box}, img.shape: {img.shape}')
                            raise e

        return shrink_boxes

    def show_img(self, img: np.ndarray, sp: str = None):
        if self.platform != 'aistudio':
            cv2.imshow('Image', img)

        # Wait for a key press and then close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if sp:
            cv2.imwrite(sp, img)
        return 0

    def get_box_img(self, box: List, img: np.ndarray):
        """_summary_

        Args:
            box (List): [x0, y0, x1, y1]
            img (np.ndarray): image matrix array

        Returns:
            _type_: _description_
        """
        box_img = img[box[1]:box[3], box[0]:box[2]]
        return box_img

    def check_and_read_img(self, img_path: str):
        assert os.path.exists(img_path), f"img_path doesn't exists \n{img_path}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def shrink_text_box(self, box_img: np.ndarray, origin_box: List):

        """only consider the table line is vertical or horizontal

        Args:
            box_img (np.ndarray): img
            origin_box (List): [x0, y0, x1, y1]
        Returns:
            shrink_box_img (np.ndarray): img after shrink
            shrink_box (List): [x, y, w, h]

        """
        line_thresh = 0.05 * box_img.shape[1]
        margin_thresh = 0.95 * box_img.shape[1]

        if self.logger_flag == DEBUG:
            self.logger.debug(f'origin_box: {origin_box}')
            self.logger.debug(f'--- ver \nline_thresh: {line_thresh}, margin_thresh: {margin_thresh}')

        _, binary_image = cv2.threshold(box_img, 127, 1, cv2.THRESH_BINARY)

        # vertical analysis
        vertical_accum = []
        for i in range(binary_image.shape[0]):
            vertical_accum.append(np.sum(binary_image[i, :]))

        if self.logger_flag == DEBUG:
            self.logger.debug(f'vertical_accum: {vertical_accum}')

        detail = {'margin': [], 'line': []}
        margin_st = -1
        line_st = -1
        for k in range(binary_image.shape[0]):
            if vertical_accum[k] < line_thresh:
                if margin_st != -1:
                    detail['margin'].append([margin_st, k - 1])
                    margin_st = -1
                if line_st == -1:
                    line_st = k
            elif vertical_accum[k] > margin_thresh:
                if line_st != -1:
                    detail['line'].append([line_st, k - 1])
                    line_st = -1
                if margin_st == -1:
                    margin_st = k
            else:
                if line_st != -1:
                    detail['line'].append([line_st, k - 1])
                    line_st = -1
                if margin_st != -1:
                    detail['margin'].append([margin_st, k - 1])
                    margin_st = -1

        if margin_st != -1:
            detail['margin'].append([margin_st, binary_image.shape[0]-1])          

        if len(detail['margin']) == 1 and len(detail['line']) == 0:
            return box_img, [origin_box[0], origin_box[1], origin_box[2] - origin_box[0], origin_box[3] - origin_box[1]]

        ver_main_scope_len = 0
        ver_main_scope = [0, 0]
        for j in range(len(detail['margin'])-1):
            text_line_st = detail['margin'][j][1]
            text_line_ed = detail['margin'][j+1][0]
            if text_line_ed - text_line_st >= ver_main_scope_len:
                ver_main_scope_len = text_line_ed - text_line_st
                ver_main_scope = [text_line_st, text_line_ed]

        if self.logger_flag == DEBUG:
            self.logger.debug(f'ver_main_scope: {ver_main_scope}, \ndetail: {detail}')

        if ver_main_scope == [0, 0]:
            ver_main_scope = [0, binary_image.shape[0]]
        vertical_shrink_box = binary_image[ver_main_scope[0]: ver_main_scope[1], :]

        # horizontal analysis
        line_thresh = 0.05 * vertical_shrink_box.shape[0]
        margin_thresh = 0.95 * vertical_shrink_box.shape[0]
        horizontal_accum = []
        for i in range(vertical_shrink_box.shape[1]):
            horizontal_accum.append(np.sum(vertical_shrink_box[:, i]))

        if self.logger_flag == DEBUG:
            self.logger.debug(f'--- hor  \nline_thresh: {line_thresh}, margin_thresh: {margin_thresh}\nhorizontal_accum: {horizontal_accum}')

        detail = {'margin': [], 'line': []}
        margins = []
        margin_st = -1
        line_st = -1
        for k in range(vertical_shrink_box.shape[1]):
            if horizontal_accum[k] < line_thresh:
                if margin_st != -1:
                    if k - 1 - margin_st > 0:
                        detail['margin'].append([margin_st, k - 1])
                        margins.append(k - 1 - margin_st)
                    margin_st = -1
                if line_st == -1:
                    line_st = k
            elif horizontal_accum[k] > margin_thresh:
                if line_st != -1:
                    detail['line'].append([line_st, k - 1])
                    line_st = -1
                if margin_st == -1:
                    margin_st = k
            else:
                if line_st != -1:
                    line_st = -1
                if margin_st != -1:
                    if k - 1 - margin_st > 0:
                        detail['margin'].append([margin_st, k - 1])
                        margins.append(k - 1 - margin_st)
                    margin_st = -1

        if self.logger_flag == DEBUG:
            self.logger.debug(f'detail: {detail}')

        if len(detail['margin']) == 1 and len(detail['line']) == 0:
            self.logger.debug(f'box_img.shape: {box_img.shape}, shrink_box: {[origin_box[0], origin_box[1] + ver_main_scope[0], origin_box[2], ver_main_scope[1] - ver_main_scope[0]]}')
            return box_img[ver_main_scope[0]:ver_main_scope[1], :], [origin_box[0], origin_box[1] + ver_main_scope[0], origin_box[2] - origin_box[0], ver_main_scope[1] - ver_main_scope[0]]

        hor_main_scope = [0, 0]
        hor_main_scope_len = 0

        if len(detail['line']) > 0:
            for n in range(len(detail['line'])):
                line = detail['line'][n]
                if n == 0 and line[0] != 0:
                    hor_main_scope = [0, line[0] - 1]
                    hor_main_scope_len = line[0] - 1
                elif n != 0:
                    if detail['line'][n][0] - 1 - (detail['line'][n-1][1] + 1) > hor_main_scope_len:
                        hor_main_scope = [detail['line'][n-1][1] + 1, detail['line'][n][0] - 1]
                        hor_main_scope_len = detail['line'][n][0] - 1 - (detail['line'][n-1][1] + 1)

        if hor_main_scope == [0, 0]:
            hor_main_scope = [0, box_img.shape[1]]

        if self.logger_flag == DEBUG:
            self.logger.debug(f'hor_main_scope: {hor_main_scope}')

        shrink_img = box_img[ver_main_scope[0]:ver_main_scope[1], hor_main_scope[0]:hor_main_scope[1]]
        shrink_box = [origin_box[0] + hor_main_scope[0], origin_box[1] + ver_main_scope[0], hor_main_scope[1] - hor_main_scope[0], ver_main_scope[1] - ver_main_scope[0]]

        return shrink_img, shrink_box

    def get_img_ocr_result(self, img_path: str = None, save_dir: str = None):
        assert os.path.exists(img_path), "img_path don't exist."
        output = self.pipeline.predict(
            img_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        if self.logger_flag <= DEBUG:
            for res in output:
                if save_dir:
                    img_name = os.path.basename(img_path)
                    os.makedirs(save_dir, exist_ok=True)
                    img_sp = os.path.join(save_dir, img_name)
                    json_sp = os.path.join(save_dir, ".".join([img_name, 'json']))
                    res.save_to_img(img_sp)
                    res.save_to_json(json_sp)

        return next(output)

    def box_to_four_coordinates(self, box: List):
        """transform box from [x, y, w, h] to four points, x is vertical axis, and y is horizontal axis

        Args:
            box (list): [x, y, w, h]
        Returns:
            pts (list): four points of polygon
        """
        origin_x, origin_y, w, h = box
        pts = [(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)]
        return pts

    def transform_x1y1x2y2_into_four_coordinates(self, ocr_box: List):
        """_summary_

        Args:
            ocr_box (List): [x0, y0, x1, y1]
        Returns:
            pts (List): four points of ocr box
        """
        x0, y0, x1, y1 = ocr_box
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return pts

    def ocr_box_canvas(self, text_boxes: List, img_shape: Tuple):
        """draw text boxes in a canvas

        Args:
            text_boxes (List): [x, y, width, height]
            img_shape (Tuple): img.shape
        Returns:
            canvas (np.ndarray): blank matrix with text box as 1
        """
        canvas = np.zeros(img_shape)
        if self.logger_flag <= DEBUG:
            vis_blank_canvas = np.zeros(img_shape)
        for box in text_boxes:
            x, y, w, h = box
            canvas[y:y+h, x:x+w] = 1
            if self.logger_flag == DEBUG:
                vis_blank_canvas[y:y+h, x:x+w] = 255
        if self.logger_flag <= DEBUG:
            save_dir = self.save_dir
            sp = None
            if save_dir:
                sp = os.path.join(save_dir, 'vis_blank_canvas.png')
            self.show_img(img=vis_blank_canvas, sp=sp)
        self.logger.debug(f'canvas.shape: {canvas.shape}')
        return canvas

    def analysis_canvas(self, canvas: np.ndarray, save_dir: str=None):
        import matplotlib.pyplot as plt

        row_projection = []
        for i in range(canvas.shape[0]):
            row_projection.append(np.sum(canvas[i, :]))

        row_inds = [int(ind) for ind in range(canvas.shape[0])]

        plt.figure(figsize=(8, 6))
        plt.bar(row_inds, row_projection, color='skyblue')
        plt.title('row_projection')
        plt.xlabel('row index')
        plt.ylabel('pixel sum')

        if save_dir is not None:
            row_sp = os.path.join(save_dir, 'row_proj.png')
            plt.savefig(row_sp, dpi=300)
        else:
            plt.show()
            plt.close()

        col_projection = []
        for j in range(canvas.shape[1]):
            col_projection.append(np.sum(canvas[:, j]))

        col_inds = [ind for ind in range(canvas.shape[1])]

        plt.figure(figsize=(8, 6))
        plt.bar(col_inds, col_projection, color='skyblue')
        plt.title('col_projection')
        plt.xlabel('col index')
        plt.ylabel('pixel sum')

        if save_dir is not None:
            col_sp = os.path.join(save_dir, 'col_proj.png')
            plt.savefig(col_sp, dpi=300)
        else:
            plt.show()
            plt.close()        

    def split_into_region(self, canvas: np.ndarray, text_boxes: List, img: np.ndarray = None, iou_thresh=0.8):
        """ split text boxes into region cell

        Args:
            canvas (np.ndarray): blank matrix with text box as 1
            text_boxes (List): the small text boxes, [[x, y, w, h], ...]
            img (np.ndarray): for debug
        Returns:
            region (List): list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...]}, ...]
        """
        subgraphs = self.split_into_subgraph(canvas=canvas, text_boxes=text_boxes, img=img)
        row_subgraphs = subgraphs['row']
        col_subgraphs = subgraphs['col']

        regions = []          # [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h]]}]
        i = 0
        j = 0

        for i in row_subgraphs.keys():
            r_y1, r_y2 = row_subgraphs[i]['scope']
            for j in col_subgraphs.keys():
                r_x1, r_x2 = col_subgraphs[j]['scope']
                rect2 = [r_x1, r_y1, r_x2, r_y2]
                region = {'bound': rect2, 'text_boxes': [], 'empty_cell': 0}
                for box in text_boxes:
                    x1, y1, w, h = box
                    x2 = x1 + w
                    y2 = y1 + h
                    rect1 = [x1, y1, x2, y2]
                    iou = compute_iou(box1=rect1, box2=rect2)
                    if iou >= iou_thresh:
                        region['text_boxes'].append(box)
                        self.logger.debug(f'row {i} col {j} inside text box: {rect1}, cell bound: {rect2}')
                if len(region['text_boxes']) == 0:
                    region['empty_cell'] = 1

                if self.logger_flag == DEBUG and img is not None:
                    from PIL import Image, ImageDraw
                    _region_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    _region_img = _region_canvas.copy()
                    random.seed(0)
                    draw_region_img = ImageDraw.Draw(_region_img)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    _bound = region['bound']
                    _region_pts = self.transform_x1y1x2y2_into_four_coordinates(_bound)
                    draw_region_img.polygon(_region_pts, fill=color)

                    _cell_text_boxes = region['text_boxes']
                    _p = 0
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    for _p in range(len(_cell_text_boxes)):
                        _box = _cell_text_boxes[_p]
                        _pts = self.box_to_four_coordinates(box=_box)
                        draw_region_img.polygon(_pts, fill=color)

                    tmp_region_img = np.array(Image.blend(_region_canvas, _region_img, 0.5))
                    tmp_region_sp = None
                    if self.save_dir is not None:
                        tmp_region_sp = os.path.join(self.save_dir, f'region_img_row_{i}_col_{j}.png')
                    self.show_img(img=tmp_region_img, sp=tmp_region_sp)
                    self.logger.info(f'finish tmp region, out into {tmp_region_sp}')
                    del tmp_region_img
                    del _region_canvas
                    gc.collect()

                regions.append(region)

        self.logger.debug(f'row_subgraphs: {row_subgraphs}\ncol_subgraphs: {col_subgraphs}\nregions: {regions}')

        if self.logger_flag == INFO and img is not None:
            from PIL import Image, ImageDraw, ImageFont
            region_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            region_img = region_canvas.copy()
            random.seed(0)
            draw_region_img = ImageDraw.Draw(region_img)

            i = 0
            for i in range(len(regions)):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                bound = regions[i]['bound']
                region_pts = self.box_to_four_coordinates(box=bound)
                draw_region_img.polygon(region_pts, fill=color)

                text_boxes = regions[i]['text_boxes']
                j = 0
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for j in range(len(text_boxes)):
                    box = text_boxes[j]
                    pts = self.box_to_four_coordinates(box=box)
                    draw_region_img.polygon(pts, fill=color)

            region_img = np.array(Image.blend(region_canvas, region_img, 0.5))
            region_img_sp = None
            if self.save_dir:
                region_img_sp = os.path.join(self.save_dir, 'region_img.png')
            self.show_img(img=region_img, sp=region_img_sp)
            self.logger.debug(f'finish region, out into {region_img_sp}')
            del region_img
            del region_canvas
            gc.collect()

        return regions

    def get_bounds(self, bin_canvas: np.ndarray, mode: str):
        """get margin bounds of row or column of one image

        Args:
            bin_canvas (np.ndarray): binary image to analyse
            mode (str): 'row'|'col'
        Returns:
            list: bounds [int, ...]
        """
        assert mode in ['row', 'col'], f'Mode {mode} is not supported yet'
        margin_bounds = []
        st = -1
        ed = -1
        margin_st = -1
        margin_ed = -1
        if mode == 'row':
            range_scope = bin_canvas.shape[0]
        elif mode == 'col':
            range_scope = bin_canvas.shape[1]

        for i in range(range_scope):
            if mode == 'row':
                proj = np.sum(bin_canvas[i, :])
            elif mode == 'col':
                proj = np.sum(bin_canvas[:, i])
            # start st point
            if proj != 0 and margin_st == -1:
                if i == 0:
                    st = i
                else:
                    st = i - 1
                margin_st = 0
                continue

            if proj == 0 and margin_st != -1:
                margin_st = i
                continue
            elif proj != 0 and margin_st > 0:
                margin_ed = i - 1
                margin_bounds.append([margin_st, margin_ed])
                margin_st = 0
                margin_ed = 0

            if proj != 0:
                ed = i

        margin_bounds = [int(np.mean(bound)) for bound in margin_bounds]

        margin_bounds.insert(0, st)
        margin_bounds.append(ed)

        return margin_bounds

    # repeat function of split into groups
    def deal_big_region_frame(self, region: List, img_shape: Tuple):
        """use info of region to generate frame, and then use the frame to add empty cell into region and modify region bound into cell bound
           use after method split_into_region

        Args:
            region (List): list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...]}, ...]
            img_shape (Tuple): image shape

        Returns:
            dict : {'region': region, 'row_bounds': row_bounds, 'col_bounds': col_bounds},
                    region (list), {'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 1|0}
                    row_bounds (list): [int, ...] from top to bottom
                    col_bounds (list): [int, ...] from left to right
        """
        region_boxes = []
        for d in region:
            x1, y1, x2, y2 = d['bound']
            region_boxes.append([x1, y1, x2 - x1, y2 - x1])
        canvas = self.ocr_box_canvas(region_boxes, img_shape=img_shape)

        # get row bound
        row_bounds = self.get_bounds(bin_canvas=canvas, mode='row')

        # get col bound
        col_bounds = self.get_bounds(bin_canvas=canvas, mode='col')

        for i in range(len(row_bounds) - 1):
            row_st = row_bounds[i]
            row_ed = row_bounds[i + 1]
            for j in range(len(col_bounds) - 1):
                col_st = col_bounds[j]
                col_ed = col_bounds[j + 1]
                tmp_cell_bound = [col_st, row_st, col_ed, row_ed]
                miss_cell_guard = True
                for q in range(len(region)):
                    r_bound = region[q]['bound']
                    iou = compute_iou(box1=r_bound, box2=tmp_cell_bound)
                    if iou >= 0.8:
                        miss_cell_guard = False
                        region[q]['bound'] = tmp_cell_bound
                if miss_cell_guard:
                    region.append({'bound': tmp_cell_bound, 'text_boxes': [], 'empty_cell': 1})

        for p in region:
            if 'empty_cell' not in p.keys():
                p['empty_cell'] = 0

        return {'region': region, 'row_bounds': row_bounds, 'col_bounds': col_bounds}

    def merge_and_split(self, regions: list, img: np.ndarray, iou_thresh=0.6):
        """split regions into cells

        Args:
            regions (List): list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]
            img (np.ndarray): image
        Returns:
            new_regions_info (List): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1, ...]
        """
        new_regions_info = []
        for i in range(len(regions)):
            text_boxes = regions[i]['text_boxes']
            empty_cell_flag = regions[i]['empty_cell']
            bound = regions[i]['bound']
            if len(text_boxes) == 1 or empty_cell_flag:
                new_regions_info.append(regions[i])
                continue

            boxes_rel = {}  # {'index of box': {'same_cell': [index of boxes|int], 'same_row': [index of boxes|int}, 'same_col': [index of boxes|int]}
            for n in range(len(text_boxes)):
                box1 = text_boxes[n]
                boxes_rel[str(n)] = {'same_cell': [], 'same_row': [], 'same_col': []}
                for m in range(n + 1, len(text_boxes)):
                    box2 = text_boxes[m]
                    rel = self.get_boxes_rel(box1=box1, box2=box2, res_boxes=text_boxes)
                    if rel == 0:
                        boxes_rel[str(n)]['same_cell'].append(m)
                    elif rel == 1:
                        boxes_rel[str(n)]['same_row'].append(m)
                    elif rel == 2:
                        boxes_rel[str(n)]['same_col'].append(m)
                    elif rel == 3:
                        continue

            # same cell merge
            same_cells_indexes = []
            for j in boxes_rel.keys():
                if len(boxes_rel[j]['same_cell']) == 0:
                    continue
                repeat_guard = False
                for k in range(len(same_cells_indexes)):
                    if int(j) in same_cells_indexes[k]:
                        repeat_guard = True
                        break
                if repeat_guard:
                    continue

                same_cell_idxs = boxes_rel[j]['same_cell']
                add_idxs = []
                for idx in same_cell_idxs:
                    tmp_idxes = boxes_rel[str(idx)]['same_cell']
                    for _idx in tmp_idxes:
                        if _idx in same_cell_idxs:
                            continue
                        add_idxs.append(idx)
                complete_same_cell_idxs = same_cell_idxs + add_idxs + [int(j)]
                same_cells_indexes.append(complete_same_cell_idxs)

            self.logger.info(f'same_cells_indexes: {same_cells_indexes}')
            # merge same cell
            cell_region_info = []         # [{'bound': [x1, y1, x2, y2], 'text_boxes_idxs': [], 'text_boxes': [x, y, w, h]}]
            deal_text_boxes_idxs = []
            text_boxes_to_cell_region_map = {}
            for q in range(len(same_cells_indexes)):
                cell_boxes_indexes = same_cells_indexes[q]
                x1 = 1e5
                y1 = 1e5
                x2 = -1
                y2 = -1
                same_cells_text_boxes = []
                for p in cell_boxes_indexes:
                    tmp_x1, tmp_y1, tmp_w, tmp_h = text_boxes[p]
                    same_cells_text_boxes.append(text_boxes[p])
                    tmp_x2 = tmp_x1 + tmp_w
                    tmp_y2 = tmp_y1 + tmp_h
                    x1 = min(x1, tmp_x1)
                    x2 = max(x2, tmp_x2)
                    y1 = min(y1, tmp_y1)
                    y2 = max(y2, tmp_y2)
                    text_boxes_to_cell_region_map[str(p)] = q
                cell_region_info.append({'bound': [x1, y1, x2, y2], 'box': [x1, y1, x2 - x1, y2 - y1], 'text_boxes_idxs': cell_boxes_indexes, 'text_boxes': text_boxes})
                deal_text_boxes_idxs.extend(cell_boxes_indexes)

            self.logger.info(f'cell_region_info: {cell_region_info}')

            if self.logger_flag == INFO and img is not None:
                from PIL import Image, ImageDraw
                for _q in cell_region_info:
                    _region_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    _region_img = _region_canvas.copy()
                    random.seed(0)
                    draw_region_img = ImageDraw.Draw(_region_img)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    _bound = _q['bound']
                    _region_pts = self.transform_x1y1x2y2_into_four_coordinates(_bound)
                    draw_region_img.polygon(_region_pts, fill=color)

                    _cell_text_boxes = _q['text_boxes']
                    _p = 0
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    for _p in range(len(_cell_text_boxes)):
                        _box = _cell_text_boxes[_p]
                        _pts = self.box_to_four_coordinates(box=_box)
                        draw_region_img.polygon(_pts, fill=color)

                    tmp_region_img = np.array(Image.blend(_region_canvas, _region_img, 0.5))
                    tmp_region_sp = None
                    if self.save_dir is not None:
                        tmp_region_sp = os.path.join(self.save_dir, f'merge_cell_big_region_{i}_cell_{q}.png')
                    self.show_img(img=tmp_region_img, sp=tmp_region_sp)
                    self.logger.info(f'finish merge cell region, out into {tmp_region_sp}')
                    del tmp_region_img
                    del _region_canvas
                    gc.collect()

            # get new cell relation map after cell merge, because the original relation is predict by text boxes
            q = 0
            p = 0
            for q in range(len(text_boxes)):
                if q in deal_text_boxes_idxs:
                    continue
                _x, _y, _w, _h = text_boxes[q]
                text_boxes_to_cell_region_map[q] = len(cell_region_info)
                cell_region_info.append({'bound': [_x, _y, _x + _w, _y + _h], 'box': text_boxes[q], 'text_boxes_idxs': [q], 'text_boxes': [text_boxes[q]]})

            if len(cell_region_info) == 1:
                new_regions_info.extend(cell_region_info)
                continue

            # if self.logger_flag == INFO and img is not None:
            #     from PIL import Image, ImageDraw
            #     for _q in cell_region_info:
            #         _region_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #         _region_img = _region_canvas.copy()
            #         random.seed(0)
            #         draw_region_img = ImageDraw.Draw(_region_img)
            #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #         _bound = _q['bound']
            #         _region_pts = self.transform_x1y1x2y2_into_four_coordinates(_bound)
            #         draw_region_img.polygon(_region_pts, fill=color)

            #         _cell_text_boxes = _q['text_boxes']
            #         _p = 0
            #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #         for _p in range(len(_cell_text_boxes)):
            #             _box = _cell_text_boxes[_p]
            #             _pts = self.box_to_four_coordinates(box=_box)
            #             draw_region_img.polygon(_pts, fill=color)

            #         tmp_region_img = np.array(Image.blend(_region_canvas, _region_img, 0.5))
            #         tmp_region_sp = None
            #         if self.save_dir is not None:
            #             tmp_region_sp = os.path.join(self.save_dir, f'merge_cell_big_region_{i}_cell_{q}.png')
            #         self.show_img(img=tmp_region_img, sp=tmp_region_sp)
            #         self.logger.info(f'finish merge cell region, out into {tmp_region_sp}')
            #         del tmp_region_img
            #         del _region_canvas
            #         gc.collect()

            # analyze row
            # #transform coordinate from absolute to relative
            q = 0
            inside_region_cell_boxes = []
            for q in cell_region_info:
                _x, _y, _x2, _y2 = q['bound']
                _w = _x2 - _x
                _h = _y2 - _y
                inside_region_cell_boxes.append([_x - bound[0], _y - bound[1], _w, _h])

            img_shape = img[bound[1]:bound[3], bound[0]:bound[2]].shape
            inside_region_canvas = self.ocr_box_canvas(text_boxes=inside_region_cell_boxes, img_shape=img_shape)

            row_subgraphs = self.row_analyse(canvas=inside_region_canvas)
            # #transform coordinate from relative to absolute
            q = 0
            for q in row_subgraphs.keys():
                row_subgraphs[q]['scope'] = [row_subgraphs[q]['scope'][0] + bound[1], row_subgraphs[q]['scope'][1] + bound[1]]

            # case 1
            if len(row_subgraphs) == 1:
                sorted_cell_region_info = list(sorted(cell_region_info, key=lambda x: x["bound"][0]))
                new_regions_info.extend(sorted_cell_region_info)

                # fresh bound
                new_split_cell_info = []

                q = 0
                for q in range(len(sorted_cell_region_info)):
                    tmp_cell_region = sorted_cell_region_info[q]
                    tmp_text_boxes = tmp_cell_region['text_boxes']
                    new_x1 = 2e10
                    new_y1 = 2e10
                    new_x2 = -1
                    new_y2 = -1

                    for box in tmp_text_boxes:
                        bx, by, bw, bh = box
                        new_x1 = min(new_x1, bx)
                        new_x2 = max(new_x2, bx + bw)
                        new_y1 = min(new_y1, by)
                        new_y2 = max(new_y2, by + bh)

                    # if q == 0:
                    #     new_x1 = bound[0]
                    # elif q == len(sorted_cell_region_info) - 1:
                    #     new_x2 = bound[2]
                    tmp_cell_region['bound'] = [new_x1, new_y1, new_x2, new_y2]
                    new_split_cell_info.append(tmp_cell_region)

                new_regions_info.extend(new_split_cell_info)
                continue
            else:
                # split rows
                p = 0
                q = 0
                m = 0
                for p in row_subgraphs.keys():
                    subgraph_scope = row_subgraphs[p]['scope']
                    row_subgraphs[p]['cells_info'] = []   # [{'cell_idx': int, 'cell_bound': [x1, y1, x2, y2], 'cell_box': [x, y, w, h], 'text_boxes_idxs': list of int|indexes of text boxes which belong to the same cell, 'text_boxes': []}]
                    row_subgraphs[p]['bound'] = [bound[0], subgraph_scope[0], bound[2], subgraph_scope[1]]
                    for q in range(len(cell_region_info)):
                        cell_bound = cell_region_info[q]['bound']
                        x, y, x1, y1 = cell_bound
                        h = y1 - y
                        if y >= subgraph_scope[0] and y + h <= subgraph_scope[1]:
                            row_subgraphs[p]['cells_info'].append({'cell_idx': q, 'cell_bound': cell_region_info[q]['bound'], 'text_boxes_idxs': cell_region_info[q]['text_boxes_idxs'], 'text_boxes': cell_region_info[q]['text_boxes']})
                        elif y + h <= subgraph_scope[0] or y >= subgraph_scope[1]:
                            pass
                        else:
                            iou = 0
                            if y >= subgraph_scope[0] and y < subgraph_scope[1] and y + h > subgraph_scope[1]:
                                iou = round((subgraph_scope[1] - y) / h, 2)
                            elif y < subgraph_scope[0] and y + h > subgraph_scope[0] and y + h <= subgraph_scope[1]:
                                iou = round((y + h - subgraph_scope[0]) / h, 2)
                            elif y <= subgraph_scope[0] and y + h >= subgraph_scope[1]:
                                iou = round((subgraph_scope[1] - subgraph_scope[0]) / h, 2)
                            if iou >= iou_thresh:
                                row_subgraphs[p]['cells_info'].append({'cell_idx': q, 'cell_bound': cell_region_info[q]['bound'], 'text_boxes_idxs': cell_region_info[q]['text_boxes_idxs'], 'text_boxes': cell_region_info[q]['text_boxes']})

                # get new split cells
                for q in row_subgraphs.keys():
                    tmp_row_info = row_subgraphs[q]
                    if len(tmp_row_info['cells_info']) == 1:
                        new_regions_info.append(tmp_row_info)
                        continue
                    row_subgraphs[p]['cells_info'] = list(sorted(tmp_row_info['cells_info'], key=lambda x: x["cell_bound"][0]))
                    scope = row_subgraphs[p]['scope']
                    row_cells_info = row_subgraphs[p]['cells_info']

                    # refresh the bound
                    new_split_row_cell_info = []
                    for q in range(len(row_cells_info)):
                        cell_info = row_cells_info[q]
                        text_boxes = cell_info['text_boxes']
                        new_x1 = 5e10
                        new_x2 = -1
                        new_y1 = 5e10
                        new_y2 = -1
                        for box in text_boxes:
                            bx, by, bw, bh = box
                            new_x1 = min(new_x1, bx)
                            new_y1 = min(new_y1, by)
                            new_x2 = max(new_x2, bx + bw)
                            new_y2 = max(new_y2, by + bh)
                        # if q == 0:
                        #     new_x1 = bound[0]
                        # elif q == len(row_cells_info) - 1:
                        #     new_x2 = bound[1]
                        cell_info['bound'] = [new_x1, new_y1, new_x2, new_y2]
                        new_split_row_cell_info.append(cell_info)

                    # new_split_cell_info.append(new_split_row_cell_info)
                    new_regions_info.extend(new_split_row_cell_info)
                    continue

            # transform text boxes rel
            # cells_rel = {}
            # q = 0
            # n = 0
            # m = 0
            # for q in boxes_rel.keys():
            #     rel = boxes_rel[str(q)]
            #     cell_idx = text_boxes_to_cell_region_map[q]
            #     same_row = []
            #     for n in rel['same_row']:
            #         same_row.append(text_boxes_to_cell_region_map[str(n)])
            #     same_col = []
            #     for m in rel['same_col']:
            #         same_col.append(text_boxes_to_cell_region_map[str(m)])
            #     if cell_idx not in cells_rels.keys():
            #         cells_rel[str(cell_idx)] = {'same_row': same_row, 'same_col': same_col}
            #     else:
            #         n = 0
            #         for n in same_row:
            #             if n not in cells_rel[str(cell_idx)]['same_row']:
            #                 cells_rel[str(cell_idx)]['same_row'].append(n)
            #         m = 0
            #         for m in same_col:
            #             if m not in cells_rel[str(cell_idx)]['same_col']:
            #                 cells_rel[str(cell_idx)]['same_col'].append(m)
        return new_regions_info

    def row_analyse(self, canvas: np.ndarray):
        row_proj = []
        for i in range(canvas.shape[0]):
            row_proj.append(np.sum(canvas[i, :]))

        row_subgraphs = {}

        row_st = -1
        for j in range(canvas.shape[0]):
            if row_proj[j] > 0 and row_st == -1:
                row_st = j
            elif row_proj[j] == 0 and row_st != -1:
                row_subgraphs[str(len(row_subgraphs))] = {'scope': [row_st, j - 1], 'text_boxes': [], 'cell_info': []}
                row_st = -1

        if row_st != -1:
            row_subgraphs[str(len(row_subgraphs))] = {'scope': [row_st, canvas.shape[0] - 1], 'text_boxes': [], 'cell_info': []}

        correct_st = 0
        for k in range(len(row_subgraphs)-1):
            correct_ed = int(row_subgraphs[str(k)]['scope'][1] + row_subgraphs[str(k+1)]['scope'][0] / 2)
            row_subgraphs[str(k)]['scope'] = [correct_st, correct_ed]
        return row_subgraphs

    def col_analyse(self, canvas: np.ndarray, img: np.ndarray):
        col_proj = []
        for k in range(canvas.shape[1]):
            col_proj.append(np.sum(canvas[:, k]))

        col_subgraphs = {}

        col_st = -1
        for q in range(canvas.shape[1]):
            if col_proj[q] > 0 and col_st == -1:
                col_st = q
            elif col_proj[q] == 0 and col_st != -1:
                col_subgraphs[str(len(col_subgraphs))] = {'scope': [col_st, q - 1], 'text_boxes': []}
                col_st = -1

        if col_st != -1:
            col_subgraphs[str(len(col_subgraphs))] = {'scope': [col_st, canvas.shape[1] - 1], 'text_boxes': []}
        return col_subgraphs

    def split_into_subgraph(self, canvas: np.ndarray, text_boxes: List, img: np.ndarray = None, iou_thresh=0.6):
        """ split text boxes into subgraphs

        Args:
            canvas (np.ndarray): blank matrix with text box as 1
            text_boxes (List): the small text boxes, [[x, y, w, h], ...]
            img (np.ndarray): for debug
        Returns:
            box_groups (List): list of group of text boxes
        """
        row_proj = []
        for i in range(canvas.shape[0]):
            row_proj.append(np.sum(canvas[i, :]))

        row_subgraphs = {}

        row_st = -1
        for j in range(canvas.shape[0]):
            if row_proj[j] > 0 and row_st == -1:
                row_st = j
            elif row_proj[j] == 0 and row_st != -1:
                row_subgraphs[str(len(row_subgraphs))] = {'scope': [row_st, j - 1], 'text_boxes': []}
                row_st = -1

        if row_st != -1:
            row_subgraphs[str(len(row_subgraphs))] = {'scope': [row_st, canvas.shape[0] - 1], 'text_boxes': []}

        col_proj = []
        for k in range(canvas.shape[1]):
            col_proj.append(np.sum(canvas[:, k]))

        col_subgraphs = {}

        col_st = -1
        for q in range(canvas.shape[1]):
            if col_proj[q] > 0 and col_st == -1:
                col_st = q
            elif col_proj[q] == 0 and col_st != -1:
                col_subgraphs[str(len(col_subgraphs))] = {'scope': [col_st, q - 1], 'text_boxes': []}
                col_st = -1

        if col_st != -1:
            col_subgraphs[str(len(col_subgraphs))] = {'scope': [col_st, canvas.shape[1] - 1], 'text_boxes': []}

        for box in text_boxes:
            x, y, w, h = box
            for n in range(len(row_subgraphs)):
                subgraph_scope = row_subgraphs[str(n)]['scope']
                if y >= subgraph_scope[0] and y + h <= subgraph_scope[1]:
                    row_subgraphs[str(n)]['text_boxes'].append(box)
                elif y + h <= subgraph_scope[0] or y >= subgraph_scope[1]:
                    pass
                else:
                    iou = 0
                    if y >= subgraph_scope[0] and y < subgraph_scope[1] and y + h > subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - y)/h, 2)
                    elif y < subgraph_scope[0] and y + h > subgraph_scope[0] and y + h <= subgraph_scope[1]:
                        iou = round((y+h-subgraph_scope[0])/h, 2)
                    elif y <= subgraph_scope[0] and y + h >= subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - subgraph_scope[0])/h, 2)
                    if iou >= iou_thresh:
                        row_subgraphs[str(n)]['text_boxes'].append(box)

            for m in range(len(col_subgraphs)):
                subgraph_scope = col_subgraphs[str(m)]['scope']
                if x >= subgraph_scope[0] and x + h <= subgraph_scope[1]:
                    col_subgraphs[str(m)]['text_boxes'].append(box)
                elif x + h <= subgraph_scope[0] or x >= subgraph_scope[1]:
                    pass
                else:
                    iou = 0
                    if x >= subgraph_scope[0] and x < subgraph_scope[1] and x + w > subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - x)/w, 2)
                    elif x < subgraph_scope[0] and x + w > subgraph_scope[0] and x + w <= subgraph_scope[1]:
                        iou = round((x + w - subgraph_scope[0])/w, 2)
                    elif x <= subgraph_scope[0] and x + w > subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - subgraph_scope[0])/w, 2)
                    if iou >= iou_thresh:
                        col_subgraphs[str(m)]['text_boxes'].append(box)

        # deal some error row
        i = 0
        for i in row_subgraphs.keys():
            if len(row_subgraphs[i]['text_boxes']) > 1:
                continue
            y1, y2 = row_subgraphs[i]['scope']
            if int(i) == 0:
                continue
            elif int(i) == len(row_subgraphs) - 1:
                row_subgraphs[str(int(i) - 1)]['text_boxes'].extend(row_subgraphs[i]['text_boxes'])
                y21, y22 = row_subgraphs[str(int(i) - 1)]['scope']
                row_subgraphs[str(int(i) - 1)]['scope'] = [y21, y2]
            else:
                yl1, yl2 = row_subgraphs[str(int(i) - 1)]['scope']
                yr1, yr2 = row_subgraphs[str(int(i) + 1)]['scope']
                if y1 - yl2 <= yr1 - y2:
                    row_subgraphs[str(int(i) - 1)]['text_boxes'].extend(row_subgraphs[i]['text_boxes'])
                    row_subgraphs[str(int(i) - 1)]['scope'] = [yl1, y2]
                else:
                    row_subgraphs[str(int(i) + 1)]['text_boxes'].extend(row_subgraphs[i]['text_boxes'])
                    row_subgraphs[str(int(i) + 1)]['scope'] = [y1, yr2]
            row_subgraphs[i]['text_boxes'] = []
        row_subgraphs = {n: row_subgraphs[n] for n in row_subgraphs.keys() if row_subgraphs[n]['text_boxes'] != []}

        # deal some error col
        i = 0
        for i in col_subgraphs.keys():
            if len(col_subgraphs[i]['text_boxes']) > 1:
                continue
            x1, x2 = col_subgraphs[i]['scope']
            if int(i) == 0:
                continue
            elif int(i) == len(col_subgraphs) - 1:
                col_subgraphs[str(int(i) - 1)]['text_boxes'].extend(col_subgraphs[i]['text_boxes'])
                x21, x22 = col_subgraphs[str(int(i) - 1)]['scope']
                col_subgraphs[str(int(i) - 1)]['scope'] = [x21, x2]
            else:
                xl1, xl2 = col_subgraphs[str(int(i) - 1)]['scope']
                xr1, xr2 = col_subgraphs[str(int(i) + 1)]['scope']
                if x1 - xl2 <= xr1 - x2:
                    col_subgraphs[str(int(i) - 1)]['text_boxes'].extend(col_subgraphs[i]['text_boxes'])
                    col_subgraphs[str(int(i) - 1)]['scope'] = [xl1, x2]
                else:
                    col_subgraphs[str(int(i) + 1)]['text_boxes'].extend(col_subgraphs[i]['text_boxes'])
                    col_subgraphs[str(int(i) + 1)]['scope'] = [x1, xr2]
            col_subgraphs[i]['text_boxes'] = []

        col_subgraphs = {n: col_subgraphs[n] for n in col_subgraphs.keys() if col_subgraphs[n]['text_boxes'] != []}

        subgraphs = {'row': row_subgraphs, 'col': col_subgraphs}

        if self.logger_flag == INFO and img is not None:
            from PIL import Image, ImageDraw, ImageFont
            # row
            i = 0
            row_subgraph_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            row_img = row_subgraph_canvas.copy()
            random.seed(0)
            draw_row_img = ImageDraw.Draw(row_img)

            for i in row_subgraphs.keys():
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                scope = row_subgraphs[i]['scope']
                scope_pts = self.box_to_four_coordinates(box=[0, scope[0], img.shape[1], scope[1] - scope[0]])
                draw_row_img.polygon(scope_pts, fill=color)

                text_boxes = row_subgraphs[i]['text_boxes']
                j = 0
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for j in range(len(text_boxes)):
                    box = text_boxes[j]
                    pts = self.box_to_four_coordinates(box=box)
                    draw_row_img.polygon(pts, fill=color)

            row_img = np.array(Image.blend(row_subgraph_canvas, row_img, 0.5))
            row_img_sp = None
            if self.save_dir:
                row_img_sp = os.path.join(self.save_dir, 'row_img.png')
            self.show_img(img=row_img, sp=row_img_sp)
            self.logger.debug('finish row')
            del row_img
            del row_subgraph_canvas
            gc.collect()

            # col
            i = 0
            col_subgraphs_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            col_img = col_subgraphs_canvas.copy()
            draw_col_img = ImageDraw.Draw(col_img)

            for i in col_subgraphs.keys():
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                scope = col_subgraphs[i]['scope']
                scope_pts = self.box_to_four_coordinates(box=[scope[0], 0, scope[1] - scope[0], img.shape[0]])
                draw_col_img.polygon(scope_pts, fill=color)

                text_boxes = col_subgraphs[i]['text_boxes']
                j = 0
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for j in range(len(text_boxes)):
                    box = text_boxes[j]
                    pts = self.box_to_four_coordinates(box=box)
                    draw_col_img.polygon(pts, fill=color)

            col_img = np.array(Image.blend(col_subgraphs_canvas, col_img, 0.5))
            col_img_sp = None
            if self.save_dir:
                col_img_sp = os.path.join(self.save_dir, 'col_img.png')
            self.show_img(img=col_img, sp=col_img_sp)
            gc.collect()
            self.logger.debug('finish column')

        return subgraphs

    def modify_text_boxes(self, text_box: List, box_img: np.ndarray):
        """correct the results of ocr det model 

        Args:
            text_boxes (List): the text boxes,  [x, y, w, h]
            box_img (np.ndarray): the text box image
        Returns:
            new_text_boxes (List): the text boxes
            new_box_img (np.ndarray): the binary image
        """
        _, bin_box_img = cv2.threshold(box_img, 127, 1, cv2.THRESH_BINARY)
        bin_box_img = 1 - bin_box_img
        x, y, w, h = text_box
        hor_proj = []
        for i in range(bin_box_img.shape[1]):
            hor_proj.append(np.sum(bin_box_img[:, i]))

        margins = []
        st = -1
        for j in range(len(hor_proj)):
            if j == 0 and hor_proj[j] == 0:
                continue
            if hor_proj[j] == 0 and hor_proj[j-1] != 0:
                st = j
            elif hor_proj[j] != 0 and st != -1 and j - 1 - st > 0:
                margins.append({'scope': [st, j - 1], 'length': j - 1 - st})
                st = -1

        if len(margins) <= 3:
            return [text_box], [box_img]

        median_thresh = np.percentile([_d['length'] for _d in margins], 90)
        bonds = []
        for k in margins:
            if k['length'] > median_thresh:
                bonds.append(k['scope'])

        self.logger.debug(f'----- modify text box -----\nhor_proj: {hor_proj}\nmargins: {margins}\nmedian_thresh: {median_thresh}\nbonds:{bonds}')

        if len(bonds) == 0:
            return [text_box], [box_img]

        new_st = x
        new_text_boxes = []
        new_box_imgs = []
        for n in range(len(bonds)):
            if n == 0:
                if bonds[n][0] > 1 and h > 1:
                    new_text_boxes.append([new_st, y, bonds[n][0], h])
                    tmp_box_img = box_img[0:h, new_st-x:new_st-x+bonds[n][0]]
                else:
                    tmp_box_img = []
                # assert tmp_box_img != [], f'[0:h, new_st-x:new_st-x+bonds[n][0]]: [0:{h}, {new_st-x}:{new_st-x+bonds[n][0]}], box_img: {box_img.shape}, bonds: {bonds}, n: {n}'
            else:
                if bonds[n][0] - bonds[n-1][1] > 1 and h > 1:
                    new_text_boxes.append([new_st, y, bonds[n][0] - bonds[n-1][1], h])
                    tmp_box_img = box_img[0:h, new_st-x:new_st-x+bonds[n][0]-bonds[n-1][0]]
                else:
                    tmp_box_img = []
                # assert tmp_box_img != [], f'[0:h, new_st-x:new_st-x+bonds[n][0] - bonds[n-1][1]]: [0:{h}, {new_st-x}:{new_st-x+bonds[n][0] - bonds[n-1][1]}], box_img: {box_img.shape}, bonds: {bonds}, n: {n}'

            if tmp_box_img != []:
                new_box_imgs.append(tmp_box_img)
            new_st = x + bonds[n][1]

        if new_st != len(hor_proj) - 1:
            if x + w - new_st > 1 and h > 1:
                new_text_boxes.append([new_st, y, x + w - new_st, h])
                tmp_box_img = box_img[0:h, new_st-x: w]
                assert tmp_box_img != [], f'[0:h, new_st-x: w]: [0:{h}, {new_st-x}: {w}], box_img: {box_img.shape}, new_st: {new_st}, w: {w}'
                new_box_imgs.append(tmp_box_img)

        self.logger.debug(f'new_text_boxes: {new_text_boxes}')

        return new_text_boxes, new_box_imgs

    def draw_boxes(self, img: np.ndarray, boxes: List, color_mode: str = 'random', color: Tuple= None):
        """tool for debug

        Args:
            img (np.ndarray): the canvas image
            boxes (List): [x, y, w, h]
            color_mode (str): can be 'random' or 'same'.

        Returns:
            img (np.ndarray): the img with text boxes colored as expected
        """
        img = img.astype(np.uint8)
        from PIL import Image, ImageDraw, ImageFont
        if len(img.shape) == 3:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif len(img.shape) == 2:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        else:
            raise ValueError(f'img.shape: {img.shape} not supported')

        h, w = image.height, image.width
        img_top = image.copy()
        img_bottom = cv2.cvtColor(np.ones((h, w, 3), dtype=np.uint8) * 255, cv2.COLOR_BGR2RGB)
        random.seed(0)

        draw_top = ImageDraw.Draw(img_top)
        if color_mode == 'same' and color is None:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for box in boxes:
            if color_mode == "random":
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_top.polygon(box, fill=color)
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bottom, [pts], True, color, 1)
        img_top = Image.blend(image, img_top, 0.5)
        img_show = Image.new("RGB", (w, h * 2), (255, 255, 255))
        img_show.paste(img_top, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_bottom), (0, h, w, h  * 2))
        return np.array(img_show)

    def get_boxes_rel(self, box1: list, box2: list, res_boxes):
        """predict the relationship between two boxes

        Args:
            box1 (list): [x, y, w, h]
            box2 (list): [x, y, w, h]

        Returns:
            int: 0: same cell, 1: same row, 2: same col, 3: same no relation
        """
        ws = []
        hs = []
        for _box in res_boxes:
            ws.append(_box[2])
            hs.append(_box[3])
        median_w = np.median(ws)
        median_h = np.median(hs)

        x1, y1, w1, h1 = box1
        x12 = x1 + w1
        y12 = y1 + h1
        core_x1 = x1 + 0.5 * w1
        core_y1 = y1 + 0.5 * h1

        x2, y2, w2, h2 = box2
        x22 = x2 + w2
        y22 = y2 + h2
        core_x2 = x2 + 0.5 * w2
        core_y2 = y2 + 0.5 * h2

        core_x_diff = round((core_x1 - core_x2) / median_w, 4)
        core_y_diff = round((core_y1 - core_y2) / median_h, 4)
        lt_x_diff = round((x1 - x2) / median_w, 4)
        br_x_diff = round((x12 - x22) / median_w, 4)
        lt_y_diff = round((y1 - y2) / median_h, 4)
        br_y_diff = round((y12 - y22) / median_h, 4)
        w_diff = round((w1 - w2) / median_w, 4)
        h_diff = round((h1 - h2) / median_h, 4)

        x_data = [{'core_x_diff': core_x_diff, 'core_y_diff': core_y_diff, 'lt_x_diff': lt_x_diff, 'br_x_diff': br_x_diff, 'lt_y_diff': lt_y_diff, 'br_y_diff': br_y_diff, 'w_diff': w_diff, 'h_diff': h_diff}]
        x_df = pd.DataFrame(data=x_data, columns=['core_x_diff', 'core_y_diff', 'lt_x_diff', 'br_x_diff', 'lt_y_diff', 'br_y_diff', 'w_diff', 'h_diff'])

        y = self.model.predict(x_df)[0]

        return y


# Function to compute IoU between two rectangles, from paddlex\inference\pipelines\table_recognition\pipeline_v2.py
def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two rectangles.

    Args:
        box1 (array-like): [x1, y1, x2, y2] of the first rectangle.
        box2 (array-like): [x1, y1, x2, y2] of the second rectangle.

    Returns:
        float: The IoU between the two rectangles.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Calculate the area of both rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the IoU
    iou = intersection_area / float(box1_area)
    return iou


def compute_iou_cal_metrics(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two rectangles.

    Args:
        box1 (array-like): [x1, y1, x2, y2] of the first rectangle.
        box2 (array-like): [x1, y1, x2, y2] of the second rectangle.

    Returns:
        float: The IoU between the two rectangles.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Calculate the area of both rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the IoU
    iou = intersection_area / (float(box1_area) + float(box2_area) - intersection_area)
    return iou