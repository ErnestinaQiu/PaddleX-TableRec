import os
import gc
import cv2
import math
import random
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
from logging import NOTSET, DEBUG, INFO, ERROR
from typing import Any, List, Optional, Tuple
from paddlex import create_pipeline
from paddlex import create_model
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger
from paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2 import sort_table_cells_boxes

from exp_exist_label import draw_tables

from PIL import Image, ImageDraw


class TableRec:
    def __init__(
        self,
        log_level=DEBUG,
        log_file='./output/exp/logs/debug.log',
        save_dir=None,
        md_path='./output/exp/XGB/20250610115334/xgb.pickle',
    ) -> None:
        """an ocr and its postprocess for table
        Args:
            log_level (int, optional):
            log_file (str, optional): 
            save_dir (str, optional): default None

        Returns:
            _type_: _description_
        """
        self.logger_flag = log_level
        self.logger = get_logger(name='OcrRec', log_file=log_file, log_level=log_level)
        self.table_ocr = TableOCR(log_level, log_file, save_dir=save_dir, md_path=md_path)
        self.cell_det_model = create_model(model_name="RT-DETR-L_wireless_table_cell_det")
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def predict(self, img_path):
        img = self.read_img(img_path=img_path)
        frame_lines = self.classify_table(img)
        if len(frame_lines) == 0:
            pred_bounds = self.predict_no_frame(img, img_path=img_path)
        else:
            pred_bounds = self.predict_three_lines_frame(img, frame_lines)
        return pred_bounds

    def predict_no_frame(self, img, img_path):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output = self.cell_det_model.predict(img, threshold=0.3, batch_size=1)
        pred_bounds = []
        for res in output:
            pred_dicts = res._to_json()['res']['boxes']
            for k in range(len(pred_dicts)):
                pred_bounds.append(pred_dicts[k]['coordinate'])
        regions = self.table_ocr.predict(img_path)
        cells_info = self.merge_predictions(img, pred_bounds, regions)
        pred_bounds = [d['bound'] for d in cells_info]
        # pred_bounds = self.frame_alignment(img, pred_bounds)
        return pred_bounds

    # TODO not finish yet, need to alignment frame by def frame_alignment and use inform frame lines
    def predict_three_lines_frame(self, img, frame_lines):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output = self.cell_det_model.predict(img, threshold=0.3, batch_size=1)
        pred_bounds = []
        for res in output:
            pred_dicts = res._to_json()['res']['boxes']
            for k in range(len(pred_dicts)):
                pred_bounds.append(pred_dicts[k]['coordinate'])

        return pred_bounds


    def frame_alignment(self, img, bounds):
        """align all the bounds into frame

        Args:
            bounds (list): the bounds of cells. [(x1, y1, x2, y2), ......]
        """
        def get_weighted_value(bound_votes, st, ed):
            """Include end

            Args:
                bound_votes (list): Record the vote of new bounds of the predicted bounds.
                st (int): Start of no zero vote
                ed (int): End of no zero vote

            Returns:
                _type_: _description_
            """
            votes = np.sum(bound_votes[st: ed + 1])
            new_b = 0
            for q in range(st, ed + 1):
                new_b += q * bound_votes[q] / votes
            new_b = int(new_b)
            return new_b

        def get_new_bounds(bound_votes, align_thresh):
            new_bounds = []
            i = 0
            st = -1
            ed = -1
            while i < len(bound_votes):
                if bound_votes[i] == 0 and st == -1:
                    i += 1
                    continue
                elif bound_votes[i] == 0 and st != -1:
                    stop_guard = False
                    if st + align_thresh - 1 < len(bound_votes):
                        if np.sum(bound_votes[st: st + align_thresh]) > np.sum(bound_votes[st: i]):
                            pass
                        else:
                            stop_guard = True
                    else:
                        if np.sum(bound_votes[st: -1]) > np.sum(bound_votes[st: i]):
                            pass
                        else:
                            stop_guard = True

                    if stop_guard:
                        ed = i - 1
                        new_b = get_weighted_value(bound_votes, st, ed)
                        new_bounds.append(new_b)
                        st = -1
                        ed = -1
                    else:
                        ed = i
                elif bound_votes[i] != 0 and st == -1:
                    st = i
                else:
                    ed = i

                if ed - st >= align_thresh - 1:
                    new_b = get_weighted_value(bound_votes, st, ed)
                    st = -1
                    ed = -1
                    new_bounds.append(new_b)

                i += 1

            return new_bounds

        row_votes = [0 for i in range(img.shape[0])]
        col_votes = [0 for i in range(img.shape[1])]
        for bound in bounds:
            x1, y1, x2, y2 = bound
            if y1 < img.shape[0]:
                row_votes[int(y1)] += 1
            else:
                row_votes[-1] += 1
            if y2 < img.shape[0]:
                row_votes[int(y2)] += 1
            else:
                row_votes[-1] += 1

            if x1 < img.shape[1]:
                col_votes[int(x1)] += 1
            else:
                col_votes[-1] += 1
            if x2 < img.shape[1]:
                col_votes[int(x2)] += 1
            else:
                col_votes[-1] += 1

        # get new row bounds and col bounds
        row_bounds = get_new_bounds(bound_votes=row_votes, align_thresh=7)
        col_bounds = get_new_bounds(bound_votes=col_votes, align_thresh=11)

        if self.logger_flag == DEBUG:
            canvas = np.ones(img.shape) * 255
            img_copy = deepcopy(img)
            for a in row_bounds:
                cv2.line(img_copy, (0, int(a)), (img.shape[1], int(a)), (150, 150, 150), 2)
                cv2.line(canvas, (0, int(a)), (img.shape[1], int(a)), (0, 150, 0), 2)
            for b in col_bounds:
                cv2.line(img_copy, (int(b), 0), (int(b), img.shape[0]), (150, 150, 150), 2)
                cv2.line(canvas, (int(b), 0), (int(b), img.shape[0]), (0, 150, 0), 2)
            cv2.imshow('Frame Alignment', img_copy)
            cv2.imshow('Frame Alignment Canvas', canvas)
            cv2.waitKey(0)

        new_bounds = []
        for k in range(len(bounds)):
            x1, y1, x2, y2 = bounds[k]
            for _y in row_bounds:
                if abs(y1 - _y) < 10:
                    y1 = _y
                if abs(y2 - _y) < 10:
                    y2 = _y
            for _x in col_bounds:
                if abs(x1 - _x) < 10:
                    x1 = _x
                if abs(x2 - _x) < 10:
                    x2 = _x
            new_bounds.append((int(x1), int(y1), int(x2), int(y2)))
        return new_bounds

    def merge_predictions(self, img, pred_bounds, regions, match_iou_thresh=0.5):
        """merge predictions

        Args:
            img (Array): gray image
            pred_bounds (list): [(x1, y1, x2, y2), ...]
            regions (list): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]

        Returns:
            cells_info (list): list of dict
        """
        add_cells_info = []
        for i in range(len(regions)):
            region_info = regions[i]
            region_bound = region_info['bound']
            miss_guard = True
            for pred_bound in pred_bounds:
                region_iou = compute_iou(region_bound, pred_bound)
                if region_iou >= match_iou_thresh:
                    miss_guard = False
                    break
            if miss_guard:
                if 'empty_cell' in region_info.keys():
                    x1, y1, x2, y2 = region_bound
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    empty_cell = region_info['empty_cell']
                else:
                    x1, y1, x2, y2 = region_bound
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    tmp_img = img[y1: y2, x1:x2]
                    if np.sum(tmp_img) < 5:
                        empty_cell = 1
                    else:
                        empty_cell = 0
                add_cells_info.append({'bound': (x1, y1, x2, y2), 'empty_cell': empty_cell})

        cells_info = add_cells_info
        for pred_bound in pred_bounds:
            x1, y1, x2, y2 = pred_bound
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            tmp_img = img[y1: y2, x1:x2]
            if np.sum(tmp_img) < 5:
                empty_cell = 1
            else:
                empty_cell = 0
            info = {'bound': (x1, y1, x2, y2), 'empty_cell': empty_cell}
            add_cells_info.append(info)

        return cells_info

    def analyse_frame_lines(self, img, frame_lines, header_lines_thresh=0.6):
        frame_lines = list(sorted(frame_lines, key=lambda x: x[1]))
        max_length = -1
        max_line_x1 = -1
        max_line_x2 = -1
        for i in range(len(frame_lines)):
            line = frame_lines[i]
            x1, y1, x2, y2 = line
            if max_length < abs(x2 - x1):
                max_length = abs(x2 - x1)
                max_line_x1 = min(x1, x2)
                max_line_x2 = max(x1, x2)

        # get complex row lines
        frame_info = {'frame_width': max_length, 'frame_x1': max_line_x1, 'frame_x2': max_line_x2, 'header_lines': [], 'skip_index_cell_lines': []}
        # # get header lines
        for line in frame_lines:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) < max_length * header_lines_thresh:
                frame_info['header_lines'].append(line)

        old_header_lines = deepcopy(frame_info['header_lines'])
        for line in frame_lines:
            x1, y1, x2, y2 = line
            y = np.mean([y1, y2])
            if line in old_header_lines:
                continue
            for header_line in old_header_lines:
                h_x1, h_y1, h_x2, h_y2 = header_line
                tmp_y = np.mean([h_y1, h_y2])
                if abs(y - tmp_y) < 3:
                    frame_info['header_lines'].append(line)

        # # get merged index cells lines
        for line in frame_lines:
            x1, y1, x2, y2 = line
            if abs(x2 - x1) >= max_length * header_lines_thresh and abs(x2 - x1) < (max_length - 20) and abs(max_line_x2 - max(x1, x2)) < 3:
                frame_info['skip_index_cell_lines'].append(line)

        if self.logger_flag == NOTSET:
            canvas = np.ones(img.shape) * 255
            img_copy = deepcopy(img)
            for line in frame_info['header_lines']:
                x1, y1, x2, y2 = line
                cv2.line(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imshow('Header Lines', img_copy)
            cv2.imshow('Header Lines canvas', canvas)
            cv2.waitKey(0)

            canvas = np.ones(img.shape) * 255
            img_copy = deepcopy(img)
            for line in frame_info['skip_index_cell_lines']:
                x1, y1, x2, y2 = line
                cv2.line(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imshow('Skip Index Cell Lines', img_copy)
            cv2.imshow('Skip Index Cell Lines canvas', canvas)
            cv2.waitKey(0)
        return frame_info

    def classify_table(self, img, min_length_thresh=0.1):
        h, w = img.shape
        lsd = cv2.createLineSegmentDetector(
            refine=None,
            scale=0.8,
            sigma_scale=0.6,
            quant=2.0,
            ang_th=22.5,
            log_eps=0,
            density_th=0.7,
            n_bins=1024
        )
        lines, width, prec, nfa = lsd.detect(img)
        frame_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dis = euclidean_distance(point1=(x1, y1), point2=(x2, y2))
            if dis < min_length_thresh * w:
                continue
            frame_lines.append((x1, y1, x2, y2))

        if self.logger_flag == NOTSET:
            img_copy = deepcopy(img)
            canvas = np.ones(img.shape) * 255
            if lines is not None:
                for i in range(len(frame_lines)):
                    x1, y1, x2, y2 = frame_lines[i]
                    dis = euclidean_distance(point1=(x1, y1), point2=(x2, y2))
                    if dis < min_length_thresh * w:
                        continue
                    cv2.line(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imshow('LSD Result', img_copy)
            cv2.imshow('Result canvas', canvas)
            cv2.waitKey(0)

        return frame_lines

    def read_img(self, img_path: str):
        assert os.path.exists(img_path), f"img_path doesn't exists \n{img_path}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points
    Args:
        point1 (Tuple|List|Array): (x1, y1)
        point2 (Tuple|List|Array): (x2, y2)

    Returns:
        float: Euclidean distance
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


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
        self.cell_det_model = create_model(model_name="RT-DETR-L_wireless_table_cell_det")
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.model = pickle.load(open(md_path, 'rb'))

    def predict(self, img_path):
        img = self.check_and_read_img(img_path=img_path)
        res_boxes = self.get_ocr_text_boxes(img_path=img_path)
        canvas = self.ocr_box_canvas(text_boxes=res_boxes, img_shape=img.shape)
        canvas = canvas * 255
        regions = self.split_into_region(canvas=canvas, text_boxes=res_boxes, img=img)
        regions = self.merge_same_cells_deal_complex_region(regions=regions, img=img)
        return regions

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
            # new_shrink_boxes = shrink_boxes

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

    def show_img(self, img: np.ndarray, sp: str = None, show=False):
        if self.platform != 'aistudio' or show:
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
        # detect line and margin
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
        """ split text boxes into region cell, detect empty cell inside

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
                    self.logger.debug(f'finish tmp region, out into {tmp_region_sp}')
                    del tmp_region_img
                    del _region_canvas
                    gc.collect()

                regions.append(region)

        self.logger.debug(f'row_subgraphs: {row_subgraphs}\ncol_subgraphs: {col_subgraphs}\nregions: {regions}')

        if self.logger_flag == NOTSET and img is not None:
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

        proj_values = []  # debug param
        for i in range(range_scope):
            if mode == 'row':
                proj = np.sum(bin_canvas[i, :])
            elif mode == 'col':
                proj = np.sum(bin_canvas[:, i])

            proj_values.append(proj)
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

    # have done inside split into region
    def modify_region_bound_by_big_region_frame(self, regions_info):
        """modify region bound according to big region frame

        Args:
            region_info (dict): {'region': region, 'row_bounds': row_bounds, 'col_bounds': col_bounds}
                                region (list), {'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 1|0}

        Raises:
            ValueError: _description_

        Returns:
            region: list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...]}, ...]
        """
        regions = regions_info['region']
        row_bounds = regions_info['row_bounds']
        col_bounds = regions_info['col_bounds']
        for i in range(len(regions)):
            x1, y1, x2, y2 = regions[i]['bound']
            new_x1_info = {'min_dis': 999999, 'value': -1}
            new_y1_info = {'min_dis': 999999, 'value': -1}
            new_x2_info = {'min_dis': 999999, 'value': -1}
            new_y2_info = {'min_dis': 999999, 'value': -1}
            for p in row_bounds:
                if abs(p - y1) < new_y1_info['min_dis']:
                    new_y1_info['value'] = p
                    new_y1_info['min_dis'] = abs(p - y1)
                if abs(p - y2) < new_y2_info['min_dis']:
                    new_y2_info['value'] = p
                    new_y2_info['min_dis'] = abs(p - y2)
            for q in col_bounds:
                if abs(q - x1) < new_x1_info['min_dis']:
                    new_x1_info['value'] = q
                    new_x1_info['min_dis'] = abs(q - x1)
                if abs(q - x2) < new_x2_info['min_dis']:
                    new_x2_info['value'] = q
                    new_x2_info['min_dis'] = abs(q - x2)

            if self.logger_flag == NOTSET:
                if (x1, y1, x2, y2) != (new_x1_info['value'], new_y1_info['value'], new_x2_info['value'], new_y2_info['value']):
                    self.logger.debug(f"old bound: {(x1, y1, x2, y2)}, new_bound: {(new_x1_info['value'], new_y1_info['value'], new_x2_info['value'], new_y2_info['value'])}")

            regions[i]['bound'] = (new_x1_info['value'], new_y1_info['value'], new_x2_info['value'], new_y2_info['value'])

            self.logger.debug(f"region {i} old_bound: {[x1, y1, x2, y2]}, new_bound: {regions[i]['bound']}\nnew_x1_info: {new_x1_info}, new_y1_info: {new_y1_info}, new_x2_info: {new_x2_info}, new_y2_info: {new_y2_info}\nrow_bounds: {row_bounds}\ncol_bounds: {col_bounds}")

        return regions

    # repeat method of split into region
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
            region_boxes.append([x1, y1, x2 - x1, y2 - y1])
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

    # TODO assign bounds and to correct wrong bounds use cv
    def recheck_cells(self, regions: list, img: np.ndarray):
        """check whether one region has more than one cell, if yes, split them. Assume every region only have one row

        Args:
            regions (list): [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1, ...]
            img (np.ndarray): image

        Raises:
            ValueError: _description_

        Returns:
            list: [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1, ...]
        """
        new_regions = []
        for i in range(len(regions)):
            region_info = regions[i]

    def complex_region_frame(self, ):

        return

    # TODO deal empty cell inside
    def merge_same_cells_deal_complex_region(self, regions: list, img: np.ndarray, iou_thresh=0.6, dis_thresh=25):
        """merge same cell text ocr boxes, 

        Args:
            regions (List): list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]
            img (np.ndarray): image

        Returns:
            new_regions_info (List): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]
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
                        pass
                    if rel != 0:
                        dis, _ = self.cal_box_dis(box1, box2)
                        if dis < 0.2 * dis_thresh:
                            boxes_rel[str(n)]['same_cell'].append(m)

            # same cell merge
            same_cells_indexes = []
            for j in boxes_rel.keys():
                if len(boxes_rel[j]['same_cell']) == 0:
                    continue

                repeat_guard = False
                cur_box = text_boxes[int(j)]
                # check repeat
                for k in range(len(same_cells_indexes)):
                    if int(j) in same_cells_indexes[k]:
                        repeat_guard = True
                        continue
                if repeat_guard:
                    continue

                same_cell_idxs = boxes_rel[j]['same_cell']
                checked_same_cell_idxes = []
                for w in same_cell_idxs:
                    tmp_box = text_boxes[w]
                    dis, _ = self.cal_box_dis(cur_box, tmp_box)
                    if dis > dis_thresh:
                        continue
                    checked_same_cell_idxes.append(w)
                same_cell_idxs = checked_same_cell_idxes

                add_idxs = []
                for idx in same_cell_idxs:
                    tmp_idxes = boxes_rel[str(idx)]['same_cell']
                    for _idx in tmp_idxes:
                        if _idx in same_cell_idxs:
                            continue
                        tmp_box = text_boxes[_idx]
                        dis, _ = self.cal_box_dis(cur_box, tmp_box)
                        if dis > dis_thresh:
                            continue
                        add_idxs.append(idx)
                complete_same_cell_idxs = same_cell_idxs + add_idxs + [int(j)]

                if len(complete_same_cell_idxs) == 1:
                    continue
                same_cells_indexes.append(complete_same_cell_idxs)

            self.logger.debug(f'same_cells_indexes: {same_cells_indexes}')
            # merge same cell text boxes
            cell_region_info = []         # [{'bound': [x1, y1, x2, y2], 'text_boxes_idxs': [], 'text_boxes': [x, y, w, h]}]
            deal_text_boxes_idxs = []
            for q in range(len(same_cells_indexes)):
                cell_boxes_indexes = same_cells_indexes[q]
                x1 = 1e5
                y1 = 1e5
                x2 = -1
                y2 = -1
                same_cells_text_boxes = []
                for p in cell_boxes_indexes:
                    tmp_x1, tmp_y1, tmp_w, tmp_h = text_boxes[p]
                    tmp_x2 = tmp_x1 + tmp_w
                    tmp_y2 = tmp_y1 + tmp_h
                    x1 = min(x1, tmp_x1)
                    x2 = max(x2, tmp_x2)
                    y1 = min(y1, tmp_y1)
                    y2 = max(y2, tmp_y2)

                merged_cell_info = {'bound': (x1, y1, x2, y2), 'box': [x1, y1, x2 - x1, y2 - y1], 'text_boxes_idxs': cell_boxes_indexes, 'text_boxes': same_cells_text_boxes, 'empty_cell': 0}
                cell_region_info.append(merged_cell_info)
                deal_text_boxes_idxs.extend(cell_boxes_indexes)

            self.logger.debug(f'cell_region_info: {cell_region_info}')

            if self.logger_flag == NOTSET and img is not None:
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
                    self.logger.debug(f'finish merge cell region, out into {tmp_region_sp}')
                    del tmp_region_img
                    del _region_canvas
                    gc.collect()

            # deal the remained text_boxes
            for q in range(len(text_boxes)):
                if q in deal_text_boxes_idxs:
                    continue
                _x, _y, _w, _h = text_boxes[q]
                text_bound = (_x, _y, _x + _w, _y + _h)
                cell_region_info.append({'bound': text_bound, 'box': text_boxes[q], 'text_boxes_idxs': [q], 'text_boxes': [text_boxes[q]], 'empty_cell': 0})

            # deal overlap
            deal_overlap_cell_region = []
            overlap_indexes = []
            for a in range(len(cell_region_info)):
                bound_a = cell_region_info[a]['bound']
                if a in overlap_indexes:
                    continue
                append_guard = True
                for b in range(a + 1, len(cell_region_info)):
                    if b in overlap_indexes:
                        continue
                    bound_b = cell_region_info[b]['bound']
                    iou = max(compute_iou(bound_a, bound_b), compute_iou(bound_b, bound_a))
                    if iou > 0.3:
                        overlap_indexes.append(b)
                        cell_region_info[a]['text_boxes'].extend(cell_region_info[b]['text_boxes'])
                        new_bound = (min(bound_a[0], bound_b[0]), min(bound_a[1], bound_b[1]), max(bound_a[2], bound_b[2]), max(bound_b[3], bound_a[3]))
                        new_text_boxes_idxs = cell_region_info[a]['text_boxes_idxs'].extend(cell_region_info[b]['text_boxes_idxs'])
                        new_text_boxes = cell_region_info[a]['text_boxes'] + cell_region_info[b]['text_boxes']
                        new_cell_region = {'bound': new_bound, 'box': (new_bound[0], new_bound[1], new_bound[2] - new_bound[0], new_bound[3] - new_bound[1]), 'text_boxes_idxs': new_text_boxes_idxs, 'text_boxes': new_text_boxes, 'empty_cell': 0}
                        deal_overlap_cell_region.append(new_cell_region)
                        append_guard = False
                        continue
                if append_guard:
                    deal_overlap_cell_region.append(cell_region_info[a])

            cell_region_info = deal_overlap_cell_region

            if len(cell_region_info) == 1:
                cell_region_info[0]['bound'] = bound
                new_regions_info.extend(cell_region_info)
                continue

            # deal complex region
            cell_region_info = self.split_complex_region(cell_region_info=cell_region_info, bound=bound, img=img, iou_thresh=0.6)

            new_regions_info.extend(cell_region_info)
            continue
        return new_regions_info

    def split_complex_region(self, cell_region_info: List, bound: List, img: np.ndarray, iou_thresh=0.6):
        """split complex region into cells

        Args:
            cell_region_info (List): [{'bound': [x1, y1, x2, y2], 'text_boxes': [x, y, w, h], 'empty_cell': 0, 'box': [x1, y1, x2, y2], 'text_boxes_idxs': cell_boxes_indexes}]
            bound (List): bound of the complex region [x1, y1, x2, y2]
            img (np.ndarray): origin image
            iou_thresh (float, optional): _description_. Defaults to 0.6.

        Returns:
            new_regions_info (List): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1, ...]
        """
        if self.logger_flag == NOTSET:
            _cell_region_pts = [k['bound'] for k in cell_region_info]
            _bound_pts = [bound]
            print(f'_cell_region_pts: {_cell_region_pts}, _bound_pts: {_bound_pts}')
            _compare_table = self.draw_compare_table(img=img, gt_boxes=_bound_pts, pred_boxes=_cell_region_pts, mode='xyx1y1')
            self.logger.debug('input of split_complex_region')
            self.show_img(img=_compare_table, show=True)

        new_regions_info = []    # the same structure as cell_region_info
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

        if self.logger_flag == NOTSET:
            self.show_img(img=inside_region_canvas * 255, show=True)

        row_subgraphs = self.row_analyse(canvas=inside_region_canvas)
        # #transform coordinate from relative to absolute
        for s in range(len(row_subgraphs.keys())):
            q = list(row_subgraphs.keys())[s]
            if s == len(row_subgraphs.keys()) - 1:
                row_subgraphs[q]['scope'] = [row_subgraphs[q]['scope'][0] + bound[1], bound[3]]
            else:
                row_subgraphs[q]['scope'] = [row_subgraphs[q]['scope'][0] + bound[1], row_subgraphs[q]['scope'][1] + bound[1]]

        self.logger.debug(f'row_subgraphs: {row_subgraphs}, bound: {bound}')

        # case 1
        if len(row_subgraphs) == 1:
            sorted_cell_region_info = list(sorted(cell_region_info, key=lambda x: x["bound"][0]))
            # fresh bound
            new_bound_cell_region_info = self.fresh_bound(sorted_cell_region_info=sorted_cell_region_info, bound=bound)
            new_regions_info.extend(new_bound_cell_region_info)

            if self.logger_flag == NOTSET:
                _pts = [self.transform_x1y1x2y2_into_four_coordinates(_d['bound']) for _d in new_regions_info]
                case1_table = draw_tables(img, _pts)
                self.show_img(case1_table, show=True)
            return new_regions_info
        else:
            # split rows
            for p in row_subgraphs.keys():
                subgraph_scope = row_subgraphs[p]['scope']
                row_subgraphs[p]['cells_info'] = []   # [{'cell_idx': int, 'bound': [x1, y1, x2, y2], 'cell_box': [x, y, w, h], 'text_boxes_idxs': list of int|indexes of text boxes which belong to the same cell, 'text_boxes': []}]
                row_subgraphs[p]['bound'] = [bound[0], subgraph_scope[0], bound[2], subgraph_scope[1]]
                for q in range(len(cell_region_info)):
                    cell_bound = cell_region_info[q]['bound']
                    x, y, x1, y1 = cell_bound
                    h = y1 - y
                    if y >= subgraph_scope[0] and y + h <= subgraph_scope[1]:
                        row_subgraphs[p]['cells_info'].append({'cell_idx': q, 'bound': cell_region_info[q]['bound'], 'text_boxes_idxs': cell_region_info[q]['text_boxes_idxs'], 'text_boxes': cell_region_info[q]['text_boxes']})
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
                            row_subgraphs[p]['cells_info'].append({'cell_idx': q, 'bound': cell_region_info[q]['bound'], 'text_boxes_idxs': cell_region_info[q]['text_boxes_idxs'], 'text_boxes': cell_region_info[q]['text_boxes']})
                self.logger.debug(f'row_subgraphs: {row_subgraphs}')

                if self.logger_flag == NOTSET:
                    self.logger.debug(f"split row\nrow_subgraphs[{p}]['bound']: {row_subgraphs[p]['bound']}\nrow_subgraphs[{p}]['cells_info']: {row_subgraphs[p]['cells_info']}")
                    _pts = [self.transform_x1y1x2y2_into_four_coordinates(_d['bound']) for _d in row_subgraphs[p]['cells_info']]
                    case1_table = draw_tables(img, _pts)
                    self.show_img(case1_table, show=True)

            # get new split cells
            for k in range(len(row_subgraphs.keys())):
                q = list(row_subgraphs.keys())[k]
                tmp_row_info = row_subgraphs[q]
                if len(tmp_row_info['cells_info']) == 1:
                    tmp_row_info['cells_info'][0]['bound'] = tmp_row_info['bound']
                    new_regions_info.append(tmp_row_info['cells_info'][0])

                    if self.logger_flag == NOTSET:
                        self.logger.debug('get new split cells')
                        _pts = [self.transform_x1y1x2y2_into_four_coordinates(tmp_row_info['bound'])]
                        case1_table = draw_tables(img, _pts)
                        self.show_img(case1_table, show=True)
                    continue

                row_subgraphs[p]['cells_info'] = list(sorted(tmp_row_info['cells_info'], key=lambda x: x["bound"][0]))
                # scope = row_subgraphs[p]['scope']
                # row_bound = (row_subgraphs[p]['bound'][0], scope[0], row_subgraphs[p]['bound'][1], scope[1])
                row_bound = row_subgraphs[p]['bound']
                row_cells_info = row_subgraphs[p]['cells_info']

                if len(row_cells_info) == 0:
                    continue

                # refresh the bound
                new_split_row_cell_info = self.fresh_bound(sorted_cell_region_info=row_cells_info, bound=row_bound)

                new_regions_info.extend(new_split_row_cell_info)
                continue

        if self.logger_flag == NOTSET:
            self.logger.debug('Output of split_complex_region')
            _pts = [self.transform_x1y1x2y2_into_four_coordinates(_d['bound']) for _d in new_regions_info]
            case1_table = draw_tables(img, _pts)
            self.show_img(case1_table, show=True)

        return new_regions_info

    def fresh_bound(self, sorted_cell_region_info, bound):
        q = 0
        x_st = bound[0]
        new_bound_cell_region_info = []
        for q in range(len(sorted_cell_region_info) - 1):
            cur_bound = sorted_cell_region_info[q]['bound']
            next_bound = sorted_cell_region_info[q + 1]['bound']
            x_ed = 0.5 * (cur_bound[2] + next_bound[0])
            sorted_cell_region_info[q]['bound'] = [x_st, bound[1], x_ed, bound[3]]
            new_bound_cell_region_info.append(sorted_cell_region_info[q])
            x_st = x_ed

        sorted_cell_region_info[-1]['bound'] = [x_st, bound[1], bound[2], bound[3]]
        new_bound_cell_region_info.append(sorted_cell_region_info[-1])
        return new_bound_cell_region_info

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
                row_subgraphs[str(len(row_subgraphs))] = {'scope': [row_st, j - 1], 'text_boxes': [], 'cells_info': []}
                row_st = -1

        if row_st != -1:
            row_subgraphs[str(len(row_subgraphs))] = {'scope': [row_st, canvas.shape[0] - 1], 'text_boxes': [], 'cells_info': []}

        self.logger.debug(f'row_analyse row_subgraphs: {row_subgraphs}')

        correct_st = 0
        for k in range(len(row_subgraphs) - 1):
            correct_ed = int((row_subgraphs[str(k)]['scope'][1] + row_subgraphs[str(k+1)]['scope'][0]) / 2)
            row_subgraphs[str(k)]['scope'] = [correct_st, correct_ed]
            correct_st = correct_ed

        row_subgraphs[list(row_subgraphs.keys())[-1]]['scope'] = [correct_st, row_subgraphs[list(row_subgraphs.keys())[-1]]['scope'][1]]

        self.logger.debug(f'row_analyse 2 row_subgraphs: {row_subgraphs}')
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

        if self.logger_flag == NOTSET and img is not None:
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

    def modify_text_boxes(self, text_box: List, box_img: np.ndarray, col_split_percent=90):
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

        median_thresh = np.percentile([_d['length'] for _d in margins], col_split_percent)
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

    def cal_box_dis(self, box1: List, box2: List):
        """calculate the Euclidean distance between core of box

        Args:
            box1 (List): [x1, y1, w1, h1]
            box2 (List): [x2, y2, w2, h2]

        Returns:
            float: Euclidean distance

            Tuple: closest point pair
        """
        # pts on box1
        row_samples = np.linspace(box1[1], box1[1] + box1[3], num=10)
        col_samples = np.linspace(box1[0], box1[0] + box1[2], num=10)
        pts1 = []
        for i in row_samples:
            for j in col_samples:
                pts1.append((i, j))

        # pts on box2
        row_samples = np.linspace(box2[1], box2[1] + box2[3], num=4)
        col_samples = np.linspace(box2[0], box2[0] + box2[2], num=4)
        pts2 = []
        for i in row_samples:
            for j in col_samples:
                pts2.append((i, j))

        min_distance = float('inf')
        closest_pair = (None, None)

        for point1 in pts1:
            for point2 in pts2:
                distance = euclidean_distance(point1, point2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (point1, point2)

        return min_distance, closest_pair

    def draw_compare_table(self, img, gt_boxes, pred_boxes, mode='xyx1y1'):
        """draw gt_boxes by polyline and pred_boxes by colorful polygon

        Args:
            gt_boxes (list): ground truth boxes, [x1, y1, x2, y2] | [x, y, w, h]
            pred_boxes (list): prediction boxes, [x1, y1, x2, y2] | [x, y, w, h]
            mode (str): 'xyx1y1' | 'xywh'
        """
        if mode == 'xyx1y1':
            gt_pts = [self.transform_x1y1x2y2_into_four_coordinates(box) for box in gt_boxes]
            pred_pts = [self.transform_x1y1x2y2_into_four_coordinates(box) for box in pred_boxes]
        elif mode == 'xywh':
            gt_pts = self.box_to_four_coordinates(gt_boxes)
            pred_pts = self.box_to_four_coordinates(pred_boxes)
        else:
            raise ValueError(f'mode {mode} is not supported')

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_top = image.copy()
        random.seed(0)

        draw_top = ImageDraw.Draw(img_top)
        draw_top = ImageDraw.Draw(img_top)

        for pts in pred_pts:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_top.polygon(pts, fill=color)

        img_top = Image.blend(image, img_top, 0.5)
        compare_region = np.array(img_top)

        color = (255, 0, 0)
        for box in gt_pts:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(compare_region, [pts], True, color, 1)

        return compare_region
    

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