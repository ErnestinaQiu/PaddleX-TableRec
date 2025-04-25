"""
author: ErnestinaQiu
description: get ocr result from PaddleOCR repos
"""
import os
import cv2
import numpy as np
from logging import NOTSET, DEBUG, ERROR
from typing import Any, List, Optional, Tuple
from paddlex import create_pipeline
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


class TableOCR:
    def __init__(
        self,
        log_level=ERROR,
        log_file='./output/exp/logs/debug.log'
    ) -> None:
        """an ocr and its postprocess for table

        Returns:
            _type_: _description_
        """
        self.logger_flag = log_level
        self.logger = get_logger(name='ocrtable', log_file=log_file, log_level=log_level)
        self.pipeline = create_pipeline(pipeline="OCR")

    def get_ocr_text_boxes(self, img_path: str = None, save_dir: str = None):
        """_summary_

        Args:
            img_path (str, optional): image path. Defaults to None.
            save_dir (str, optional): directory to save. Defaults to None.

        Returns:
            _type_: _description_
        """
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
            assert shrink_box_img.shape[0] != 0 and shrink_box_img.shape[1] != 0, f'shrink_box_img is empty, img_path: {img_path}'
            shrink_boxes.append(shrink_box)
            if self.logger_flag == NOTSET:
                self.show_img(img=box_img)
                self.show_img(img=shrink_box_img)
            if save_dir:
                box_img_name = '.'.join(['_'.join([img_name, str(i)]), 'jpg'])
                box_img_path = os.path.join(save_dir, box_img_name)
                if os.path.exists(box_img_path):
                    pass
                else:
                    cv2.imwrite(box_img_path, shrink_box_img)
        return shrink_boxes

    def show_img(self, img: np.ndarray, sp: str = None):
        cv2.imshow('Image', img)

        # Wait for a key press and then close all windows
        if self.logger_flag == NOTSET:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if sp:
            cv2.imwrite(sp, img)


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
        assert os.path.exists(img_path), "img_path doesn't exists"
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
        for res in output:
            if save_dir:
                img_name = os.path.basename(img_path)
                os.makedirs(save_dir, exist_ok=True)
                img_sp = os.path.join(save_dir, img_name)
                json_sp = os.path.join(save_dir, ".".join([img_name, 'json']))
                res.save_to_img(img_sp)
                res.save_to_json(json_sp)

        return res

    def box_to_four_coordinates(self, box: List):
        """transform box from [x, y, w, h] to four points, x is vertical axis, and y is horizontal axis

        Args:
            box (list): [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        Returns:
            pts (list): four points of polygon
        """
        origin_x, origin_y, w, h = box
        pts = [(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)]
        return pts

    def transform_ocr_box_into_four_coordinates(self, ocr_box: List):
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
            self.show_img(img=vis_blank_canvas)
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

    def split_into_row_subgraph(self, canvas):
        """ split text boxes into rows

        Args:
            canvas (np.ndarray): blank matrix with text box as 1
        Returns:
            box_groups (List): list of group of text boxes
        """
        row_proj = []
        for i in range(canvas.shape[0]):
            row_proj.append(np.sum(canvas[i, :]))

        # for j in range(canvas.shape[0]):
            

        pass

    def split_into_column_subgraph(self):
        """ split text boxes into rows
        """

        pass

    def split_into_cell_subgraph(self):
        """ split text boxes into cells
        """
        pass
