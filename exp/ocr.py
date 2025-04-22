"""
author: ErnestinaQiu
description: get ocr result from PaddleOCR repos
"""
import os
import cv2
import numpy as np
from logging import DEBUG, ERROR
from typing import Any, List, Optional
from paddlex import create_pipeline
from paddlex.utils.logging import logging


class TableOCR:
    def __init__(
        self,
        logger_flag=ERROR,
    ) -> None:
        """an ocr and its postprocess for table

        Returns:
            _type_: _description_
        """
        self.logger_flag = logger_flag
        self.pipeline = create_pipeline(pipeline="OCR")

    def get_ocr_text_box(self, img_path: str = None, save_dir: str = None):
        img_name = os.path.basename(img_path).split('.')[0]
        save_dir = os.path.join(save_dir, 'ocr_text_box', img_name)
        os.makedirs(save_dir, exist_ok=True)
        ocr_res = self.get_img_ocr_result(img_path=img_path, save_dir=save_dir)
        ocr_res_json = ocr_res._to_json()['res']
        rec_boxes = ocr_res_json['rec_boxes']
        img = self.check_and_read_img(img_path=img_path)
        for i in range(len(rec_boxes)):
            if self.logger_flag == DEBUG:
                print('-'*10, i, '-'*10)
            box = rec_boxes[i]
            box_img = self.get_box_img(box=box, img=img)
            assert box_img.shape[0] != 0 and box_img.shape[1] != 0, f'box_img is empty, img_path: {img_path}'
            shrink_box_img = self.shrink_text_box(box_img=box_img)
            assert shrink_box_img.shape[0] != 0 and shrink_box_img.shape[1] != 0, f'shrink_box_img is empty, img_path: {img_path}'
            if self.logger_flag == DEBUG:
                self.show_img(img=box_img)
                self.show_img(img=shrink_box_img)
            if save_dir:
                box_img_name = '.'.join(['_'.join([img_name, str(i)]), 'jpg'])
                box_img_path = os.path.join(save_dir, box_img_name)
                cv2.imwrite(box_img_path, shrink_box_img)

    def show_img(self, img: np.ndarray):
        cv2.imshow('Image', img)

        # Wait for a key press and then close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_box_img(self, box: List, img: np.ndarray):
        box_img = img[box[1]:box[3], box[0]:box[2]]
        return box_img

    def check_and_read_img(self, img_path: str):
        assert os.path.exists(img_path), "img_path doesn't exists"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def shrink_text_box(self, box_img):
        """only consider the table line is vertical or horizontal

        Args:
            box_img (_type_): _description_
        """
        line_thresh = 0.05 * box_img.shape[1]
        margin_thresh = 0.95 * box_img.shape[1]

        if self.logger_flag == DEBUG:
            print(f'--- ver \nline_thresh: {line_thresh}, margin_thresh: {margin_thresh}')

        _, binary_image = cv2.threshold(box_img, 127, 1, cv2.THRESH_BINARY)

        # vertical analysis
        vertical_accum = []
        for i in range(binary_image.shape[0]):
            vertical_accum.append(np.sum(binary_image[i, :]))

        if self.logger_flag == DEBUG:
            print(f'vertical_accum: {vertical_accum}')

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
            return box_img

        ver_main_scope_len = 0
        ver_main_scope = [0, 0]
        for j in range(len(detail['margin'])-1):
            text_line_st = detail['margin'][j][1]
            text_line_ed = detail['margin'][j+1][0]
            if text_line_ed - text_line_st >= ver_main_scope_len:
                ver_main_scope_len = text_line_ed - text_line_st
                ver_main_scope = [text_line_st, text_line_ed]

        if self.logger_flag == DEBUG:
            print(f'ver_main_scope: {ver_main_scope}, \ndetail: {detail}')

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
            print(f'--- hor  \nline_thresh: {line_thresh}, margin_thresh: {margin_thresh}\nhorizontal_accum: {horizontal_accum}')

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
            print(f'detail: {detail}')

        if len(detail['margin']) == 1 and len(detail['line']) == 0:
            return box_img[ver_main_scope[0]:ver_main_scope[1], :]


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
        
        shrink_box = box_img[ver_main_scope[0]:ver_main_scope[1], hor_main_scope[0]:hor_main_scope[1]]

        return shrink_box

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
