"""
author: ErnestinaQiu
description: get ocr result from PaddleOCR repos
"""
import os
import gc
import cv2
import random
from copy import deepcopy
import numpy as np
from logging import NOTSET, DEBUG, INFO, ERROR
from typing import Any, List, Optional, Tuple
from paddlex import create_pipeline
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


class TableOCR:
    def __init__(
        self,
        log_level=DEBUG,
        log_file='./output/exp/logs/debug.log',
        platform='aistudio',
        save_dir=None
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

    def get_ocr_text_boxes(self, img_path: str = None, save_dir: str = None):
        """_summary_

        Args:
            img_path (str, optional): image path. Defaults to None.
            save_dir (str, optional): directory to save. Defaults to None.

        Returns:
            shrink_boxes (list): box [x, y, w, h]
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
            if save_dir or self.save_dir:
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
                if save_dir or self.save_dir:
                    tmp_box_img = img[tmp_box[1]: tmp_box[1] + tmp_box[3], tmp_box[0]: tmp_box[0] + tmp_box[2]]
                    assert tmp_box_img != [], f'rec_boxes {i}, new_shrink_boxes {k}, tmp_box_img: {tmp_box_img}, tmp_box: {tmp_box}, img.shape: {img.shape}'
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
            box (list): [x, y, w, h]
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

        for box in text_boxes:
            x, y, w, h = box
            for n in range(len(row_subgraphs)):
                subgraph_scope = row_subgraphs[str(n)]['scope']
                if y >= subgraph_scope[0] and y + h <= subgraph_scope[1]:
                    row_subgraphs[str(n)]['text_boxes'].append(box)
                elif y + h <= subgraph_scope[0] or y >= subgraph_scope[1]:
                    pass
                else:
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
                    if x >= subgraph_scope[0] and x < subgraph_scope[1] and x + w > subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - x)/w, 2)
                    elif x < subgraph_scope[0] and x + w > subgraph_scope[0] and x + w <= subgraph_scope[1]:
                        iou = round((x+w-subgraph_scope[0])/w, 2)
                    elif x <= subgraph_scope[0] and x + w > subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - subgraph_scope[0])/w, 2)
                    if iou >= iou_thresh:
                        col_subgraphs[str(m)]['text_boxes'].append(box)

        subgraphs = {'row': row_subgraphs, 'col': col_subgraphs}

        if self.logger_flag == DEBUG and img is not None:
            from PIL import Image, ImageDraw, ImageFont
            # row
            i = 0
            row_subgraph_canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            row_img = row_subgraph_canvas.copy()
            random.seed(0)
            draw_row_img = ImageDraw.Draw(row_img)

            for i in range(len(row_subgraphs)):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                scope = row_subgraphs[str(i)]['scope']
                scope_pts = self.box_to_four_coordinates(box=[0, scope[0], img.shape[1], scope[1] - scope[0]])
                draw_row_img.polygon(scope_pts, fill=color)

                text_boxes = row_subgraphs[str(i)]['text_boxes']
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

            for i in range(len(col_subgraphs)):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                scope = col_subgraphs[str(i)]['scope']
                scope_pts = self.box_to_four_coordinates(box=[scope[0], 0, scope[1] - scope[0], img.shape[0]])
                draw_col_img.polygon(scope_pts, fill=color)

                text_boxes = col_subgraphs[str(i)]['text_boxes']
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
                new_text_boxes.append([new_st, y, bonds[n][0], h])
                tmp_box_img = box_img[0:h, new_st-x:new_st-x+bonds[n][0]]
                assert tmp_box_img != [], f'[0:h, new_st-x:new_st-x+bonds[n][0]]: [0:{h}, {new_st-x}:{new_st-x+ bonds[n][0]}], box_img: {box_img.shape}, bonds: {bonds}, n: {n}'
            elif n!= 0:
                new_text_boxes.append([new_st, y, bonds[n][0] - bonds[n-1][1], h])
                tmp_box_img = box_img[0:h, new_st-x:new_st-x+bonds[n][0] - bonds[n-1][0]]
                assert tmp_box_img != [], f'[0:h, new_st-x:new_st-x+bonds[n][0] - bonds[n-1][1]]: [0:{h}, {new_st-x}:{new_st-x+bonds[n][0] - bonds[n-1][1]}], box_img: {box_img.shape}, bonds: {bonds}, n: {n}'

            new_box_imgs.append(tmp_box_img)
            new_st = x + bonds[n][1]

        if new_st != len(hor_proj) - 1:
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