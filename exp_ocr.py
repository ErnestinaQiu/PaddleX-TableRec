"""
author: ErnestinaQiu
description: test exp ocr
"""
import os
import cv2
import time
import yaml
import json
import random
import logging
import numpy as np
from PIL import Image, ImageDraw
from exp.ocr import TableOCR, compute_iou_cal_metrics
from exp_exist_label import check_and_read
from paddlex.utils.config import parse_config
from paddlex import create_pipeline
from exp_exist_label import draw_tables
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


def test_ocr_pipeline(img_path, save_dir):
    pipeline = create_pipeline(pipeline="OCR")
    img_name = os.path.basename(img_path)
    output = pipeline.predict(
        img_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    for res in output:
        print(res)
        res.print()
        res.save_to_img(os.path.join(save_dir, "OCR_table_config", img_name))
        res.save_to_json(os.path.join(save_dir, "OCR_table_config", '.'.join([img_name.split('.')[0], 'json'])))
    return output


def test_my_ocr(img_path, save_dir=None):
    table_ocr = TableOCR()

    img = table_ocr.check_and_read_img(img_path=img_path)
    # origin img
    ocr_res = table_ocr.get_img_ocr_result(img_path=img_path)
    ocr_res_json = ocr_res._to_json()['res']
    origin_rec_boxes = ocr_res_json['rec_boxes']
    rec_boxes = []
    for rec_box in origin_rec_boxes:
        rec_boxes.append(table_ocr.transform_x1y1x2y2_into_four_coordinates(ocr_box=rec_box))
    origin_table = draw_tables(img=img, boxes=rec_boxes)
    table_ocr.show_img(origin_table)

    res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path)

    res_pts_list = []
    for i in range(len(res_boxes)):
        # if i != 3:
        #     continue
        print(f'----- i: {i}')
        box = res_boxes[i]
        res_pts = table_ocr.box_to_four_coordinates(box)
        res_pts_list.append(res_pts)
        # tmp_vis_img = draw_tables(img=img, boxes=[res_pts])
        # table_ocr.show_img(tmp_vis_img)


    vis_img = draw_tables(img=img, boxes=res_pts_list)
    table_ocr.show_img(vis_img)

    return res_pts_list

def test_split_into_groups(img_path, platform, save_dir=None):
    # img save dir
    if save_dir is not None:
        img_name = os.path.basename(img_path).split('.')[0]

        img_dir = os.path.join(save_dir, 'proj_analysis', img_name)
        os.makedirs(img_dir, exist_ok=True)
    else:
        img_dir = None

    table_ocr = TableOCR(platform=platform, save_dir=img_dir)
    img = table_ocr.check_and_read_img(img_path=img_path)
    # origin img
    ocr_res = table_ocr.get_img_ocr_result(img_path=img_path)
    ocr_res_json = ocr_res._to_json()['res']
    origin_rec_boxes = ocr_res_json['rec_boxes']
    rec_boxes = []
    for rec_box in origin_rec_boxes:
        rec_boxes.append(table_ocr.transform_x1y1x2y2_into_four_coordinates(ocr_box=rec_box))
    origin_table = draw_tables(img=img, boxes=rec_boxes)

    if save_dir is not None:
        sp = os.path.join(img_dir, 'table_ocr_img.png')
    else:
        sp = None
    table_ocr.show_img(origin_table, sp=sp)

    res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path)

    res_pts_poly = []
    for _box in res_boxes:
        __box = table_ocr.box_to_four_coordinates(_box)
        res_pts_poly.append(__box)
    mod_shrinked_table = draw_tables(img=img, boxes=res_pts_poly)

    if save_dir is not None:
        sp = os.path.join(img_dir, 'table_ocr_mod_shrinked_img.png')
    else:
        sp = None
    table_ocr.show_img(mod_shrinked_table, sp=sp)

    canvas = table_ocr.ocr_box_canvas(text_boxes=res_boxes, img_shape=img.shape)

    if save_dir is not None:
        canvas_sp = os.path.join(img_dir, 'table_ocr_text_boxes.png')
    else:
        canvas_sp = None
    canvas = canvas * 255
    table_ocr.show_img(canvas, sp=canvas_sp)

    table_ocr.analysis_canvas(canvas=canvas, save_dir=img_dir)

    table_ocr.split_into_subgraph(canvas=canvas, text_boxes=res_boxes, img=img)

    table_ocr.split_into_region(canvas=canvas, text_boxes=res_boxes, img=img)


def analyse_deal_big_region_frame(data_img_dir, platform, save_dir=None):
    data_dir = "D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets"
    anns_path = os.path.join(data_dir, "annotations", "instance_train.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    anns = val['annotations']
    imgs_info = val['images']

    log_file = os.path.join(save_dir, 'info.log')
    table_ocr = TableOCR(platform=platform, save_dir=None, log_level=logging.INFO, log_file=log_file)
    for img_name in os.listdir(data_img_dir):
        if 'no_border' not in img_name:
            continue

        for a in imgs_info:
            tmp_img_name = a['file_name']
            if tmp_img_name != img_name:
                continue
            img_id = a['id']

            ann_boxes = []
            for j in range(len(anns)):
                ann = anns[j]
                if ann['image_id'] != img_id:
                    continue
                origin_x, origin_y, w, h = ann['bbox']
                box = [(origin_x, origin_y), (origin_x + w, origin_y), (origin_x + w, origin_y + h), (origin_x, origin_y + h)]
                ann_boxes.append(box)

        img_path = os.path.join(data_img_dir, img_name)
        img_base_name = os.path.basename(img_path).split('.')[0]
        save_img_dir = os.path.join(save_dir, 'deal_big_region_frame', img_base_name)
        os.makedirs(save_img_dir, exist_ok=True)
        table_ocr.save_dir = save_img_dir
        img = table_ocr.check_and_read_img(img_path=img_path)
        # origin img
        ocr_res = table_ocr.get_img_ocr_result(img_path=img_path)
        ocr_res_json = ocr_res._to_json()['res']
        origin_rec_boxes = ocr_res_json['rec_boxes']
        rec_boxes = []
        for rec_box in origin_rec_boxes:
            rec_boxes.append(table_ocr.transform_x1y1x2y2_into_four_coordinates(ocr_box=rec_box))
        origin_table = draw_tables(img=img, boxes=rec_boxes)

        if save_dir is not None:
            sp = os.path.join(save_img_dir, 'table_ocr_img.png')
        else:
            sp = None
        table_ocr.show_img(origin_table, sp=sp)

        res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path)

        canvas = table_ocr.ocr_box_canvas(text_boxes=res_boxes, img_shape=img.shape)

        if save_dir is not None:
            canvas_sp = os.path.join(save_img_dir, 'table_ocr_text_boxes.png')
        else:
            canvas_sp = None
        canvas = canvas * 255
        table_ocr.show_img(canvas, sp=canvas_sp)

        regions = table_ocr.split_into_region(canvas=canvas, text_boxes=res_boxes, img=img)
        region_pts_poly = []
        for d in regions:
            pts = table_ocr.transform_x1y1x2y2_into_four_coordinates(d['bound'])
            region_pts_poly.append(pts)
        region_table = draw_tables(img=img, boxes=region_pts_poly)

        if save_dir is not None:
            sp = os.path.join(save_img_dir, 'region_table.png')
        else:
            sp = None
        table_ocr.show_img(region_table, sp=sp)

        regions_info = table_ocr.deal_big_region_frame(region=regions, img_shape=img.shape)
        modify_frame_regions = regions_info['region']

        region_pts_poly = []
        for d in modify_frame_regions:
            pts = table_ocr.transform_x1y1x2y2_into_four_coordinates(d['bound'])
            region_pts_poly.append(pts)
        region_table = draw_tables(img=img, boxes=region_pts_poly)

        if save_dir is not None:
            sp = os.path.join(save_img_dir, 'modify_frame_region_table.png')
        else:
            sp = None
        table_ocr.show_img(region_table, sp=sp)

        # plot big region frame
        row_bounds = regions_info['row_bounds']
        col_bounds = regions_info['col_bounds']
        big_frame_img = img.copy()
        for p in range(len(row_bounds)):
            big_frame_img[row_bounds[p], col_bounds[0]: col_bounds[-1]] = 125
        for q in range(len(col_bounds)):
            big_frame_img[row_bounds[0]: row_bounds[-1], col_bounds[q]] = 125

        if save_dir is not None:
            sp = os.path.join(save_img_dir, 'big_frame_table.png')
        else:
            sp = None
        table_ocr.show_img(big_frame_img, sp=sp)

        new_regions_info = table_ocr.merge_same_cells_deal_complex_region(regions=regions, img=img)
        region_pts_poly = []
        for k in range(len(new_regions_info)):
            bound = new_regions_info[k]['bound']
            if bound is None:
                continue
            pts = table_ocr.transform_x1y1x2y2_into_four_coordinates(bound)
            region_pts_poly.append(pts)
        new_region_table = draw_tables(img=img, boxes=region_pts_poly)

        if save_dir is not None:
            sp = os.path.join(save_img_dir, 'new_region_table.png')
        else:
            sp = None
        table_ocr.show_img(new_region_table, sp=sp)

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w = image.height, image.width
        img_top = image.copy()
        random.seed(0)

        draw_top = ImageDraw.Draw(img_top)
        draw_top = ImageDraw.Draw(img_top)

        for pts in region_pts_poly:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_top.polygon(pts, fill=color)

        img_top = Image.blend(image, img_top, 0.5)
        compare_region = np.array(img_top)

        for box in ann_boxes:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(compare_region, [pts], True, color, 1)
        if save_dir is not None:
            sp = os.path.join(save_img_dir, 'compare_new_region_table.png')
        else:
            sp = None
        table_ocr.show_img(compare_region, sp=sp)

        print(f'out into {sp}')


def test_split_and_merge_index_iou(data_dir, platform, iou_thresh=0.5, save_dir=None):
    log_dir = './output/exp/TableOcr'
    os.makedirs(log_dir, exist_ok=True)
    index_logger = get_logger(name='TableOcrTest', log_file=os.path.join(log_dir, 'test_indexes.log'), log_level=logging.DEBUG)

    table_ocr = TableOCR(platform=platform, save_dir=None, log_level=logging.ERROR)
    test_indexes = {'total_cells': 0, 'correct_cells': 0, 'total_pred_cells': 0, 'iou_thresh': iou_thresh}   # correct cells meet the requirements where the iou > iou_thresh

    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_val.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    imgs_info = val['images']
    anns = val['annotations']
    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        if 'no_border' not in img_name:
            continue
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)

        img = table_ocr.check_and_read_img(img_path=img_path)
        res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path)
        canvas = table_ocr.ocr_box_canvas(text_boxes=res_boxes, img_shape=img.shape)
        canvas = canvas * 255
        regions = table_ocr.split_into_region(canvas=canvas, text_boxes=res_boxes, img=img)
        new_regions = table_ocr.merge_and_split(regions=regions, img=img)

        test_indexes['total_pred_cells'] += len(regions)
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            test_indexes['total_cells'] += 1
            x, y, w, h = ann['bbox']
            cell_box = (x, y, x + w, y + h)
            for bound in new_regions:
                iou = compute_iou(box1=cell_box, box2=bound)
                if iou > iou_thresh:
                    test_indexes['correct_cells'] += 1
                    break
        index_logger.debug(f'test_indexes: {test_indexes}')

    test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total']
    test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']

    index_logger.info(f'test_indexes: {test_indexes}')


def test_deal_big_region_frame(data_dir, platform, save_dir=None, iou_thresh=0.):
    save_dir = './output/exp/deal_big_region_frame'
    os.makedirs(save_dir, exist_ok=True)
    index_logger = get_logger(name='TableOcrTest', log_file=os.path.join(save_dir, 'test_indexes.log'), log_level=logging.DEBUG)

    table_ocr = TableOCR(platform=platform, save_dir=save_dir, log_level=logging.CRITICAL)
    test_indexes = {'total_cells': 0, 'correct_cells': 0, 'total_pred_cells': 0, 'recall': 0, 'precision': 0, 'time_consume': 0, 'images_num': 0, 'per_image_time_consume': 0}  # correct cells meet the requirements where the iou > iou_thresh

    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_val.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    imgs_info = val['images']
    anns = val['annotations']
    for i in range(len(imgs_info)):
        st = time.time()
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        if 'no_border' not in img_name:
            continue
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)

        regions = table_ocr.predict(img_path=img_path)
        ed = time.time()
        test_indexes['images_num'] += 1
        test_indexes['time_consume'] += ed - st
        test_indexes['total_pred_cells'] += len(regions)
        test_indexes['per_image_time_consume'] = round(test_indexes['time_consume'] / test_indexes['images_num'], 4)
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            test_indexes['total_cells'] += 1
            x, y, w, h = ann['bbox']
            cell_box = (x, y, x + w, y + h)
            region_bounds = [d['bound'] for d in regions]
            ans = cal_metrics(gt_cell_bound=cell_box, pred_bounds=region_bounds, iou_thresh=iou_thresh)
            if ans:
                test_indexes['correct_cells'] += 1

        test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total_cells']
        test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']
        index_logger.debug(f'test_indexes: {test_indexes}')

    test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total_cells']
    test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']

    index_logger.info(f'test_indexes: {test_indexes}')


def cal_metrics(gt_cell_bound, pred_bounds, iou_thresh):
    for bound in pred_bounds:
        iou = compute_iou_cal_metrics(box1=bound, box2=gt_cell_bound)
        if iou >= iou_thresh:
            return 1
    return 0


def test_my_ocr_img_dir(img_dir, save_dir=None):
    table_ocr = TableOCR()
    table_boxes_img_dir = os.path.join(save_dir, 'shrink_boxes')
    os.makedirs(table_boxes_img_dir, exist_ok=True)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        res_boxes = table_ocr.get_ocr_text_boxes(img_path=img_path, save_dir=None)
        res_pts = []
        for box in res_boxes:
            res_pts.append(table_ocr.box_to_four_coordinates(box))
        img = table_ocr.check_and_read_img(img_path=img_path)
        vis_img = draw_tables(img=img, boxes=res_pts)
        shrink_box_img_path = os.path.join(table_boxes_img_dir, img_name)
        cv2.imwrite(shrink_box_img_path, vis_img)
        table_ocr.show_img(vis_img)
    return None


def test_shrink_box():
    img_path = 'D:/work/TableRec/PaddleX-TableRec/output/exp/ocr_text_box/border_bottom_18_M2YV6IY0NXGYQQURBAVT_39.jpg'
    table_ocr = TableOCR()
    img = table_ocr.check_and_read_img(img_path=img_path)
    table_ocr.show_img(img)
    shrink_box = table_ocr.shrink_text_box(box_img=img)
    table_ocr.show_img(shrink_box)


def draw_shrink_box_img_dir(img_dir, save_dir):
    table_ocr = TableOCR()
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        ocr_res = table_ocr.get_ocr_text_box(img_path=img_path, save_dir=save_dir)
    return ocr_res


if __name__ == "__main__":
    mode = 'pc'  # 'pc' or 'aistudio'
    save_dir = "./output/exp"

    if mode == 'pc':
        data_dir = 'D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets'
    elif mode == 'aistudio':
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        data_dir = './data/dataset'

    img_dir = os.path.join(data_dir, 'images')
    # img_path = os.path.join(data_dir, 'table-rec-v2-pipe_practical_datasets/images', 'border_bottom_18_M2YV6IY0NXGYQQURBAVT.jpg')
    # ocr_res = test_ocr(img_path=img_path)
    # ocr_res = test_ocr_pipeline(img_path=img_path, save_dir=save_dir)

    # img_path = os.path.join(data_dir, 'table-rec-v2-pipe_practical_datasets/images/border_bottom_0_8CTA75BO6N49PDO4WLJD.jpg')

    # test_my_ocr(img_path=img_path, save_dir=save_dir)
    # test_my_ocr_img_dir(img_dir=img_dir, save_dir=save_dir)
    # test_shrink_box()
    # test_split_into_groups(img_path=img_path, save_dir=os.path.join(save_dir, 'split_into_region'), platform='pc')

    # for img_name in os.listdir(img_dir):
    #     img_path = os.path.join(img_dir, img_name)
    #     test_split_into_groups(img_path=img_path, save_dir=os.path.join(save_dir, 'split_into_region'), platform='aistudio')
    # test_split_and_merge_data_dir(data_img_dir=img_dir, platform='aistudio', save_dir=os.path.join(save_dir, 'split_into_region'))

    # test_split_and_merge_index_iou(data_dir=data_dir, platform='aistudio', iou_thresh=0.5)
    # test_deal_big_region_frame(data_dir=data_dir, platform='aistudio', iou_thresh=0.5)
    # analyse_deal_big_region_frame(data_img_dir=img_dir, platform='aistudio', save_dir=os.path.join(save_dir, 'deal_big_region_frame'))
    test_deal_big_region_frame(data_dir=data_dir, platform='aistudio', save_dir=os.path.join(save_dir, 'deal_big_region_frame_test'))
