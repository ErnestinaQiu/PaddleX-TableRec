import os
import json
import shutil
import logging
from exp.TableRec import TableRec
from exp.ocr import TableOCR
import numpy as np
from exp_ocr import cal_metrics
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger


def test_analyse_frame_lines(data_dir):
    table_rec = TableRec(log_level=logging.DEBUG)

    # test in data dir
    save_dir = './output/exp/classify_table'
    os.makedirs(save_dir, exist_ok=True)
    border_save_dir = os.path.join(save_dir, 'border_wrong')
    os.makedirs(border_save_dir, exist_ok=True)
    no_border_save_dir = os.path.join(save_dir, 'no_border_wrong')
    os.makedirs(no_border_save_dir, exist_ok=True)

    # load val dataset
    ann_info = read_dataset(data_dir)
    imgs_info = ann_info['imgs_info']
    anns = ann_info['annotations']
    imgs_dir = ann_info['imgs_dir']

    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)
        img = table_rec.read_img(img_path)
        frame_lines = table_rec.classify_table(img)
        if len(frame_lines) == 0:
            continue
        frame_info = table_rec.analyse_frame_lines(img, frame_lines)
    return 0

def read_dataset(data_dir):
    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_val.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'

    imgs_info = val['images']
    anns = val['annotations']

    return {'imgs_info': imgs_info, 'annotations': anns, 'imgs_dir': imgs_dir}


def test_classify_table(data_dir):
    table_rec = TableRec(log_level=logging.INFO)

    # test in data dir
    save_dir = './output/exp/classify_table'
    os.makedirs(save_dir, exist_ok=True)
    border_save_dir = os.path.join(save_dir, 'border_wrong')
    os.makedirs(border_save_dir, exist_ok=True)
    no_border_save_dir = os.path.join(save_dir, 'no_border_wrong')
    os.makedirs(no_border_save_dir, exist_ok=True)

    # load val dataset
    ann_info = read_dataset(data_dir)
    imgs_info = ann_info['imgs_info']
    anns = ann_info['annotations']
    imgs_dir = ann_info['imgs_dir']

    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)
        img = table_rec.read_img(img_path)
        frame_lines = table_rec.classify_table(img)
        if len(frame_lines) == 0 and 'no_border' not in img_name:
            sp = os.path.join(border_save_dir, img_name)
            shutil.copy(img_path, sp)
        if len(frame_lines) != 0 and 'no_border' in img_name:
            sp = os.path.join(no_border_save_dir, img_name)
            shutil.copy(img_path, sp)


def analyse_frame_alignment(data_dir):
    table_rec = TableRec(log_level=logging.DEBUG)

    # test in data dir
    save_dir = './output/exp/classify_table'
    os.makedirs(save_dir, exist_ok=True)
    border_save_dir = os.path.join(save_dir, 'border_wrong')
    os.makedirs(border_save_dir, exist_ok=True)
    no_border_save_dir = os.path.join(save_dir, 'no_border_wrong')
    os.makedirs(no_border_save_dir, exist_ok=True)

    # load val dataset
    ann_info = read_dataset(data_dir)
    imgs_info = ann_info['imgs_info']
    anns = ann_info['annotations']
    imgs_dir = ann_info['imgs_dir']

    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        print(img_name)
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)
        img = table_rec.read_img(img_path)
        frame_lines = table_rec.classify_table(img)
        pred_bounds = table_rec.predict_three_lines_frame(img, frame_lines=frame_lines)
        table_rec.frame_alignment(img, bounds=pred_bounds)


def analyse_wrong_classify_samples():
    wrong_data_dir = 'D:/work/TableRec/PaddleX-TableRec/output/exp/classify_table/no_border_wrong'
    wrong_img_names = os.listdir(wrong_data_dir)
    table_rec = TableRec(log_level=logging.DEBUG)
    for i in range(len(wrong_img_names)):
        img_name = wrong_img_names[i]
        print(f'---------  img name: {img_name}')
        img_path = os.path.join(wrong_data_dir, img_name)
        img = table_rec.read_img(img_path)
        frame_lines = table_rec.classify_table(img=img)
        print(f'frame_lines: {frame_lines}')


def test_table_rec_border_table(data_dir):
    save_dir = 'D:/work/TableRec/PaddleX-TableRec/output/exp/test_border_predict'
    os.makedirs(save_dir, exist_ok=True)

    index_logger = get_logger(name='TableOcrTest', log_file=os.path.join(save_dir, f'test_table_magic_v2_cell_predictor_indexes_no_border.log'), log_level=logging.DEBUG)

    # load val dataset
    ann_info = read_dataset(data_dir)
    imgs_info = ann_info['imgs_info']
    anns = ann_info['annotations']
    imgs_dir = ann_info['imgs_dir']
    table_rec = TableRec(log_level=logging.DEBUG)

    table_ocr = TableOCR(log_level=logging.ERROR)

    bad_case_record = {'mean_acc': 0, 'imgs_num': 0, 'acc_list': [], 'img_names': []}
    bad_case_save_dir = os.path.join(save_dir, 'bad_case')
    os.makedirs(bad_case_save_dir, exist_ok=True)
    test_indexes = {'total_cells': 0, 'correct_cells': 0, 'total_pred_cells': 0, 'recall': 0, 'precision': 0, 'time_consume': 0, 'images_num': 0, 'per_image_time_consume': 0}  # correct cells meet the requirements where the iou > iou_thresh

    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']

        if 'no_border' in img_name:
            continue

        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)

        img = table_rec.read_img(img_path)
        frame_lines = table_rec.classify_table(img)
        pred_cells = table_rec.predict_three_lines_frame(img, frame_lines)  # list of bounds of pred cells

        img_indexes = {'correct_cells': 0, 'cells': 0, 'pred_cells': len(pred_cells), 'acc': 0, 'precision': 0}

        gt_boxes = []
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            test_indexes['total_cells'] += 1
            img_indexes['cells'] += 1
            x, y, w, h = ann['bbox']
            cell_box = (x, y, x + w, y + h)
            gt_boxes.append(cell_box)
            ans = cal_metrics(gt_cell_bound=cell_box, pred_bounds=pred_cells, iou_thresh=0.7)
            if ans:
                test_indexes['correct_cells'] += 1
                img_indexes['correct_cells'] += 1

        test_indexes['images_num'] += 1
        test_indexes['total_pred_cells'] += len(pred_cells)
        test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total_cells']
        test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']

        if img_indexes['acc'] < 0.7:
            bad_case_record['img_names'].append(img_name)
            bad_case_record['imgs_num'] += 1
            bad_case_record['acc_list'].append(img_indexes['acc'])
            bad_case_record['mean_acc'] = np.mean(bad_case_record['acc_list'])
            sp = os.path.join(bad_case_save_dir, img_name)
            gt_compare_img = table_ocr.draw_compare_table(img=img, gt_boxes=gt_boxes, pred_boxes=pred_cells)
            table_ocr.show_img(img=gt_compare_img, sp=sp)

        if i % 10:
            tmp_show_bad_case_record = {k: v for k, v in bad_case_record.items() if k != 'img_names'}
            index_logger.info(f'bad_case_record: {tmp_show_bad_case_record}')
            index_logger.debug(f'test_indexes: {test_indexes}')


def test_table_rec_no_border_table(data_dir):
    save_dir = 'D:/work/TableRec/PaddleX-TableRec/output/exp/test_no_border_predict_tactic_2'
    os.makedirs(save_dir, exist_ok=True)

    index_logger = get_logger(name='TableOcrTest1', log_file=os.path.join(save_dir, 'test_table_rec_indexes_no_border.log'), log_level=logging.DEBUG)

    # load val dataset
    ann_info = read_dataset(data_dir)
    imgs_info = ann_info['imgs_info']
    anns = ann_info['annotations']
    imgs_dir = ann_info['imgs_dir']
    table_rec = TableRec(log_level=logging.INFO)

    table_ocr = TableOCR(log_level=logging.ERROR)

    bad_case_record = {'mean_acc': 0, 'imgs_num': 0, 'acc_list': [], 'img_names': []}
    bad_case_save_dir = os.path.join(save_dir, 'bad_case')
    os.makedirs(bad_case_save_dir, exist_ok=True)
    test_indexes = {'total_cells': 0, 'correct_cells': 0, 'total_pred_cells': 0, 'recall': 0, 'precision': 0, 'time_consume': 0, 'images_num': 0, 'per_image_time_consume': 0}  # correct cells meet the requirements where the iou > iou_thresh

    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']

        if 'no_border' not in img_name:
            continue

        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)

        img = table_rec.read_img(img_path)
        frame_lines = table_rec.classify_table(img)
        pred_cells = table_rec.predict_no_frame(img, img_path=img_path)  # list of bounds of pred cells

        img_indexes = {'correct_cells': 0, 'cells': 0, 'pred_cells': len(pred_cells), 'acc': 0, 'precision': 0}

        gt_boxes = []
        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            test_indexes['total_cells'] += 1
            img_indexes['cells'] += 1
            x, y, w, h = ann['bbox']
            cell_box = (x, y, x + w, y + h)
            gt_boxes.append(cell_box)
            ans = cal_metrics(gt_cell_bound=cell_box, pred_bounds=pred_cells, iou_thresh=0.7)
            if ans:
                test_indexes['correct_cells'] += 1
                img_indexes['correct_cells'] += 1

        test_indexes['images_num'] += 1
        test_indexes['total_pred_cells'] += len(pred_cells)
        test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total_cells']
        test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']

        if img_indexes['acc'] < 0.7:
            bad_case_record['img_names'].append(img_name)
            bad_case_record['imgs_num'] += 1
            bad_case_record['acc_list'].append(img_indexes['acc'])
            bad_case_record['mean_acc'] = np.mean(bad_case_record['acc_list'])
            sp = os.path.join(bad_case_save_dir, img_name)
            gt_compare_img = table_ocr.draw_compare_table(img=img, gt_boxes=gt_boxes, pred_boxes=pred_cells)
            table_ocr.show_img(img=gt_compare_img, sp=sp)

        if i % 10:
            tmp_show_bad_case_record = {k: v for k, v in bad_case_record.items() if k != 'img_names'}
            index_logger.info(f'bad_case_record: {tmp_show_bad_case_record}')
            index_logger.debug(f'test_indexes: {test_indexes}')


if __name__ == "__main__":
    data_dir = "D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets"
    # test_classify_table(data_dir=data_dir)
    # analyse_wrong_classify_samples()
    # test_table_rec_border_table(data_dir=data_dir)
    # test_analyse_frame_lines(data_dir)
    # analyse_frame_alignment(data_dir)
    test_table_rec_no_border_table(data_dir)