import os
import json
import time
import logging

from paddlex import create_model
from paddlex.repo_manager.repos.PaddleOCR.ppocr.utils.logging import get_logger
from exp_exist_label import draw_tables
from exp_ocr import TableOCR, cal_metrics


def test_indexes(data_dir, iou_thresh=0.5):
    log_dir = './output/exp/TableOcr'
    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_val.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'

    index_logger = get_logger(name='TableOcrTest', log_file=os.path.join(log_dir, f'test_table_magic_v2_cell_predictor_indexes_no_border_{iou_thresh}.log'), log_level=logging.DEBUG)

    model = create_model(model_name="RT-DETR-L_wireless_table_cell_det")

    imgs_info = val['images']
    anns = val['annotations']
    test_indexes = {'total_cells': 0, 'correct_cells': 0, 'total_pred_cells': 0, 'recall': 0, 'precision': 0, 'time_consume': 0, 'images_num': 0, 'per_image_time_consume': 0}  # correct cells meet the requirements where the iou > iou_thresh
    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        if 'no_border' not in img_name:
            continue
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)

        st = time.time()
        output = model.predict(img_path, threshold=0.3, batch_size=1)

        pred_bounds = []
        for res in output:
            pred_dicts = res._to_json()['res']['boxes']
            for k in range(len(pred_dicts)):
                pred_bounds.append(pred_dicts[k]['coordinate'])
        ed = time.time()

        for j in range(len(anns)):
            ann = anns[j]
            if ann['image_id'] != img_id:
                continue
            test_indexes['total_cells'] += 1
            x, y, w, h = ann['bbox']
            cell_box = (x, y, x + w, y + h)
            ans = cal_metrics(gt_cell_bound=cell_box, pred_bounds=pred_bounds, iou_thresh=iou_thresh)
            if ans:
                test_indexes['correct_cells'] += 1

        test_indexes['images_num'] += 1
        test_indexes['time_consume'] += (ed - st)
        test_indexes['total_pred_cells'] += len(pred_bounds)
        test_indexes['per_image_time_consume'] = round(test_indexes['time_consume'] / test_indexes['images_num'], 4)
        test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total_cells']
        test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']
        index_logger.debug(f'test_indexes: {test_indexes}')

    test_indexes['recall'] = test_indexes['correct_cells'] / test_indexes['total_cells']
    test_indexes['precision'] = test_indexes['correct_cells'] / test_indexes['total_pred_cells']

    index_logger.info(f'test_indexes: {test_indexes}')


def visualize_model_prediction(data_dir):
    save_dir = './output/RT-DETR-L_wireless_table_cell_det'
    imgs_dir = os.path.join(data_dir, "images")
    anns_path = os.path.join(data_dir, "annotations", "instance_val.json")
    with open(anns_path, 'r', encoding='utf8') as f:
        val = json.load(f)
    cat = val['categories']
    assert len(cat) == 1, 'number of category is more than one, please check dataset'

    table_ocr = TableOCR()

    model = create_model(model_name="RT-DETR-L_wireless_table_cell_det")

    imgs_info = val['images']
    anns = val['annotations']
    test_indexes = {'total_cells': 0, 'correct_cells': 0, 'total_pred_cells': 0, 'recall': 0, 'precision': 0}  # correct cells meet the requirements where the iou > iou_thresh
    for i in range(len(imgs_info)):
        image_info = imgs_info[i]
        img_name = image_info['file_name']
        img_id = image_info['id']
        img_path = os.path.join(imgs_dir, img_name)

        img = table_ocr.check_and_read_img(img_path=img_path)

        output = model.predict(img_path, threshold=0.3, batch_size=1)
        pred_pts = []
        for res in output:
            pred_dicts = res._to_json()['res']['boxes']
            for k in range(len(pred_dicts)):
                pred_pts.append(table_ocr.transform_x1y1x2y2_into_four_coordinates(pred_dicts[k]['coordinate']))
        pred_table = draw_tables(img=img, boxes=pred_pts)
        sp = os.path.join(path=os.path.join(save_dir, img_name))
        table_ocr.show_img(pred_table, sp=sp)


if __name__ == "__main__":
    mode = 'pc'  # 'pc' or 'aistudio'
    save_dir = "./output/exp"

    if mode == 'pc':
        data_dir = 'D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets'

    test_indexes(data_dir=data_dir, iou_thresh=0.3)
    # visualize_model_prediction(data_dir)