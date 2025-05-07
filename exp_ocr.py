"""
author: ErnestinaQiu
description: test exp ocr
"""
import os
import cv2
import yaml
from exp.ocr import TableOCR
from exp_exist_label import check_and_read
from paddlex.utils.config import parse_config
from paddlex import create_pipeline
from exp_exist_label import draw_tables


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
        rec_boxes.append(table_ocr.transform_ocr_box_into_four_coordinates(ocr_box=rec_box))
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
        rec_boxes.append(table_ocr.transform_ocr_box_into_four_coordinates(ocr_box=rec_box))
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
    import shutil
    mode = 'aistudio'  # 'pc' or 'aistudio'
    save_dir = "./output/exp"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if mode == 'pc':
        data_dir = 'D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless'
    elif mode == 'aistudio':
        data_dir = './data/dataset'

    # img_dir = os.path.join(data_dir, 'table-rec-v2-pipe_practical_datasets/images')
    # img_path = os.path.join(data_dir, 'table-rec-v2-pipe_practical_datasets/images', 'border_bottom_18_M2YV6IY0NXGYQQURBAVT.jpg')
    # ocr_res = test_ocr(img_path=img_path)
    # ocr_res = test_ocr_pipeline(img_path=img_path, save_dir=save_dir)

    img_path = os.path.join(data_dir, 'table-rec-v2-pipe_practical_datasets/images/border_bottom_0_8CTA75BO6N49PDO4WLJD.jpg')

    # test_my_ocr(img_path=img_path, save_dir=save_dir)
    # test_my_ocr_img_dir(img_dir=img_dir, save_dir=save_dir)
    # test_shrink_box()
    test_split_into_groups(img_path=img_path, save_dir=os.path.join(save_dir, 'split_into_groups'), platform='aistudio')
