"""
author: ErnestinaQiu
description: test exp ocr
"""
import os
import yaml
from exp.ocr import TableOCR
from exp_exist_label import check_and_read
from paddlex.utils.config import parse_config
from paddlex import create_pipeline


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
    ocr_res = table_ocr.get_ocr_text_box(img_path=img_path, save_dir=save_dir)
    return ocr_res

def test_my_ocr_img_dir(img_dir, save_dir=None):
    table_ocr = TableOCR()
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        ocr_res = table_ocr.get_ocr_text_box(img_path=img_path, save_dir=save_dir)
    return ocr_res

def test_shrink_box():
    img_path = 'D:/work/TableRec/PaddleX-TableRec/output/exp/ocr_text_box/border_bottom_18_M2YV6IY0NXGYQQURBAVT_39.jpg'
    table_ocr = TableOCR()
    img = table_ocr.check_and_read_img(img_path=img_path)
    table_ocr.show_img(img)
    shrink_box = table_ocr.shrink_text_box(box_img=img)
    table_ocr.show_img(shrink_box)

if __name__ == "__main__":
    save_dir = "./output/exp"
    img_dir = os.path.join('D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets/images')
    # img_path = os.path.join('D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets/images', 'border_bottom_18_M2YV6IY0NXGYQQURBAVT.jpg')
    # ocr_res = test_ocr(img_path=img_path)
    # ocr_res = test_ocr_pipeline(img_path=img_path, save_dir=save_dir)
    # img_path = 'D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets/images/border_bottom_0_8CTA75BO6N49PDO4WLJD.jpg'
    # test_my_ocr(img_path=img_path, save_dir=save_dir)
    test_my_ocr_img_dir(img_dir=img_dir, save_dir=save_dir)
    # test_shrink_box()
