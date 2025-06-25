from paddlex.inference.pipelines.table_recognition.utils import TableRec
from paddlex import create_pipeline, create_model
# from paddlex.inference.pipelines import create_pipeline
# from paddlex.model import create_model


def test_table_rec():
    img_path = 'D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets/images/no_border_479_32SIVUBCY6GP06O6Y3G0.jpg'
    ocr_pipeline = create_pipeline(pipeline="OCR")
    cell_det = create_model(model_name="RT-DETR-L_wireless_table_cell_det")
    ocr_result = next(ocr_pipeline.predict(img_path))
    cell_result = cell_det.predict(img_path)
    table_rec = TableRec()
    pred_bounds = table_rec.predict(img_path, ocr_result, cell_result)
    print(pred_bounds)

if __name__ == "__main__":
    test_table_rec()