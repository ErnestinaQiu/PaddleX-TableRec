from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="D:/work/TableRec/paddlex/test/data/table-rec-v2-pipe_practical_datasets_wireless/table-rec-v2-pipe_practical_datasets/images/border_bottom_3_RULO0JIY5ZQDYSMCK617.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_ocr_results_with_table_cells=True,
    use_e2e_wireless_table_rec_model=True,
    use_wireless_table_cells_trans_to_html=True,
)

for res in output:
    res.print()
    res.save_to_img("./output/table_rec_v2")
    res.save_to_xlsx("./output/table_rec_v2")
    res.save_to_html("./output/table_rec_v2")
    res.save_to_json("./output/table_rec_v2")