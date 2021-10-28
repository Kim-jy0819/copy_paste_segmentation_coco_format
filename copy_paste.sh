pip install imgviz
pip install Shapely # requirement.txt
python get_coco_mask.py  --input_dir ../input/data/ --split train_all # segmentation mask img 생성
python mask_convert_json.py --input_dir ../input/data/ --output_dir ../input/data/ # mask img로부터 랜덤하게 copy_paste 
python create-copy-paste-dataset.py # copy_paste mask로부터 coco format json 파일 만들기