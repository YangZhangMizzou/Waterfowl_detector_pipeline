import os
import glob

model_list = ['yolo','faster','retinanet']
dataset = 'drone_collection_habitat'
# dataset = 'drone_collection_height'
# dataset = 'drone_collection_dataset'
all_folders_dataset = glob.glob(os.path.join('example_images',dataset,'test','ice'))
csv_root = os.path.join('example_images',dataset,'image_info.csv')

for det_model in model_list:
	for folder in all_folders_dataset:
		category = os.path.split(folder)[-1]
		save_dir = os.path.join('Result','0413',dataset,det_model,category)
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --csv_root {}'.format(folder,save_dir,det_model,'JPG',csv_root))
 
