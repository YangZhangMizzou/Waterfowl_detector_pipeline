import os
import glob
from datetime import date

today = date.today()
model_list = ['yolonas']
dataset = 'drone_collection_height'
all_folders_dataset = glob.glob(os.path.join('example_images',dataset,'test','*'))
csv_root = os.path.join('example_images',dataset,'image_info.csv')
for det_model in model_list:
	for folder in all_folders_dataset:
		height = folder.split(os.sep)[-1]
		subset = folder.split(os.sep)[-3]
		save_dir = os.path.join('Result',str(today),subset,height)
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --csv_root {}'.format(folder,save_dir,det_model,'JPG',csv_root))
 