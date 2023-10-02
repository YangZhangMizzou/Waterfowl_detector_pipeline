import os
import glob
from datetime import date

today = date.today()
model_list = ['yolonas']
dataset = 'drone_collection_height'
dataset_root = '/home/yangzhang/waterfowl_datasets'
all_folders_dataset = glob.glob(os.path.join(dataset_root,dataset,'test','*'))
csv_root = os.path.join(dataset_root,dataset,'image_info.csv')
for det_model in model_list:
	for folder in all_folders_dataset:
		height = folder.split(os.sep)[-1]
		subset = folder.split(os.sep)[-3]
		save_dir = os.path.join('Result',str(today),subset,det_model,height)
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --csv_root {}'.format(folder,save_dir,det_model,'JPG',csv_root))
 