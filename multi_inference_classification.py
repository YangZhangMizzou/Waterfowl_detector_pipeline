import os
import glob
from datetime import date

today = date.today()
dataset = 'Bird_I_Waterfowl_SpeciesClassification'
det_models = ['yolonas']
cla_models = ['mixmatch']
all_folders_dataset = glob.glob(os.path.join('/home/yangzhang/waterfowl_datasets',dataset,'test'))
csv_root = os.path.join('/home/yangzhang/waterfowl_datasets',dataset,'image_info.csv')

for folder in all_folders_dataset:
	for det_model in det_models:
		for cla_model in cla_models:
			save_dir = os.path.join('Result',str(today),dataset,det_model,cla_model)
			os.system('python inference_image_height.py --image_root {}  --out_dir {} --det_model {} --cla_model {} --image_ext {}'.format(folder,save_dir,det_model,cla_model,'jpg'))
