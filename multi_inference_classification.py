import os
import glob
from datetime import date

today = date.today()
dataset = 'Bird_I_Waterfowl_SpeciesClassification'
dataset_root = '/home/yangzhang/waterfowl_datasets'#change it to your dataset.

det_models = ['yolo','faster']
cla_models = ['mixmatch']
all_folders_dataset = glob.glob(os.path.join(dataset_root,dataset,'test'))
csv_root = os.path.join(dataset_root,dataset,'image_info.csv')

for folder in all_folders_dataset:
	for det_model in det_models:
		for cla_model in cla_models:
			save_dir = os.path.join('Result',str(today),dataset,det_model,cla_model)
			os.system('python inference_image_height.py --image_root {}  --out_dir {} --det_model {} --cla_model {} --image_ext {}'.format(folder,save_dir,det_model,cla_model,'jpg'))
