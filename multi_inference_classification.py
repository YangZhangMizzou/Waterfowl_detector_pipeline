import os
import glob

dataset = 'Bird_L_SpeciesClassification_Images'
det_models = ['yolonas']
cla_models = ['mixmatch']
all_folders_dataset = glob.glob(os.path.join('example_images',dataset))
csv_root = ''
# csv_root = os.path.join('example_images',dataset,'image_info.csv')

for folder in all_folders_dataset:
	for det_model in det_models:
		for cla_model in cla_models:
			category = os.path.split(folder)[-1]
			save_dir = os.path.join('Result','0611',dataset,det_model,cla_model,category)
			os.system('python inference_image_height.py --image_root {}  --out_dir {} --det_model {} --cla_model {} --image_ext {}'.format(folder,save_dir,det_model,cla_model,'JPG'))
