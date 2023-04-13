import os
import glob

model_list = ['retinanet','yolo','faster']
dataset = 'Bird_I_Test_habitat'
all_folders_dataset = glob.glob(os.path.join('example_images',dataset,'test','*'))
csv_root = os.path.join('example_images',dataset,'image_info.csv')
cla_model = 'res18'

for det_model in model_list:
	for folder in all_folders_dataset:
		category = os.path.split(folder)[-1]
		save_dir = os.path.join('Result','0413',dataset,det_model,cla_model,category)
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {} --cla_model {} --csv_root {}'.format(folder,save_dir,det_model,'jpg',cla_model,csv_root))
 