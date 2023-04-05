import os
import glob

model_list = ['retinanet','yolo','faster']
all_folders_dataset = glob.glob('./example_images/drone_collection_dataset/test/*')

for folder in all_folders_dataset:
	for model in model_list:
		image_type = folder.split('/')[-2]
		category = folder.split('/')[-1]
		save_dir = './Result/{}/{}/{}'.format(model,image_type,category)
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {}'.format(folder,save_dir,model,'JPG'))
 