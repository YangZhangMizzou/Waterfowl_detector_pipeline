import os
import glob

datasets = [
['Bird-G','JPG'],
['Bird-I','JPG'],
['Bird-J','JPG'],
]
models = ['yolo','retinanet',]

for model in models:
	for dataset in datasets:
		save_dir = './Result_exp/{}/{}'.format(model,dataset[0])
		image_dir = './example_images/drone_collection_test/{}'.format(dataset[0])
		os.system('python inference_image_height.py --image_root {}  --out_dir {} --evaluate True --det_model {} --image_ext {}'.format(image_dir,save_dir,model,dataset[1]))