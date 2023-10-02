import glob
import shutil
import os
import random

root_dir = '/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/example_images/deepforest'
folder_list = glob.glob(root_dir+'/*/test')
save_dir = '/home/yangzhang/super-gradients-master/dataset/deepforest_local'
for folder in folder_list:
	sub_dataset = folder.split('/')[-2]
	os.makedirs(save_dir+'/{}'.format(sub_dataset), exist_ok=True)
	txt_list = glob.glob(folder+'/*.txt')
	image_list = glob.glob(folder+'/*.png')
	random.shuffle(txt_list)
	annotation_counter = 0
	selected_image_list = []
	for txt_dir in txt_list:
		with open(txt_dir) as f:
			lines = f.readlines()
		annotation_counter += len(lines)
		selected_image_list.append(txt_dir.replace('.txt','.png'))
		# if annotation_counter >= 1000:
		# 	break
	if annotation_counter >= 0:
		os.makedirs(save_dir+'/{}/test'.format(sub_dataset), exist_ok=True)
		for selected_image in image_list:
			shutil.copy(selected_image,save_dir+'/{}/test/{}'.format(sub_dataset,selected_image.split('/')[-1]))
			shutil.copy(selected_image.replace('.png','.txt'),save_dir+'/{}/test/{}'.format(sub_dataset,selected_image.split('/')[-1]).replace('.png','.txt'))
		# os.makedirs(save_dir+'/{}/train'.format(sub_dataset), exist_ok=True)
		# for selected_image in selected_image_list:
		# 	shutil.copy(selected_image,save_dir+'/{}/train/{}'.format(sub_dataset,selected_image.split('/')[-1]))
		# 	shutil.copy(selected_image.replace('.png','.txt'),save_dir+'/{}/train/{}'.format(sub_dataset,selected_image.split('/')[-1]).replace('.png','.txt'))

		# os.makedirs(save_dir+'/{}/test'.format(sub_dataset), exist_ok=True)
		# for selected_image in image_list:
		# 	shutil.copy(selected_image,save_dir+'/{}/test/{}'.format(sub_dataset,selected_image.split('/')[-1]))
		# 	shutil.copy(selected_image.replace('.png','.txt'),save_dir+'/{}/test/{}'.format(sub_dataset,selected_image.split('/')[-1]).replace('.png','.txt'))






