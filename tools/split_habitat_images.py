import glob
import shutil
import os
import pandas as pd

def read_csv_info(csv_dir,append_str = ''):
    df = pd.read_csv(csv_dir)
    habitat_dict = {}
    for line in df.values.tolist():
    # habitat_dict[append_str+line[0]+'.jpg'] = line[2]
        habitat_dict[append_str+line[0]+'.jpg'] = line[-1]
    return habitat_dict

def check_and_make_dir(folder_dir):
    folder = os.path.exists(folder_dir)
    if not folder:
        os.makedirs(folder_dir)

root_dir = '/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/example_images/Bird_K'
# save_dir = root_dir+'/habitat'
save_dir = root_dir+'/height'

folder_list = [root_dir]
habitat_list = []
habitat_left_list = []
for folder_dir in folder_list:
	habitat_dict = read_csv_info(folder_dir+'/image_info.csv')
	image_list = sorted(glob.glob(folder_dir+'/*.jpg'))
	for image_dir in image_list:
		image_name = image_dir.split('/')[-1]
		txt_dir = image_dir.replace('.jpg','.txt')
		if image_name in habitat_dict.keys():
			# habitat_name = habitat_dict[image_name].lower()
			habitat_name = str(habitat_dict[image_name])
			if habitat_name not in habitat_list:
				habitat_list.append(habitat_name)
				check_and_make_dir(save_dir+'/{}'.format(habitat_name))
			shutil.copy(image_dir,save_dir+'/{}/{}'.format(habitat_name,image_name))
			shutil.copy(txt_dir,save_dir+'/{}/{}'.format(habitat_name,image_name.replace('.jpg','.txt')))
		else:
			habitat_left_list.append(image_name)
print(habitat_left_list)




