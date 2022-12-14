import os
import glob

datasets = [
['Bird_A','jpg'],
['Bird_B','JPG'],
['Bird_C','JPG'],
['Bird_D','JPG'],
['Bird_E','png'],
['Bird_F','png'],
['Bird_G','JPG'],
['Bird_H','jpg'],
['Bird_I','png'],
['Bird_J','jpg']
]

model_dict = {

	'retinanet' : [
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/retinanet/Bird_A/final_model.pkl','A'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/retinanet/Bird_B/final_model.pkl','B'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/retinanet/Bird_C/final_model.pkl','C'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/retinanet/Bird_D/final_model.pkl','D'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/retinanet/Bird_E/final_model.pkl','E'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/retinanet/general/final_model_alt_60.pkl','general']
	],

	'faster' : [
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Bird_J_15m/model_final.pth','J15'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Bird_J_30m/model_final.pth','J30'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Bird_J_60m/model_final.pth','J60'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Bird_J_90m/model_final.pth','J90'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn_A/model_final.pth','A'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn_B/model_final.pth','B'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn-Bird/model_final.pth','Bird'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn_C/model_final.pth','C'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn_D/model_final.pth','D'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn-Decoy/model_final.pth','Decoy'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn_E/model_final.pth','E'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn_F/model_final.pth','F'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/faster/Fasterrcnn-high_density/model_final.pth','density'],
	],

	'yolo' : [
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/A/weights/best.pt','A'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/B/weights/best.pt','B'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/C/weights/best.pt','C'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/D/weights/best.pt','D'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/E/weights/best.pt','E'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/F/weights/best.pt','F'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/G/weights/best.pt','G'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/general/weights/best.pt','general'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/H/weights/best.pt','H'],
		['/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/yolo/J/weights/best.pt','J'],

	]

}

det_models = [
'yolo',
'retinanet',
'faster',
]


for det_model in det_models:
		for model_dir in model_dict[det_model]:
			for dataset in datasets:
				os.system('python inference_image_list.py --image_root ./example_images/{}/test  --out_dir ./Result/{}/{}/{}/ --evaluate True --det_model {} --image_ext {} --model_dir {}'.format(dataset[0],det_model,dataset[0],model_dir[1],det_model,dataset[1],model_dir[0]))