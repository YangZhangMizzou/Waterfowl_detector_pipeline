# Waterfowl detector pipeline instruction

This code is an demo only on how to apply retinanet,yolov5 and FasterRCNN onto the Waterfowl dataset. In this demo, pretrained general models of three detectors are provided and can be downloaded here https://drive.google.com/file/d/1Rgbn9JvzJb1vVu_l-l2D0g5R0-Vj3EXt/view?usp=sharing . Download and extract its content to folder 'checkpoints'.

## System requirements
Support both Windows system and Linux system, can be used with/without cuda environement.
Tested in Ubuntu 18 with python=3.8,3.10, Windows with python=3.10

## Example images

In this demo, some images are included for testing, under folder **example_images** contains different images for each corresponding pretained model by their folder name(Bird_A,Bird_B etc), along with the GT file attatched for later evaluation.

## Installation

### Clone the repository
You can either use the cmd window to clone the repo using the following command
```
git clone https://github.com/YangZhangMizzou/Waterfowl_detector_pipeline.git
```

### Create virtual environment
Virtual env is recommended to be used in order to install the package, here Anaconda is recommended to be used, link to the Anaconda https://www.anaconda.com/, once you have installed the Anaconda , refer here to create you virtual env https://conda.io/projects/conda/en/latest/user-guide/getting-started.html. It is recommend to create the env along with python 3.8, demo cmd is here:

```
conda create -n torch_py3 python==3.8
conda activate torch_py3
cd Waterfowl_detector_pipeline
```

### Install pytorch

We recommend to install pytorch with cuda to accelerate running speed.
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
### Install dependency for FasterRCNN

Install Detectron2 following the instructions on the website. https://detectron2.readthedocs.io/en/latest/index.html
and then run the following command

```
pip install pandas
pip install numpy
pip install opencv-python
pip install tqdm
pip install efficientnet_pytorch
pip install resnet_pytorch
```

### Install dependency for YOLOV5

Just run this command and all set for YOLOV5
```
pip install -r requirements.txt
```

### Install dependency for Retinanet

```
pip install opencv-contrib-python
pip install Pillow==6.1
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
```


## input format
This script requires a speical format of input describes below
```
Image_folder (eg Bird_A)
├── image_name1.jpg
├── image_name1.txt # if want to evaluate
├── image_name2.jpg
├── image_name1.txt # if want to evaluate
├── image_name3.jpg
├── image_name2.txt # if want to evaluate
└── ...
```
## Run all experiments:
You can easily run all experiments on drone_collection dataset(all height, habitat, dataset) by only one command. Please follow the precedures carefully.

1. extract all images from drone_collection.zip to example_images folder
```

```

## Run the Scripts:
Once you have the input file ready and in correct virtual env, you can use the file **inference_image_list.py** to start inference the images:
quick example(full command):
```
python inference_image_list.py \
--det_model retinanet \
--image_root ./example_images/Bird_A \
--image_ext jpg \
--image_altitude 90 \
--use_altitude False \
--image_date 2022-11-01 \
--image_location University_of_Missouri \
--out_dir ./result/retinanet/Bird_A \
--evaluate False \
--visualize true 
```
quick example(easy command):
```
python inference_image_list.py \
--det_model retinanet \
--image_root ./example_images/Bird_A \
--out_dir ./result/retinanet/Bird_A
```

run yolo only:
```
python inference_image_yolo.py\
 --det_model yolo\
 --image_root ./example_images/drone_collection\
 --out_dir ./Result/yolo/drone_collection\
```

The description of each command are as follows:
```
--det_model: name of the detection model. you can select from yolo,faster and retinanet
--image_root: specify where the iunput images stores
--use_altitude: true/false variable, this specifies whether use the input altitude to scale the image during the inference. default is true.
--image_ext: image extension of the target images, default is 'JPG'
--image_altitude: the altitude of the images being inferenced, default is set to be 90 (meters)
--image_date: specified the date the image was taken, this will be stored as description data
--image_location: where the image is taken, this will be stored as description data

--out_dir: where the output file will be generated, by default it will create 'Result' folder under current directory.
--visual: true/false value specify whether we want to have visualization on output, default is true
--evaluate: whether we want to evaluate the result, this can only be done when the input file comes with groundTruth file, default is false
-h: display help info.
```
## Output format
When you specify the output dir, you shall expecting the output in the following:
```
Result folder 
├── detection-results
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── visualize-results
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
├── configs.json
├── detection_summary.csv
├── f1_score.jpg    #if apply evaluation
└── mAP.jpg         #if apply evaluation

detection_summary contains three types of data:
Description data includes input info of the image info such as image_name, date,altitude
Meta data includes meta data read from the image Meta data(if appliable)
Sample results are shown below:
Detection data: includes num of birds detected and time spent inferencing that image(include visualization)

each txt file under detection_results file contains detected bounding box in the following format:
  category,confidence score, x1,y2,x2,y2
Sorted in condifence descending order.
```
![Screenshot from 2022-11-01 15-59-13](https://user-images.githubusercontent.com/71574752/199340134-13dc5f02-4980-4bac-9a6a-4a5d6a04050e.png)


