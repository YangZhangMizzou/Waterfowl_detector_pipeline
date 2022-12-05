# Retinanet for Waterfowl detection inference example

This code is an demo only on how to apply retinanet onto the Waterfowl dataset. In this demo contains pretrained model named as Bird_A, Bird_B, Bird_C, Bird_D, Bird_E  avaialble under folder **checkpoint** please download and unzip the checkpoint folder content using link here https://drive.google.com/file/d/1ZXva0sP5h7JiaUmU4mWpb5yrjwr7B1ai/view?usp=sharing and place the downloaded file inside the checkpoint folder.

## Branch usage:
This is the EasyUse branch which provides most easy approach of using the system as research tools, if seeking for experiment/software testing, please switch to branch **master**

## System requirements
Support both Windows system and Linux system, can be used with/without cuda environement.
Tested in Ubuntu 18 with python=3.8,3.10, Windows with python=3.10

## Example images

In this demo, some images are included for testing, under folder **example_images** contains different images for each corresponding pretained model by their folder name(Bird_A,Bird_B etc), along with the GT file attatched for later evaluation.

## Installation

### Clone the repository
You can either use the cmd window to clone the repo using the following command or you can refer to the link here:https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
```
git clone https://github.com/RobertTzc/Retinanet_inference_example.git
```
We provided two different ways of installing the packages, through virtual env and direct installation
### Virtual env

Virtual env is recommended to be used in order to install the package, here Anaconda is recommended to be used, link to the Anaconda https://www.anaconda.com/, once you have installed the Anaconda , refer here to create you virtual env https://conda.io/projects/conda/en/latest/user-guide/getting-started.html. It is recommend to create the env along with python 3.8, demo cmd is here:
```
conda create -n torch_py3 python==3.8
```
once env is created use, eg:
```
conda activate torch_py3
```
once you have activated the env, make sure you are under the env you created and run the following commands:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install opencv-contrib-python
pip install Pillow==6.1
pip install pandas
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
```
### Direct installation
Packages can also be installed with direct pip :

```
For windows:
  pip3 install torch torchvision torchaudio
For Linux:
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
The rest are identical on both systems:
pip install opencv-contrib-python
pip install Pillow==6.1
pip install pandas
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


## Run the Scripts:
Once you have the input file ready and in correct virtual env, you can use the file **inference_image_list.py** to start inference the images:
quick example(full command):
```
python inference_image_list.py \
--model_dir ./checkpoint/Bird_A/final_model.pkl \
--model_type Bird_A \
--image_root ./example_images/Bird_A \
--image_ext jpg \
--image_altitude 90 \
--use_altitude true \
--image_date 2022-11-01 \
--image_location University_of_Missouri \
--out_dir ./Result/Bird_A \
--evaluate true \
--visualize true 
```
quick example(easy command):
```
python inference_image_list.py \
--model_dir ./checkpoint/Bird_A/final_model.pkl \
--model_type Bird_A \
--image_root ./example_images/Bird_A \
--out_dir ./Result/Bird_A
```

The description of each command are as follows:
```
--model_dir: directory of the model, in this demo is where the checkpoint model locates
--model_type: we include some models that are specifically trained on different types of scenerio, in this demo for most cases, it will --simply match the model_dir type.

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


