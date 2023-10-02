import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_dir', type = str,
                        help =' the directory of the model,default using the Bird_D model included',
                        default='')
    # parser.add_argument('--model_type', type = str,
    #                     help =' the type of the model,default type used is Bird_D',
    #                     default='Bird_D')
    parser.add_argument('--det_model', type = str,
                        help ='you can select from yolo,faster, retinanetknn and retinanet',
                        default='retinanet')
    parser.add_argument('--det_conf', type = float, default=0.1,
                        help ='Confidence threshold of your detection model')
    parser.add_argument('--cla_model', type = str,
                        help ='you can select from res18 and mixmatch',
                        default='')
    parser.add_argument('--image_root',type = str,
                        help = 'The root dir where image are stores')
    parser.add_argument('--csv_root',type = str, default='',
                        help = 'The root dir where image info are stores')
    parser.add_argument('--image_ext',type = str, default = 'JPG',
                        help = 'the extension of the image(without dot), default is JPG')
    parser.add_argument('--image_altitude',type = int, default = 90,
                        help = 'the altitude of the taken image, default is set to be 90')
    parser.add_argument('--image_location',type = str, default = 'No_Where',
                        help = 'the location of the taken image, default is set to be 90')
    parser.add_argument('--image_date',type = str, default = '2022-10-26',
                        help = 'the date of the taken image, default is set to be 2022-10-26')
    parser.add_argument('--use_altitude',type = bool, default = True,
                        help = 'whether to use altitude to scale the image, default is True')
    parser.add_argument('--out_dir',type = str,
                        help = 'where the output will be generated,default is ./results',
                        default = './results')
    parser.add_argument('--visualize',type = bool,
                        help = 'whether to have visualization stored to result, default is True',
                        default = False)
    parser.add_argument('--evaluate',type = bool,
                        help = 'whether to evaluate the reslt,default is False',
                        default = True)
    args = parser.parse_args()
    
    #if the image_root input is with extension(*.JPG) wrap into list
    #else fetch the list of image
    return args