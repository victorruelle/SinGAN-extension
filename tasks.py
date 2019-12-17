from ExperimentManager import getManager
manager = getManager() # will recover the active manager (must be imported from main)

import os

from SinGAN.SinGAN.functions import post_config
from SinGAN.config import get_arguments
from SinGAN.SinGAN.training import train
from SinGAN.SinGAN import functions
from SinGAN.SinGAN.manipulate import SinGAN_generate

@manager.command
def test_standard_train(input_dir,input_image_name):

    # run the standard training procedure
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input\\Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args("--input_dir {} --input_name {} --mode train".format(input_dir,input_image_name).split(" "))
    opt = functions.post_config(opt)

    print(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    print("dir2save :",dir2save)

    if (os.path.exists(dir2save)):
        print('trained model already exist')

    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt) # !! we dont recover the output of the function and modifications do not seem to be in-place
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
