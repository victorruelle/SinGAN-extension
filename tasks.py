from ExperimentManager import getManager
manager = getManager() # will recover the active manager (must be imported from main)

import os

from SinGAN.SinGAN.functions import post_config
from SinGAN.config import get_arguments
from SinGAN.SinGAN.training import train
from SinGAN.SinGAN import functions
from SinGAN.SinGAN.manipulate import SinGAN_generate
from utils import show_ops

@manager.command
def test_standard_train(input_dir,input_image_name):

    print("Starting test_standard_train")

    # clean given dirs
    input_dir = os.path.relpath(input_dir)

    # run the standard training procedure
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input\\Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args("--input_dir {} --input_name {} --mode train".format(input_dir,input_image_name).split(" "))
    opt = functions.post_config(opt) # will also inect parameters from config

    show_ops(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt) # !! we dont recover the output of the function and modifications do not seem to be in-place
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)


@manager.command
def charlotte_experiment(sr_factor):
    ''' We don't expect an input_dir nor an input_image.
    '''

    print("Starting train_charlotte_experiment")

    # clean given dirs
    input_dir = os.path.relpath(input_dir)

    parser = get_arguments()
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    opt = parser.parse_args("--sr_factor {} ".format(sr_factor).split(" "))
    opt = functions.post_config(opt) # will also inect parameters from config

    mode = "SR"
    opt.mode = "train"

    print("\n\nStarting Training")
    opt.input_dir = os.path.join(manager.project_dir,"input_images","charlotte","training")

    for i in range(5):

        ## TODO : make everything in this loop a command so that each iteration has a seperate saving folder.

        # Training with image i
        print("\n\nStarting new image training",i)
        opt.mode = "train"
        opt.input_image = "{}.png".format(str(i))
        show_ops(opt)

        Gs = []
        Zs = []
        reals = []
        NoiseAmp = []
        
        real = functions.read_image(opt)
        opt.min_size = 18
        functions.adjust_scales2image(real, opt) # !! we dont recover the output of the function and modifications do not seem to be in-place
        train(opt, Gs, Zs, reals, NoiseAmp)

        # Testing with all testing images
        opt.mode = mode
        for j in range(5):
            print("\n\nStarting testing with training image",i,"and testing image",j)

            ## TODO : rewrite this copied testing procedure to test on another image (change real variable) and save to an understandble folder
            Zs_sr = []
            reals_sr = []
            NoiseAmp_sr = []
            Gs_sr = []
            real = reals[-1]  # read_image(opt)
            real_ = real
            opt.scale_factor = 1 / in_scale
            opt.scale_factor_init = 1 / in_scale
            for j in range(1, iter_num + 1, 1):
                real_ = imresize(real_, pow(1 / opt.scale_factor, 1), opt)
                reals_sr.append(real_)
                Gs_sr.append(Gs[-1])
                NoiseAmp_sr.append(NoiseAmp[-1])
                z_opt = torch.full(real_.shape, 0, device=opt.device)
                m = nn.ZeroPad2d(5)
                z_opt = m(z_opt)
                Zs_sr.append(z_opt)
            out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
            out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2]), 0:int(opt.sr_factor * reals[-1].shape[3])]
            dir2save = functions.generate_dir2save(opt)
            plt.imsave(os.path.join(dir2save,'{}_HR.png'.format(opt.input_name[:-4])), functions.convert_image_np(out.detach()), vmin=0, vmax=1)


    ## TODO : Add classic SR for each testing image!

