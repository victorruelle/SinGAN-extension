# SinGAN-extension
Experimenting with SinGANs (https://github.com/tamarott/SinGAN) and possible extensions in the context of my final project for the Visual Recognition course of the MVA master class.

## Extension Idea

We will try to upsample patches of a low resolution image that correspond to the patch of a given high resolution image. For instance : we have a high-res image of charlotte's face and many low res images where she appears. We will try to increase the resolution of those low res faces and compare the result with the one obtained using a regular SR training.

Can we extend this by using multiple training images? All that is needed is to provide more true labels to the disciminators.

### Steps

1. Select 10 images of Charlotte's head and downsize them to a max(height,width) of 200. Use images that have different lighting, poses, glasses or not etc.
2. Split them in 5 training images and 5 testing images.
3. For each training image, train a fixed pyramid of generators and discriminators and apply the SR testing to each of the 5 testing images.
4. For each testing image, traing with the same parameters and apply classic SR
5. Compare, for each training image, the upsampled image obtained from training with each of the training images and the actual testing image.


## How to tune experiments

### Changing the number of scales

The number of scales $N$ is defined as 

$$
N = ceil\Bigg( log\bigg( \frac{\textrm{coarsest size}}{min(\textrm{original width, height})} \bigg) \Bigg)
$$

In the config file, $\textit{coarsest size}$ is defined as $\textit{min\_size}$ and the with and height are defined by the chosen input image. Hence, the only way to modify $N$ is to modify $\textit{min\_size}$. If $\textit{min\_size} = min(\textit{original width, height})$, there will only be one scale. Reducing $\textit{min\_size}$ will increase the number of scales.


## Code changes to make

### Printing

Change the printing of scale progress to a progress bar.

### Saving 

Put all saves within the current run folders.
Delete dir2save function.

### Model retraining

Change the retraining mechanic : specify any given path. Right now, it looks at the one generated with dir2save.

# SinGAN
[Project](http://webee.technion.ac.il/people/tomermic/SinGAN/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) 
### Official pytorch implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image"
####  ICCV 2019


## Random samples from a *single* image
With SinGAN, you can train a generative model from a single natural image, and then generate random samples form the given image, for example:

![](imgs/teaser.PNG)


## SinGAN's applications
SinGAN can be also use to a line of image manipulation task, for example:
 ![](imgs/manipulation.PNG)
This is done by injecting an image to the already trained model. See section 4 in our [paper](https://arxiv.org/pdf/1905.01164.pdf) for more details.


### Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{rottshaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Rott Shaham, Tamar and Dekel, Tali and Michaeli, Tomer},
  booktitle={Computer Vision (ICCV), IEEE International Conference on},
  year={2019}
}
```

## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6  

###  Train
To train SinGAN model on your own image, put the desire training image under Input\\Images, and run

```
python main_train.py --input_name <input_file_name>
```

This will also use the resulting trained model to generate random samples starting from the coarsest scale (n=0).

To run this code on a cpu machine, specify `--not_cuda` when calling `main_train.py`

###  Random samples
To generate random samples from any starting generation scale, please first train SinGAN model for the desire image (as described above), then run 

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples --gen_start_scale <generation start scale number>
```

pay attention: for using the full model, specify the generation start scale to be 0, to start the generation from the second scale, specify it to be 1, and so on. 

###  Random samples of arbitrery sizes
To generate random samples of arbitrery sizes, please first train SinGAN model for the desire image (as described above), then run 

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```

###  Animation from a single image

To generate short animation from a single image, run

```
python animation.py --input_name <input_file_name> 
```

This will automatically start a new training phase with noise padding mode.

###  Harmonization

To harmonize a pasted object into an image (See example in Fig. 13 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model for the desire background image (as described above), then save the naively pasted reference image and it's binary mask under "Input\\Harmonization" (see saved images for an example). Run the command

```
python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>

```

Please note that different injection scale will produce different harmonization effects. The coarsest injection scale equals 1. 

###  Editing

To edit an image, (See example in Fig. 12 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desire non-edited image (as described above), then save the naive edit as a reference image under "Input\\Editing" with a corresponding binary map (see saved images for an example). Run the command

```
python editing.py --input_name <training_image_file_name> --ref_name <edited_image_file_name> --editing_start_scale <scale to inject>

```
both the masked and unmasked output will be saved.
Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 

###  Paint to Image

To transfer a paint into a realistic image (See example in Fig. 11 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desire image (as described above), then save your paint under "Input\\Paint", and run the command

```
python paint2image.py --input_name <training_image_file_name> --ref_name <paint_image_file_name> --paint_start_scale <scale to inject>

```
Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 

Advanced option: Specify quantization_flag to be True, to re-train *only* the injection level of the model, to get a on a color-quantized version of upsamled generated images from previous scale. For some images, this might lead to more realistic results.

### Super Resolution
To super resolve an image, please run:
```
python SR.py --input_name <LR_image_file_name>
```
This will automatically train a SinGAN model correspond to 4x upsampling factor (if not exist already).
For different SR factors, please specify it using the parametr `--sr_factor` when calling the function.
SinGAN's results on BSD100 dataset can be download from the 'Downloads' folder.

## Additional Data and Functions

### Single Image Fréchet Inception Distance (SIFID score)
To calculate the SIFID between real images and their corresponding fake samples, please run:
```
python SIFID/sifid_score.py --path2real <real images path> --path2fake <fake images path> --images_suffix <e.g. jpg, png>
```  
Make sure that each of the fake images file name is identical to its cooresponding real image file name. 

### Super Resolution Results
SinGAN's SR results on BSD100 dataset can be download from the 'Downloads' folder.

### User Study
The data used for the user study can be found in the 'Downloads' folder. 

'real' folder: 50 real images, randomly picked from the [places databas](http://places.csail.mit.edu/)

'fake_high_variance' folder: random samples starting from n=N for each of the real images 

'fake_mid_variance' folder: random samples starting from n=N-1 for each of the real images 

For additional details please see section 3.1 in our [paper](https://arxiv.org/pdf/1905.01164.pdf)


