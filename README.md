# StyleGAN - TensorFlow 2.x

![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![TensorFlow 2.5](https://img.shields.io/badge/tensorflow-2.5-green.svg?style=plastic)
![CUDA Toolkit 11.2.2](https://img.shields.io/badge/cudatoolkit-11.2.2-green.svg?style=plastic)
![cuDNN 8.1.0.77](https://img.shields.io/badge/cudnn-8.1.0.77-green.svg?style=plastic)

Tensorflow 2 implementation of the paper 
**"A Style-Based Generator Architecture for Generative Adversarial Networks"** (https://arxiv.org/abs/1812.04948) <br>
The code is based on the official implementation: https://github.com/NVlabs/stylegan.

**Note:** the code is under active development, so some things have not yet been tested, though when training in 
*fp32* or *mixed precision* no problems were observed. 
Settings for stabilization for *mixed precision* tricks might need to be improved to stabilize training for various configs and datasets.

This implementation allows finer control of a training process and model complexity: 
one can use different parameters which define number of filters of each network (consider function `n_filters()` in `networks.py`), 
size of latent vector, change activation functions, add/remove biases, set different numbers of images for each stage, 
use different optimizers settings,
etc.


## Training

To train a model one needs to:

1. Define a training config (see `configs` section for details).<br>
   *Note:* paths to images should be saved in a separate *.txt* file, 
   which is to be provided under key `images_paths_filename` in config.
2. Optionally configure gpu memory options (consider **GPU memory usage** section).
3. Optionally set training mode (consider **Training speed** section).
4. Start training with command:

> python train.py --config path_to_config (e.g. --config default_config.json)


## Inference

To run inference consider file `inference.py`. <br>
Example call:

> python .\inference.py --config_path .\configs\lsun_living_room.json  --weights_path .\weights\lsun_living_room\256x256\stabilization\step3000000\G_model_smoothed.h5 --image_fname images --grid_cols 4 --grid_rows 3


## Configs
Examples of configs are available in `configs` folder.

Paths to images should be saved in a separate *.txt* file, which is to be provided under key `images_paths_filename` in config.

Configs which were used in the official implementation to train on FFHQ dataset:
* `paper_config_ffhq_res1024_full.json` — all values, almost all keys have default values;
* `paper_config_ffhq_res1024_short.json` — similar to the previous config except that omitted keys automatically use default values;
* `paper_config_ffhq_res1024_short_fast.json` — similar to the previous config but with all available speed-ups (mixed precision, XLA, fused bias and activation layer). 

*Note*: options related to summaries are not aligned with the values in the official implementation. Set them according to your needs.

For debugging, it's convenient to use `debug_config.json`.

All possible options and their default values can be found in file `config.py`.


## Training speed

To get maximum performance one should prefer training each model in a separate process (`single_process_training` in `train.py`), 
as in this case all GPU resources are released after process is finished.

Another way to increase performance is to use mixed precision training, which not just speeds operations up 
(especially on Nvidia cards with compute capability 7.0 or higher, e.g., Turing or Ampere GPUs), but also allows to increase batch size.

Some notes about the tricks to enable stable mixed precision training (inspired by one of next papers from the same authors):
* Enable mixed precision only for the N (set to 4 in the official implementation) highest resolutions;
* Clamp the output of every convolutional layer to 2^8, i.e., an order of magnitude wider range than is needed in practise;
* No need to pre-normalize style vector (how to do it and why?) or inputs x (instance norm is used by default).

Enabling XLA (Accelerated Linear Algebra, jit compilation) should improve training speed and memory usage.

Note: when training with mixed precision on LSUN Living Room dataset loss scale became 1 for both (G and D) optimizers
after about 8.5M images (around 3M for the last train stage). 
It might indicate that mixed precision should be made more stable. 
As a result of this behaviour some of valid images became all black (after converting), 
yet after a number of iterations other images might become black and others become normal again.


## GPU memory usage

To control GPU memory usage one can refer to a function `prepare_gpu()` in `tf_utils.py`. 
<br>
Depending on your operating system and use case you might want to change memory managing. 
By default, on Linux `memory_growth` option is used, while on Windows memory is limited with some reasonable number to allow use of PC (such as opening browsers with small number of tabs).
<br>
*Note:* the code was used with GPUs with 8 Gb of memory, so if your card has more/less memory it is strongly recommended to consider modifying `prepare_gpu()` function. 

## System requirements

* The code was tested on Windows (and will be on Linux later, I hope). 
* The following software should be installed on your machine:
```
- NVIDIA driver 461.92 or newer
- TensorFlow-gpu 2.5  or newer
- CUDA Toolkit 11.2.2 or newer
- cuDNN 8.1.0.77 or newer
- other dependencies ..
```
* For some reason on Windows 10 with mentioned versions of NVIDIA libraries CUPTI must be manually configured. To do this:
  - Go to folder `c:\Program Files\NVIDIA Corporation\` and search for files `cupti*.dll`. 
  - Copy all of them to your CUPTI folder. 
    Let `cuda_base = c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`. 
    Then it would be `{cuda_base}\extras\CUPTI\lib64\`. 
    File `cupti64_2020.3.1.dll` already was there.
  - Add CUPTI path to `Path` variable: `{cuda_base}\extras\CUPTI\lib64`
  - If you still see error messages try copying all found and existing CUPTI`.dll` files to `{cuda_base}\bin`.
  - Thanks to https://stackoverflow.com/questions/56860180/tensorflow-cuda-cupti-error-cupti-could-not-be-loaded-or-symbol-could-not-be.
  Answer by `Malcolm Swaine`.
    
*Note:* software versions should be consistent, i.e., if you use TensorFlow from *pip* 
you should check CUDA and cuDNN versions on the official TensorFlow site.


## Metrics

Supported metrics are:
* Perceptual Path Length (PPL)
  - Similarly to the official implementation it supports a number of options:
      * Space: *w*, *z*
      * Sampling method: *full*, *end*
      * Epsilon: default is *1e-4*
      * Optional face crop for dataset with human faces (default is *False*) 
      * Number of samples (the official implementation uses 100k, which takes lots of time to run, 
        so consider using a lower value, e.g., 20k or 50k)
  - To calculate the metric when the resolution of generated images is less than 256 (VGG was trained for 224) 
    images are naively upsampled to resolution 256, if their resolution is lower than that.
      * Probably images should not be upsampled. It's not obvious how the case is handled in the official implementation.
  - Supports mixed precision and XLA.
  - Slightly changed *TensorFlow 2* port of lpips model by `moono` is used: https://github.com/moono/lpips-tf2.x.
* Frechet Inception Distance (FID)
  - Supports mixed precision and XLA.
  - To calculate the metric when the resolution of generated images is less than 256 (Inception was trained for 299) 
    images are naively upsampled to resolution 256, if their resolution is lower than that.
      * Probably images should not be upsampled. It's not obvious how the case is handled in the official implementation.
  

## Further improvements

- Add CUDA implementations for fused layers
- Tune settings for *mixed precision* training stabilization tricks
- Add multi GPU support
- Fix training in a single process
- Fix problems with name scopes inside `tf.function()`. 
  The current solution relies on the answer by `demmerichs`: https://github.com/tensorflow/tensorflow/issues/36464
  