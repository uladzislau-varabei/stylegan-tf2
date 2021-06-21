# StyleGAN - TensorFlow 2.x

![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![TensorFlow 2.5](https://img.shields.io/badge/tensorflow-2.5-green.svg?style=plastic)
![CUDA Toolkit 11.2.2](https://img.shields.io/badge/cudatoolkit-11.2.2-green.svg?style=plastic)
![cuDNN 8.1.0.77](https://img.shields.io/badge/cudnn-8.1.0.77-green.svg?style=plastic)

Tensorflow 2 implementation of the paper 
**"A Style-Based Generator Architecture for Generative Adversarial Networks"** (https://arxiv.org/abs/1812.04948) <br>
The code is based on the official implementation: https://github.com/NVlabs/stylegan.

**Note:** currently the code is in beta version, and some things have not yet been tested, so it should be used very carefully.
E.g., training in mixed precision doesn't work (see Tensorboard for details).


This implementation allows finer control of a training process and model complexity: 
one can use different parameters which define number of filters of each network (consider function `n_filters()` in `networks.py`), 
size of latent vector, change activation functions, add/remove biases, set different numbers of images for each stage, 
use different optimizers settings,
etc.


## Model training

To train a model one needs to:

1. Define a training config (example in `default_config.json`, all available options and their default values can be found in `utils.py`). *Note:* some values in config are different from original implementation due to memory constraints.
2. Optionally configure gpu memory options (consider **GPU memory usage** section).
3. Optionally set training mode (consider **Training speed** section).
4. Start training with command: <br>

> python train.py --config=path_to_config (e.g. --config=default_config.json)


## Training speed

To get maximum performance one should prefer training each model in a separate process (`single_process_training` in `train.py`), 
as in this case all GPU resources are released after process is finished.  <br>
Another way to increase performance is to use mixed precision training, which not just speeds operations up (especially on Nvidia cards with compute capability 7.0 or higher, e.g. Turing GPUs), but also allows to increase batch size. <br>


## GPU memory usage

To control GPU memory usage one can refer to a function `prepare_gpu()` in `utils.py`. 
<br>
Depending on your operating system and use case you might want to change memory managing. 
By default, on Linux `memory_growth` option is used, while on Windows memory is limited with some reasonable number to allow use of PC (such as opening browsers with small number of tabs).
<br>
*Note:* the code was used with GPUs with 8 Gb of memory, so if your card has more/less memory it is strongly recommended to consider modifying `prepare_gpu()` function. 

## System requirements

* The code was tested on Windows and Linux (will be later, I hope). 
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
    Let `cuda_base = c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`. 
    Then it would be `{cuda_base}\extras\CUPTI\lib64\`. 
    File `cupti64_2020.3.1.dll` already was there.
  - Add CUPTI path to `Path` variable: `{cuda_base}\extras\CUPTI\lib64`
  - If you still see error messages try copying all found and existing CUPTI`.dll` files to `{cuda_base}\bin`.
  - Thanks to https://stackoverflow.com/questions/56860180/tensorflow-cuda-cupti-error-cupti-could-not-be-loaded-or-symbol-could-not-be.
  Answer by `Malcolm Swaine`.
    
*Note:* software versions should be consistent, i.e., if you use TensorFlow from *pip* 
you should check CUDA and cuDNN versions on the official TensorFlow site.


## Further improvements

- Fix problems with mixed precision training
- Implement evaluation of metrics (especially Perceptual Path Length) to track numbers of quality progress
- Implement Style mixing
- Implement Truncation trick
- Add XLA support  
- Add multi GPU support
