import os
import json
import logging
from glob import glob
import shutil
import platform

import numpy as np
import tensorflow as tf
import h5py


# Recommended for Tensorflow
NCHW_FORMAT = 'NCHW'
# Recommended by Nvidia
NHWC_FORMAT = 'NHWC'

DEFAULT_DATA_FORMAT = NCHW_FORMAT

TRANSITION_MODE = 'transition'
STABILIZATION_MODE = 'stabilization'
SMOOTH_POSTFIX = '_smoothed'
OPTIMIZER_POSTFIX = '_optimizer'
RGB_NAME = 'RGB'
LOD_NAME = 'lod'
WSUM_NAME = 'WSum'
GENERATOR_NAME = 'G_model'
DISCRIMINATOR_NAME = 'D_model'
WEIGHTS_DIR = 'weights'
LOGS_DIR = 'logs'
TF_LOGS_DIR = 'tf_logs'
IMAGES_DIR = 'images'
DATASET_CACHE_DIR = 'tf_ds_cache'
CACHE_DIR = 'cache'
OS_LINUX = 'Linux'
OS_WIN = 'Windows'
HE_INIT = 'He'
LECUN_INIT = 'LeCun'

TRAIN_MODE = 'training'
INFERENCE_MODE = 'inference'
DEFAULT_MODE = INFERENCE_MODE


# ---------- Config options ----------

### ---------- General options ---------- ###

# Model name (used in folders for logs and progress images)
MODEL_NAME = 'model_name'
# Additional prefix for storage
# Note: it was meant to be used if model was to be saved not in script directory,
# only consider using it if model is trained on server, otherwise just skip it
STORAGE_PATH = 'storage_path'
# Path to a file with images paths
IMAGES_PATHS_FILENAME = 'images_paths_filename'
# Target resolution (should be a power of 2, e.g. 128, 256, etc.)
TARGET_RESOLUTION = 'target_resolution'
# Start resolution (should be a power of 2, e.g. 128, 256, etc.)
# Note: to disable progressive growing set value equal to TARGET_RESOLUTION
START_RESOLUTION = 'start_resolution'
# Size of latent vector
LATENT_SIZE = 'latent_size'
# Size of disentangled latent vector
DLATENT_SIZE = 'dlatent_size'
# Use bias layers?
USE_BIAS = 'use_bias'
# Use equalized learning rates?
USE_WSCALE = 'use_wscale'
# Truncate weights?
TRUNCATE_WEIGHTS = 'truncate_weights'
# Low-pass filter to apply when resampling activations
BLUR_FILTER = 'blur_filter'
# Weights data format, NHWC and NCHW are supported
DATA_FORMAT = 'data_format'
# Use GPU for smoothed generator?
# Note: setting this option to False saves GPU memory (but how effective is this?)
USE_GPU_FOR_GS = 'use_GPU_for_Gs'
# Use mixed precision training?
USE_MIXED_PRECISION = 'use_mixed_precision'
# Use FP16 for the N highest resolutions, regardless of dtype.
NUM_FP16_RESOLUTIONS = 'num_fp16_resolutions'
# Clamp activations to avoid float16 overflow? Note: number or None to disable
CONV_CLAMP = 'conv_clamp'
# Fuse bias and activation?
FUSED_BIAS_ACT = 'fused_bias_act'
# Data type
# Note: it is highly not recommended to change it to float16, just use float32 and mixed precision training if needed
DTYPE = 'dtype'
# Use XLA compiler?
USE_XLA = 'use_XLA'


### ---------- Training ---------- ###

# Total number of images (thousands) for training
# Note: last stabilization stage might be quite long comparing to all previous stages
TOTAL_KIMAGES = 'total_kimages'
# Base number of images (thousands) for transition stage for each resolution (in a form of dict)
TRANSITION_KIMAGES = 'transition_kimages'
# Base number of images (thousands) for stabilization stage for each resolution (in a form of dict)
STABILIZATION_KIMAGES = 'stabilization_kimages'
# Resolution specific number of transition images
# Note: keys should be given as powers of 2
TRANSITION_KIMAGES_DICT = 'transition_kimages_dict'
# Resolution specific number of stabilization images
# Note: keys should be given as powers of 2
STABILIZATION_KIMAGES_DICT = 'stabilization_kimages_dict'
# Batch size for each resolution (in a form of dict)
# Note: keys should be powers of 2
BATCH_SIZES = 'batch_sizes'
# Number of batches to run before adjusting training parameters
BATCH_REPEATS = 'batch_repeats'
# Loss function for the generator
G_LOSS_FN = 'G_loss_fn'
# Loss function for the discriminator
D_LOSS_FN = 'D_loss_fn'
# Loss function params for the generator
G_LOSS_FN_PARAMS = 'G_loss_fn_params'
# Loss function params for the discriminator
D_LOSS_FN_PARAMS = 'D_loss_fn_params'
# Base learning rate for the generator
G_LEARNING_RATE = 'G_learning_rate'
# Base learning rate for the discriminator
D_LEARNING_RATE = 'D_learning_rate'
# Resolution specific learning rate for generator
# Note: keys should be given as powers of 2
G_LEARNING_RATE_DICT = 'G_learning_rate_dict'
# Resolution specific learning rate for discriminator
# Note: keys should be given as powers of 2
D_LEARNING_RATE_DICT = 'D_learning_rate_dict'
# Adam beta 1 for generator and discriminator
ADAM_BETA1 = 'adam_beta1'
# Adam beta 2 for generator and discriminator
ADAM_BETA2 = 'adam_beta2'
# Reset optimizers states when new layers are introduced?
RESET_OPT_STATE_FOR_NEW_LOD = 'reset_opt_state_for_new_lod'
# Max models to keep
MAX_MODELS_TO_KEEP = 'max_models_to_keep'
# How often to write scalar summaries (measured in thousands)
SUMMARY_SCALARS_EVERY_KIMAGES = 'summary_scalars_every_kimages'
# How often to write histogram summaries (measured in thousands)
SUMMARY_HISTS_EVERY_KIMAGES = 'summary_hists_every_kimages'
# How often to save models weights (measured in thousands)
SAVE_MODEL_EVERY_KIMAGES = 'save_model_every_kimages'
# How often to save progress images (measured in thousands)
SAVE_IMAGES_EVERY_KIMAGES = 'save_images_every_kimages'
# How often to run metrics (measured in thousands)
RUN_METRICS_EVERY_KIMAGES = 'run_metrics_every_kimages'
# Which metrics to compute? dict of form: {metric: [options]}
METRICS_DICT = 'metrics'


### ---------- Generator Mapping network ---------- ###

# Apply pixel normalization to latent vector?
NORMALIZE_LATENTS = 'normalize_latents'
# Number of layers in Mapping network
MAPPING_LAYERS = 'mapping_layers'
# Number of units in dense layers in Mapping network
MAPPING_UNITS = 'mapping_units'
# Learning rate multiplier for Mapping network
MAPPING_LRMUL = 'mapping_lrmul'
# Activation function for Mapping network
MAPPING_ACTIVATION = 'mapping_activation'
# Use bias layers for Mapping network?
MAPPING_USE_BIAS = 'mapping_use_bias'


### ---------- Generator Synthesis network ---------- ###

# Use constant noise instead of random noise?
CONST_INPUT_LAYER = 'const_input_layer'
# Use noise inputs in generator?
USE_NOISE = 'use_noise'
# Randomize noise in generator?
RANDOMIZE_NOISE = 'randomize_noise'
# Use pixel normalization in generator?
USE_PIXEL_NORM = 'use_pixel_norm'
# Use instance normalization in generator?
USE_INSTANCE_NORM = 'use_instance_norm'
# Use style inputs?
USE_STYLES = 'use_styles'
# Use fused layers in generator?
G_FUSED_SCALE = 'G_fused_scale'
# Weights initialization technique for generator: one of He, LeCun (should only be used with Selu)
# Note: gain is selected based on activation, so only use this option if default value is not valid
G_WEIGHTS_INIT_MODE = 'G_weights_init_mode'
# Activation function in generator
G_ACTIVATION = 'G_activation'
# Kernel size of convolutional layers in generator
# Note: only change default value if there is enough video memory,
# as values higher than 3 will lead to increasing training time
G_KERNEL_SIZE = 'G_kernel_size'
# Overall multiplier for the number of feature maps of generator
G_FMAP_BASE = 'G_fmap_base'
# log2 feature map reduction when doubling the resolution of generator
G_FMAP_DECAY = 'G_fmap_decay'
# Maximum number of feature maps in any layer of generator
G_FMAP_MAX = 'G_fmap_max'
# Use smoothing of generator weights?
USE_G_SMOOTHING = 'use_G_smoothing'
# Beta for smoothing weights of generator
G_SMOOTHING_BETA = 'G_smoothed_beta'
# Half-life of the running average of generator weights
G_SMOOTHING_BETA_KIMAGES = 'G_smoothed_beta_kimages'
# Override gain in projecting layer of generator to match the original paper implementation?
OVERRIDE_G_PROJECTING_GAIN = 'override_G_projecting_gain'
# Style strength multiplier for the truncation trick. None = disable
TRUNCATION_PSI = 'truncation_psi'
# Number of layers for which to apply the truncation trick. None = disable
TRUNCATION_CUTOFF = 'truncation_cutoff'
# Decay for tracking the moving average of W during training. None = disable
DLATENT_AVG_BETA = 'dlatent_avg_beta'
# Probability of mixing styles during training. None = disable
STYLE_MIXING_PROB = 'style_mixing prob'


### ---------- Discriminator network ---------- ###

# Use fused layers in discriminator?
D_FUSED_SCALE = 'D_fused_scale'
# Weights initialization technique for discriminator: one of He, LeCun (should only be used with Selu)
# Note: gain is selected based on activation, so only use this option if default value is not valid
D_WEIGHTS_INIT_MODE = 'D_weights_init_mode'
# Activation function in generator
D_ACTIVATION = 'D_activation'
# Kernel size of convolutional layers in generator
# Note: only change default value if there is enough video memory,
# as values higher than 3 will lead to increasing training time
D_KERNEL_SIZE = 'D_kernel_size'
# Overall multiplier for the number of feature maps of discriminator
D_FMAP_BASE = 'D_fmap_base'
# log2 feature map reduction when doubling the resolution of discriminator
D_FMAP_DECAY = 'D_fmap_decay'
# Maximum number of feature maps in any layer of discriminator
D_FMAP_MAX = 'D_fmap_max'
# Group size for minibatch standard deviation layer
MBSTD_GROUP_SIZE = 'mbstd_group_size'
# Number of features for minibatch standard deviation layer
MBSTD_NUM_FEATURES = 'mbstd_num_features'
# Number of filters in a projecting layer of discriminator
# Note: it should only be used if latent size is different from 512,
# it was meant to keep number of parameters in generator and discriminator at roughly
# the same level
D_PROJECTING_NF = 'D_projecting_nf'


### ---------- Dataset options ---------- ###

# Number of parallel calls to dataset
# Note: a good choice is to use a number of cpu cores
DATASET_N_PARALLEL_CALLS = 'dataset_n_parallel_calls'
# Number of prefetched batches for dataset
DATASET_N_PREFETCHED_BATCHES = 'dataset_n_prefetched_batches'
# Maximum number of images to be used for training
DATASET_N_MAX_KIMAGES = 'dataset_max_kimages'
# Shuffle dataset every time it is finished?
# Note: on Windows one might need to set it to False
SHUFFLE_DATASET = 'shuffle_dataset'
# Enable image augmentations?
MIRROR_AUGMENT = 'mirror_augment'
# Max resolution of images to cache dataset (int, 2...max_resolution_log2)
# Note: only use this option of dataset is not very big and there is lots of available memory
DATASET_MAX_CACHE_RES = 'dataset_max_cache_res'


### ---------- Validation images options ---------- ###

# Number of rows in grid of validation images
VALID_GRID_NROWS = 'valid_grid_nrows'
# Number of columns in grid of validation images
VALID_GRID_NCOLS = 'valid_grid_ncols'
# Min size of validation image in grid
VALID_MIN_TARGET_SINGLE_IMAGE_SIZE = 'valid_min_target_single_image_size'
# Max resolution which uses .png format before using .jpeg
# Note: lower resolution have replicated values, so it is better to use .png not to lose information,
# while in higher resolutions difference is not observable, so .jpeg is used as it consumes less emory
VALID_MAX_PNG_RES = 'valid_max_png_res'


DEFAULT_STORAGE_PATH = None
DEFAULT_MAX_MODELS_TO_KEEP = 3
DEFAULT_SUMMARY_SCALARS_EVERY_KIMAGES = 5
DEFAULT_SUMMARY_HISTS_EVERY_KIMAGES = 25
DEFAULT_SAVE_MODEL_EVERY_KIMAGES = 100
DEFAULT_SAVE_IMAGES_EVERY_KIMAGES = 5
DEFAULT_RUN_METRICS_EVERY_KIMAGES = 50
DEFAULT_METRICS_DICT = {}
DEFAULT_DATASET_MAX_CACHE_RES = -1
DEFAULT_START_RESOLUTION = 8
DEFAULT_TOTAL_KIMAGES = -1
DEFAULT_TRANSITION_KIMAGES = 600
DEFAULT_STABILIZATION_KIMAGES = 600
DEFAULT_TRANSITION_KIMAGES_DICT = {}
DEFAULT_STABILIZATION_KIMAGES_DICT = {}
DEFAULT_NORMALIZE_LATENTS = True
DEFAULT_CONST_INPUT_LAYER = True
DEFAULT_USE_NOISE = True
DEFAULT_RANDOMIZE_NOISE = True
DEFAULT_USE_BIAS = True
DEFAULT_USE_WSCALE = True
DEFAULT_USE_PIXEL_NORM = False
DEFAULT_USE_INSTANCE_NORM = True
DEFAULT_USE_STYLES = True
DEFAULT_TRUNCATE_WEIGHTS = False
DEFAULT_BLUR_FILTER = [1, 2, 1]
DEFAULT_MAPPING_LAYERS = 8
DEFAULT_MAPPING_UNITS = 512
DEFAULT_MAPPING_LRMUL = 0.01
DEFAULT_MAPPING_ACTIVATION = 'leaky_relu'
DEFAULT_MAPPING_USE_BIAS = True
DEFAULT_OVERRIDE_G_PROJECTING_GAIN = True
DEFAULT_TRUNCATION_PSI = 0.7
DEFAULT_TRUNCATION_CUTOFF = 8
DEFAULT_DLATENT_AVG_BETA = 0.995
DEFAULT_STYLE_MIXING_PROB = 0.9
DEFAULT_G_FUSED_SCALE = True
DEFAULT_D_FUSED_SCALE = True
DEFAULT_FUSED_BIAS_ACT = True
DEFAULT_DTYPE = 'float32'
DEFAULT_USE_MIXED_PRECISION = True
DEFAULT_NUM_FP16_RESOLUTIONS = 'auto' # 4 - value used in the original implementation
DEFAULT_CONV_CLAMP = None # 256 - value used in the official implementation
DEFAULT_USE_XLA = True
DEFAULT_G_ACTIVATION = 'leaky_relu'
DEFAULT_D_ACTIVATION = 'leaky_relu'
DEFAULT_G_KERNEL_SIZE = 3
DEFAULT_D_KERNEL_SIZE = 3
DEFAULT_BATCH_REPEATS = 1 # 4 - value used in the official implementation
DEFAULT_G_LOSS_FN = 'G_logistic_nonsaturating'
DEFAULT_D_LOSS_FN = 'D_logistic_simplegp'
DEFAULT_G_LOSS_FN_PARAMS = {}
DEFAULT_D_LOSS_FN_PARAMS = {'r1_gamma': 10.0}
DEFAULT_G_LEARNING_RATE = 0.001
DEFAULT_D_LEARNING_RATE = 0.001
DEFAULT_G_LEARNING_RATE_DICT = {}
DEFAULT_D_LEARNING_RATE_DICT = {}
DEFAULT_ADAM_BETA1 = 0.0
DEFAULT_ADAM_BETA2 = 0.99
DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD = True
DEFAULT_USE_G_SMOOTHING = True
# StyleGAN uses different approach co choose beta
DEFAULT_G_SMOOTHING_BETA = None
# DEFAULT_G_SMOOTHING_BETA = 0.999
DEFAULT_G_SMOOTHING_BETA_KIMAGES = 10.0
DEFAULT_MBSTD_NUM_FEATURES = 1
DEFAULT_USE_GPU_FOR_GS = True
DEFAULT_DATASET_N_PARALLEL_CALLS = 4
DEFAULT_DATASET_N_PREFETCHED_BATCHES = 4
DEFAULT_DATASET_N_MAX_KIMAGES = -1
DEFAULT_SHUFFLE_DATASET = True
DEFAULT_MIRROR_AUGMENT = True
DEFAULT_VALID_GRID_NROWS = 5
DEFAULT_VALID_GRID_NCOLS = 7
DEFAULT_VALID_MIN_TARGET_SINGLE_IMAGE_SIZE = 2 ** 7 # or maybe 2**7 ?
DEFAULT_VALID_MAX_PNG_RES = 5

# Note: by default Generator and Discriminator use the same values for these constants
# Note: for the light version described in the appendix set fmap_base to 2048
DEFAULT_FMAP_BASE = 8192
DEFAULT_FMAP_DECAY = 1.0
DEFAULT_FMAP_MAX = 512

HE_GAIN = np.sqrt(2.)
LECUN_GAIN = 1.

GAIN_INIT_MODE_DICT = {
    LECUN_INIT: LECUN_GAIN,
    HE_INIT: HE_GAIN
}

GAIN_ACTIVATION_FUNS_DICT = {
    'relu': HE_GAIN,
    # The same 2 functions
    'leaky_relu': HE_GAIN,
    'lrelu': HE_GAIN,
    # The same 2 functions
    'swish': HE_GAIN,
    'silu': HE_GAIN,
    'gelu': HE_GAIN,
    'mish': HE_GAIN,
    # A special gain is to be used by default
    'selu': LECUN_GAIN
}

ACTIVATION_FUNS_DICT = {
    'linear':     lambda x: x,
    'relu':       lambda x: tf.nn.relu(x),
    # The same 2 functions
    'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=0.2),
    'lrelu':      lambda x: tf.nn.leaky_relu(x, alpha=0.2),
    'selu':       lambda x: tf.nn.selu(x),
    # The same 2 functions
    'swish':      lambda x: tf.nn.swish(x),
    'silu':       lambda x: tf.nn.swish(x),
    'gelu':       lambda x: tf.nn.gelu(x, approximate=False),
    'mish':       lambda x: x * tf.nn.tanh(tf.nn.softplus(x))
}

# Activation function which (might?) need to use float32 dtype
# Should this activation only use fp32?
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
FP32_ACTIVATIONS = ['selu']

# For jupyter notebooks
EXAMPLE_IMAGES_DIR = 'example_images'

# NCHW -> NHWC
toNHWC_AXIS = [0, 2, 3, 1]
# NHWC -> NCHW
toNCHW_AXIS = [0, 3, 1, 2]

LOSS_SCALE_KEY = 'loss_scale'
RANDOMIZE_NOISE_VAR_NAME = 'is_random_noise'

DEBUG_MODE = 'debug_mode'
DEFAULT_DEBUG_MODE = '0'


def should_log_debug_info():
    return int(os.environ.get(DEBUG_MODE, DEFAULT_DEBUG_MODE)) > 0


#----------------------------------------------------------------------------
# Utils.

# Thanks to https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60: return '%ds' % s
    if s < 60 * 60: return '%dm %02ds' % (s // 60, s % 60)
    if s < 24 * 60 * 60: return '%dh %02dm' % (s // (60 * 60), (s // 60) % 60)
    if s < 100 * 24 * 60 * 60: return '%dd %02dh' % (s // (24 * 60 * 60), (s // (60 * 60)) % 24)
    return '>100d'


def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def to_int_dict(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def validate_data_format(data_format):
    assert data_format in [NCHW_FORMAT, NHWC_FORMAT]


def to_z_dim(latent_size, data_format):
    validate_data_format(data_format)
    if data_format == NCHW_FORMAT:
        z_dim = [latent_size, 1, 1]
    else:  # data_format == NHWC_FORMAT:
        z_dim = [1, 1, latent_size]
    return z_dim


def create_images_dir_name(model_name, res, mode):
    return os.path.join(IMAGES_DIR, model_name, f'{2**res}x{2**res}', mode)


def create_images_grid_title(res, mode, step):
    return f'{2**res}x{2**res}, mode={mode}, step={step}'


def load_images_paths(config):
    images_paths_filename = config[IMAGES_PATHS_FILENAME]
    with open(images_paths_filename, 'r') as f:
        file_lines = f.readlines()
    images_paths = [x.strip() for x in file_lines]

    dataset_n_max_images = int(1000 * config.get(DATASET_N_MAX_KIMAGES, DEFAULT_DATASET_N_MAX_KIMAGES))
    if dataset_n_max_images > 0:
        logging.info(f'Dataset number of images: {len(images_paths)}, max number of images: {dataset_n_max_images}')
        if len(images_paths) > dataset_n_max_images:
            logging.info(f'Reduced dataset to {dataset_n_max_images} images')
            images_paths = images_paths[:dataset_n_max_images]

    logging.info(f'Total number of images: {len(images_paths)}')
    return images_paths


def is_last_step(step, n_steps):
    return step == (n_steps - 1)


def should_write_summary(summary_every: int, n_images: int, batch_size: int):
    return (n_images // summary_every > 0 and n_images % summary_every < batch_size) or n_images == batch_size


def level_of_details(res, resolution_log2):
    return resolution_log2 - res + 1


def compute_alpha(step, total_steps):
    return step / total_steps


def update_wsum_alpha(model, alpha):
    def recursive_update_wsum_alpha(model, alpha):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                recursive_update_wsum_alpha(layer, alpha)
            else:
                if WSUM_NAME in layer.name:
                    layer.set_weights([np.array(alpha)])
        return model

    return recursive_update_wsum_alpha(model, alpha)


def enable_random_noise(model):
    for var in model.variables:
        if RANDOMIZE_NOISE_VAR_NAME in var.name:
            var.assign(1.)


def disable_random_noise(model):
    for var in model.variables:
        if RANDOMIZE_NOISE_VAR_NAME in var.name:
            var.assign(-1.)


def mult_by_zero(weights):
    return [0. * w for w in weights]


def create_model_type_key(res, mode):
    return f'{res}_{mode}'


def get_start_fp16_resolution(num_fp16_resolutions, start_resolution_log2, target_resolution_log2):
    # 1) 2 - 4 - start block resolution
    # 2) 3 - 8 - default start resolution
    # 3) 4 - 16
    # 4) 5 - 32
    # 5) 6 - 64
    # 6) 7 - 128
    # 7) 8 - 256
    # 8) 9 - 512
    # 9) 10 - 1024
    if num_fp16_resolutions == 'auto':
        """
        # 1st value: a new init value, 2nd value: taken from the official implementation (N = 4)
        return max(
            min(start_resolution_log2 + 2, target_resolution_log2 - 4 + 1), start_resolution_log2
        )
        """
        # Let start block resolution and two consequent ones use fp32
        return 2 + 2
    else:
        return target_resolution_log2 - num_fp16_resolutions + 1


def should_use_fp16(res, start_fp16_resolution_log2, use_mixed_precision):
    return res >= start_fp16_resolution_log2 and use_mixed_precision


def adjust_clamp(clamp, use_fp16):
    # If layer doesn't use fp16 then values shouldn't be clamped
    return clamp if use_fp16 is True else None


#----------------------------------------------------------------------------
# Tf utils.

def generate_latents(batch_size: int, z_dim: list, dtype=tf.float32):
    return tf.random.normal(shape=[batch_size] + z_dim, mean=0., stddev=1., dtype=dtype)


# Linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t


def enable_mixed_precision_policy():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')


def disable_mixed_precision_policy():
    tf.keras.mixed_precision.set_global_policy('float32')


def fp32(*values):
    if len(values) == 1:
        return tf.cast(values[0], tf.float32)
    return [tf.cast(v, tf.float32) for v in values]


def is_optimizer_ready(optimizer):
    # Optimizer is ready if weights for all net params have been created, one weight for number of iters
    return len(optimizer.weights) > 1


def set_optimizer_iters(optimizer, iters):
    weights = optimizer.get_weights()
    if len(weights) > 0:
        weights[0] = iters
        optimizer.set_weights(weights)
    return optimizer


def maybe_scale_loss(loss, optimizer):
    return optimizer.get_scaled_loss(loss) if optimizer.use_mixed_precision else loss


def maybe_unscale_grads(grads, optimizer):
    return optimizer.get_unscaled_gradients(grads) if optimizer.use_mixed_precision else grads


def custom_unscale_grads(grads, vars, optimizer: tf.keras.mixed_precision.LossScaleOptimizer):
    # Note: works inside tf.function
    # All grads are casted to fp32
    dtype = tf.float32
    # TODO: check if it should be optimizer.loss_scale or optimizer._loss_scale()
    coef = fp32(1. / optimizer.loss_scale)

    def unscale_grad(g, v):
        return fp32(g) * coef if g is not None else (tf.zeros_like(v, dtype=dtype), v)

    grads_array = tf.TensorArray(dtype, len(grads))
    for i in tf.range(len(grads)):
        grads_array = grads_array.write(i, unscale_grad(grads[i], vars[i]))

    return grads_array.stack()


def maybe_custom_unscale_grads(grads, vars, optimizer):
    return custom_unscale_grads(grads, vars, optimizer) if optimizer.use_mixed_precision else grads


def is_finite_grad(grad):
    return tf.equal(tf.math.count_nonzero(~tf.math.is_finite(grad)), 0)


def trace_vars(vars, title):
    # Ugly way to trace variables:
    # Python side-effects will only happen once, when func is traced
    log_debug_info = should_log_debug_info()
    if log_debug_info:
        os_name = platform.system()
        if OS_LINUX == os_name:
            logging.info('\n' + title)
            [logging.info(var.name) for var in vars]
        else:
            # Calling logging inside tf.function on Windows can cause error
            print('\n' + title)
            [print(var.name) for var in vars]
            print()


def trace_message(message):
    log_debug_info = should_log_debug_info()
    if log_debug_info:
        os_name = platform.system()
        if os_name == OS_LINUX:
            logging.info(message)
        else:
            # Calling logging inside tf.function on Windows can cause errors
            print(message)


def set_tf_logging(debug_mode=True):
    # https://github.com/tensorflow/tensorflow/issues/31870, see a comment by ziyigogogo on 28 Aug 2019
    # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information, answer by craymichael
    if debug_mode:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.get_logger().setLevel('INFO')
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)


def prepare_gpu(mode='auto', memory_limit=None):
    os_name = platform.system()
    os_message = f'\nScript is running on {os_name}, '

    # Note: change this number based on your GPU
    if memory_limit is None:
        memory_limit = 7500
    set_memory_growth = False
    set_memory_limit = False

    if mode == 'auto':
        if os_name == OS_LINUX:
            print(os_message + 'memory growth option is used')
            set_memory_growth = True
        elif os_name == OS_WIN:
            print(os_message + 'memory limit option is used')
            set_memory_limit = True
        else:
            print(
                os_message + f'GPU can only be configured for {OS_LINUX}|{OS_WIN}, '
                f'memory growth option is used'
            )
            set_memory_growth = True
    else:
        assert mode in ['growth', 'limit']
        if mode == 'growth':
            set_memory_growth = True
        else:
            set_memory_limit = True

    physical_gpus = tf.config.experimental.list_physical_devices('GPU')

    if set_memory_limit:
        if len(physical_gpus) > 0:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ]
                    )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(
                    f'Physical GPUs: {len(physical_gpus)}, logical GPUs: {len(logical_gpus)}'
                )
                print(f'Set memory limit to {memory_limit} Mbs\n')
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('GPU is not available\n')

    if set_memory_growth:
        if len(physical_gpus) > 0:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Physical GPUs: {len(physical_gpus)} \nSet memory growth\n')
        else:
            print('GPU is not available\n')


#----------------------------------------------------------------------------
# Model saving/loading/remove utils.

h5_weights_key = 'weights'


def weights_to_dict(model, optimizer_call=False):
    vars = model.trainable_variables if not optimizer_call else model.weights
    if should_log_debug_info():
        print('\nSaving weights:')
        for idx, var in enumerate(vars):
            print(f'{idx}: {var.name}')
    return {var.name: var.numpy() for var in vars}


def load_model_weights_from_dict(model, weights_dict):
    log_debug_info = should_log_debug_info()
    print('\nLoading weights from dict')
    print('\nModel train vars:', model.trainable_variables)
    print('\nDict vars:', list(weights_dict.keys()))
    for var in model.trainable_variables:
        if var.name in weights_dict.keys():
            if log_debug_info:
                print(f'Loading {var.name}')
            var.assign(weights_dict[var.name])

    return model


def save_weights(weights_dict, filename):
    f = h5py.File(filename, 'w')
    g = f.create_group(h5_weights_key)

    for idx, var_name in enumerate(weights_dict.keys()):
        value = weights_dict[var_name]
        shape = value.shape
        dset = g.create_dataset(name=var_name, shape=shape, dtype=value.dtype.name)
        # TODO: for debugging, remove later
        # print(f'{idx}) {var_name}: {value.mean():.4f}, std={value.std():.4f}')
        if not shape:
            # Scalar
            dset[()] = value
        else:
            dset[:] = value

    f.flush()
    f.close()


def load_weights_into_dict(var_names, filename):
    f = h5py.File(filename, 'r')
    g = f[h5_weights_key]

    var_dict = {}
    for var_name in var_names:
        if var_name in g:
            # Check for scalar
            try:
                var_dict[var_name] = g[var_name][:]
            except:
                var_dict[var_name] = g[var_name][()]

    f.close()
    return var_dict


def load_weights(model, filename, optimizer_call=False):
    vars = model.trainable_variables if not optimizer_call else model.weights
    var_names = [var.name for var in vars]
    var_dict = load_weights_into_dict(var_names, filename)

    log_debug_info = should_log_debug_info()
    loaded_vars = []
    # Note: another approach is to use set_weights (load ur use existing value) for models and optimizers (maybe with deepcopy?)
    for var in vars:
        if var.name in var_dict.keys():
            if log_debug_info:
                print(f'Loading {var.name}')
            # Might be a Strange line for optimizer
            var.assign(var_dict[var.name])
            loaded_vars.append(var.name)

    # TODO: for debugging, remove later
    # print('\nOptimizer call:', optimizer_call)
    # print('Loaded these vars:\n', loaded_vars)

    return model


def create_model_dir_path(model_name, res, stage, step=None, storage_path=DEFAULT_STORAGE_PATH):
    """
    model_name - name of configuration model
    res - current resolution
    stage - one of [TRANSITION_MODE, STABILIZATION_MODE]
    step - number of steps (or processed images) for given resolution and stage
    storage_path - optional prefix path
    """
    res_dir = f'{2**res}x{2**res}'
    stage_dir = stage
    step_dir = 'step' + str(step) if step is not None else ''
    model_dir_path = os.path.join(WEIGHTS_DIR, model_name, res_dir, stage_dir, step_dir)

    if storage_path is not None:
        model_dir_path = os.path.join(storage_path, model_dir_path)

    return model_dir_path


def save_model(model, model_name, model_type, res,
               stage, step, storage_path=DEFAULT_STORAGE_PATH):
    """
    model - a model to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current log2 resolution
    stage - one of [TRANSITION_MODE, STABILIZATION_MODE]
    step - number of steps (or processed images) for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    optimizer_call = OPTIMIZER_POSTFIX in model_type

    model_dir_path = create_model_dir_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=step,
        storage_path=storage_path
    )
    if optimizer_call:
        model_dir_path += OPTIMIZER_POSTFIX
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    filepath = os.path.join(model_dir_path, model_type + '.h5')
    weights_dict = weights_to_dict(model, optimizer_call=optimizer_call)

    save_weights(weights_dict, filepath)


def save_optimizer_loss_scale(optimizer: tf.keras.mixed_precision.LossScaleOptimizer,
                              model_name: str, model_type: str, res: int, stage: str,
                              step: int, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    optimizer - an optimizer model from which loss scale is to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current log2 resolution
    stage - one of [TRANSITION_MODE, STABILIZATION_MODE]
    step - number of steps (or processed images) for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    model_dir_path = create_model_dir_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=step,
        storage_path=storage_path
    )
    model_dir_path += OPTIMIZER_POSTFIX
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    # This function is only called when loss scale is dynamic
    loss_scale = float(optimizer._loss_scale().numpy())
    if should_log_debug_info():
        print(f'Saved loss scale for {model_type}: {loss_scale}')

    filepath = os.path.join(model_dir_path, model_type + '.json')
    with open(filepath, 'w') as fp:
        json.dump({LOSS_SCALE_KEY: loss_scale}, fp)


def load_model(model, model_name, model_type, res,
               stage, step, storage_path=DEFAULT_STORAGE_PATH):
    """
    model - a model to be loaded
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME], used as a separate dir level
    res - current log2 resolution
    stage - one of [TRANSITION_MODE, STABILIZATION_MODE]
    step - number of steps (or processed images) for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of loading model
    """
    optimizer_call = OPTIMIZER_POSTFIX in model_type

    model_dir_path = create_model_dir_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=step,
        storage_path=storage_path
    )
    if optimizer_call:
        model_dir_path += OPTIMIZER_POSTFIX
    assert os.path.exists(model_dir_path),\
        f"Can't load weights: directory {model_dir_path} does not exist"

    filepath = os.path.join(model_dir_path, model_type + '.h5')
    model = load_weights(model, filepath, optimizer_call=optimizer_call)
    return model


def load_optimizer_loss_scale(model_name: str, model_type: str, res: int, stage: str,
                              step: int, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    optimizer - an optimizer model from which loss scale is to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current log2 resolution
    stage - one of [TRANSITION_MODE, STABILIZATION_MODE]
    step - number of steps (or processed images) for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    model_dir_path = create_model_dir_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=step,
        storage_path=storage_path
    )
    model_dir_path += OPTIMIZER_POSTFIX
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    filepath = os.path.join(model_dir_path, model_type +  OPTIMIZER_POSTFIX + '.json')
    with open(filepath, 'r') as fp:
        loss_scale = json.load(fp)[LOSS_SCALE_KEY]

    if should_log_debug_info():
        print(f'Loaded loss scale for {model_type}: {loss_scale}')

    return loss_scale


def remove_old_models(model_name, res, stage, max_models_to_keep, storage_path=DEFAULT_STORAGE_PATH):
    """
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current resolution
    max_models_to_keep - max number of models to keep
    storage_path - optional prefix path
    """
    # step and model_type are not used, so jut use valid values
    log_debug_info = should_log_debug_info()
    if log_debug_info:
        logging.info('\nRemoving weights...')
    weights_path = create_model_dir_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=1,
        storage_path=storage_path
    )
    res_stage_path = os.path.split(weights_path)[0]
    sorted_steps_paths = sorted(
        [x for x in glob(res_stage_path + os.sep + '*') if 'step' in x],
        key=lambda x: int(x.split('step')[1])
    )
    # Remove weights for all steps except the last one
    for p in sorted_steps_paths[:-max_models_to_keep]:
        shutil.rmtree(p)
        if log_debug_info:
            logging.info(f'Removed weights for path={p}')
