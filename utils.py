import os
import json
import logging
import platform

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from config import Config as cfg


# Recommended for Tensorflow
NCHW_FORMAT = 'NCHW'
# Recommended by Nvidia
NHWC_FORMAT = 'NHWC'

DEFAULT_DATA_FORMAT = NCHW_FORMAT
DEFAULT_DTYPE = 'float32'
DEFAULT_USE_FP16 = False

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
BENCHMARK_MODE = 'benchmark'
DEFAULT_MODE = INFERENCE_MODE


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
    # The same 2 functions
    'hard_mish': HE_GAIN,
    'hmish': HE_GAIN,
    # The same 2 functions
    'hard_swish': HE_GAIN,
    'hswish': HE_GAIN,
    # A special gain is to be used by default
    'selu': LECUN_GAIN
}


@tf.custom_gradient
def mish(x):
    x = tf.convert_to_tensor(x, name="features")

    # Inspired by source code for tf.nn.swish
    def grad(dy):
        """ Gradient for the Mish activation function"""
        # Naively, x * tf.nn.tanh(tf.nn.softplus(x)) requires keeping both
        # x and tf.nn.tanh(tf.nn.softplus(x)) around for backprop,
        # effectively doubling the tensor's memory consumption.
        # We use a control dependency here so that tanh(softplus(x)) is re-computed
        # during backprop (the control dep prevents it being de-duped with the
        # forward pass) and we can free the tanh(softplus(x)) expression immediately
        # after use during the forward pass.
        with tf.control_dependencies([dy]):
            x_tanh_sp = tf.nn.tanh(tf.nn.softplus(x))
        x_sigmoid = tf.nn.sigmoid(x)
        activation_grad = x_tanh_sp + x * x_sigmoid * (1.0 - x_tanh_sp * x_tanh_sp)
        return dy * activation_grad

    return x * tf.nn.tanh(tf.nn.softplus(x)), grad


def hard_mish(x):
    x = tf.convert_to_tensor(x, name="features")
    return tf.minimum(2.0, tf.nn.relu(x + 2.0)) * 0.5 * x


def hard_swish(x):
    x = tf.convert_to_tensor(x, name="features")
    return x * tf.nn.relu6(x + 3.0) * 0.16666667


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
    'mish':       lambda x: mish(x),
    # The same 2 functions
    'hard_mish':  lambda x: hard_mish(x),
    'hmish':      lambda x: hard_mish(x),
    # The same 2 functions
    'hard_swish': lambda x: hard_swish(x),
    'hswish':     lambda x: hard_swish(x)
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


def validate_hw_ratio(hw_ratio):
    assert hw_ratio == 1 or 0.1 < hw_ratio < 1.0


def to_z_dim(latent_size, data_format) -> list:
    validate_data_format(data_format)
    if data_format == NCHW_FORMAT:
        z_dim = [latent_size, 1, 1]
    else:  # data_format == NHWC_FORMAT:
        z_dim = [1, 1, latent_size]
    return z_dim


def to_hw_size(image_size, hw_ratio) -> tuple:
    validate_hw_ratio(hw_ratio)
    return (int(hw_ratio * image_size), image_size)


def create_images_dir_path(model_name, res, mode):
    return os.path.join(IMAGES_DIR, model_name, f'{2**res}x{2**res}', mode)


def create_images_grid_title(res, mode, step):
    return f'{2**res}x{2**res}, mode={mode}, step={step}'


def get_compute_dtype(use_mixed_precision):
    return mixed_precision.Policy('mixed_float16').compute_dtype if use_mixed_precision else 'float32'


def load_images_paths(config):
    images_paths_filename = config[cfg.IMAGES_PATHS_FILENAME]
    with open(images_paths_filename, 'r') as f:
        file_lines = f.readlines()
    images_paths = [x.strip() for x in file_lines]

    dataset_n_max_images = int(1000 * config.get(cfg.DATASET_N_MAX_KIMAGES, cfg.DEFAULT_DATASET_N_MAX_KIMAGES))
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


def update_wsum_alpha(model: tf.keras.Model, alpha):
    def recursive_update_wsum_alpha(model, alpha):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                recursive_update_wsum_alpha(layer, alpha)
            else:
                if WSUM_NAME in layer.name:
                    layer.set_weights([np.array(alpha)])
        return model

    return recursive_update_wsum_alpha(model, alpha)


def enable_random_noise(model: tf.keras.Model):
    for var in model.variables:
        if RANDOMIZE_NOISE_VAR_NAME in var.name:
            var.assign(1.)


def disable_random_noise(model: tf.keras.Model):
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


def get_gpu_memory_usage():
    """
    Returns dict with info about memory usage (in Mb) by each GPU. {device_name: {dict}}
    """
    def format_device_name(n):
        # For Tf2.5 name is like '/device:GPU:0', for other versions this function might need changes
        return n.split('device')[1][1:]

    def to_mb_size(bytes_size):
        return int(bytes_size / (1024 * 1024))

    stats = {}
    for device in tf.config.experimental.list_logical_devices('GPU'):
        stats[format_device_name(device.name)] = {k: to_mb_size(v) for k, v in tf.config.experimental.get_memory_info(device.name).items()}
    return stats


def generate_latents(batch_size: int, z_dim: list, dtype=tf.float32):
    return tf.random.normal(shape=[batch_size] + z_dim, mean=0., stddev=1., dtype=dtype)


def restore_images(images):
    # Minimum OP doesn't support uint8
    images = tf.math.round((images + 1.0) * (255 / 2))
    images = tf.clip_by_value(tf.cast(images, dtype=tf.int32), 0, 255)
    images = tf.cast(images, dtype=tf.uint8)
    return images


def convert_outputs_to_images(net_outputs, target_single_image_size, hw_ratio=1, data_format=DEFAULT_DATA_FORMAT):
    # Note: should work for linear and tanh activation
    # 1. Adjust dynamic range of images
    x = restore_images(net_outputs)
    # 2. Optionally change data format
    validate_data_format(data_format)
    if data_format == NCHW_FORMAT:
        x = tf.transpose(x, toNHWC_AXIS)
    # 3. Optionally extract target images for wide dataset
    validate_hw_ratio(hw_ratio)
    x = extract_images(x, hw_ratio, NHWC_FORMAT)
    # 4. Resize images
    x = tf.image.resize(
        x,
        size=to_hw_size(target_single_image_size, hw_ratio),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return x


def extract_images(x, hw_ratio, data_format):
    if hw_ratio != 1:
        s = tf.shape(x)
        if data_format == NCHW_FORMAT:
            n, c, h, w = s[0], s[1], s[2], s[3]
            h = int(hw_ratio * float(h))  # h is tensor, so dtypes must match
            x = x[:, :, (w - h) // 2 : (w + h) // 2, :]
        else: # data_format == NHWC_FORMAT
            n, h, w, c = s[0], s[1], s[2], s[3]
            h = int(hw_ratio * float(h))  # h is tensor, so dtypes must match
            x = x[:, (w - h) // 2 : (w + h) // 2, :, :]
    return x


# Linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t


@tf.function
def smooth_model_weights(sm_model, src_model, beta, device):
    trace_message('...Tracing smoothing weights...')
    smoothed_net_vars = sm_model.trainable_variables
    source_net_vars = src_model.trainable_variables
    trace_vars(smoothed_net_vars, 'Smoothed vars:')
    trace_vars(source_net_vars, 'Source vars:')
    with tf.device(device):
        for a, b in zip(smoothed_net_vars, source_net_vars):
            a.assign(lerp(b, a, beta))


def enable_mixed_precision_policy():
    mixed_precision.set_global_policy('mixed_float16')


def disable_mixed_precision_policy():
    mixed_precision.set_global_policy('float32')


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


def custom_unscale_grads(grads, vars, optimizer: mixed_precision.LossScaleOptimizer):
    # Note: works inside tf.function
    # All grads are casted to fp32
    dtype = tf.float32
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
        memory_limit = 7700
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
