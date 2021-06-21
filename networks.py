import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import mixed_precision

from custom_layers import WeightedSum, LRMUL,\
    dense_layer, conv2d_layer, bias_layer, act_layer, const_layer, noise_layer, blur_layer,\
    pixel_norm_layer, instance_norm_layer, style_mod_layer, downscale2d_layer, upscale2d_layer, minibatch_stddev_layer
from utils import TRANSITION_MODE, STABILIZATION_MODE, WSUM_NAME,\
    GAIN_INIT_MODE_DICT, GAIN_ACTIVATION_FUNS_DICT,\
    level_of_details, validate_data_format, create_model_type_key, to_int_dict

from utils import TARGET_RESOLUTION, START_RESOLUTION,\
    LATENT_SIZE, DLATENT_SIZE, NORMALIZE_LATENTS,\
    USE_NOISE, RANDOMIZE_NOISE,\
    DATA_FORMAT, USE_MIXED_PRECISION, USE_BIAS, USE_WSCALE,\
    USE_PIXEL_NORM, USE_INSTANCE_NORM, USE_STYLES, CONST_INPUT_LAYER, TRUNCATE_WEIGHTS,\
    G_FUSED_SCALE, G_WEIGHTS_INIT_MODE, G_ACTIVATION, G_KERNEL_SIZE,\
    D_FUSED_SCALE, D_WEIGHTS_INIT_MODE, D_ACTIVATION, D_KERNEL_SIZE,\
    MBSTD_GROUP_SIZE, OVERRIDE_G_PROJECTING_GAIN, D_PROJECTING_NF,\
    MAPPING_LAYERS, MAPPING_UNITS, MAPPING_LRMUL, MAPPING_ACTIVATION,\
    G_FMAP_BASE, G_FMAP_DECAY, G_FMAP_MAX,\
    D_FMAP_BASE, D_FMAP_DECAY, D_FMAP_MAX,\
    BATCH_SIZES
from utils import NCHW_FORMAT, NHWC_FORMAT, DEFAULT_DATA_FORMAT,\
    DEFAULT_USE_MIXED_PRECISION, DEFAULT_START_RESOLUTION,\
    DEFAULT_MAPPING_LAYERS, DEFAULT_MAPPING_UNITS, DEFAULT_MAPPING_LRMUL, DEFAULT_MAPPING_ACTIVATION,\
    DEFAULT_OVERRIDE_G_PROJECTING_GAIN, \
    DEFAULT_NORMALIZE_LATENTS, DEFAULT_CONST_INPUT_LAYER,\
    DEFAULT_USE_NOISE, DEFAULT_RANDOMIZE_NOISE,\
    DEFAULT_G_ACTIVATION, DEFAULT_D_ACTIVATION,\
    DEFAULT_G_FUSED_SCALE, DEFAULT_D_FUSED_SCALE,\
    DEFAULT_G_KERNEL_SIZE, DEFAULT_D_KERNEL_SIZE, DEFAULT_USE_BIAS,\
    DEFAULT_USE_PIXEL_NORM, DEFAULT_USE_INSTANCE_NORM, DEFAULT_USE_STYLES,\
    DEFAULT_TRUNCATE_WEIGHTS, DEFAULT_USE_WSCALE,\
    DEFAULT_FMAP_BASE, DEFAULT_FMAP_DECAY, DEFAULT_FMAP_MAX

from utils import weights_to_dict, load_model_weights_from_dict


def n_filters(stage, fmap_base, fmap_decay, fmap_max):
    """
    fmap_base  Overall multiplier for the number of feature maps.
    fmap_decay log2 feature map reduction when doubling the resolution.
    fmap_max   Maximum number of feature maps in any layer.
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class Generator:

    def __init__(self, config):
        self.config = config

        self.target_resolution = config[TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.latent_size = config[LATENT_SIZE]
        self.dlatent_size = config[DLATENT_SIZE]
        self.normalize_latents = config.get(NORMALIZE_LATENTS, DEFAULT_NORMALIZE_LATENTS)
        self.const_input_layer = config.get(CONST_INPUT_LAYER, DEFAULT_CONST_INPUT_LAYER)
        self.use_noise = config.get(USE_NOISE, DEFAULT_USE_NOISE)
        self.randomize_noise = config.get(RANDOMIZE_NOISE, DEFAULT_RANDOMIZE_NOISE)
        self.use_bias = config.get(USE_BIAS, DEFAULT_USE_BIAS)
        self.use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
        self.use_pixel_norm = config.get(USE_PIXEL_NORM, DEFAULT_USE_PIXEL_NORM)
        self.use_instance_norm = config.get(USE_INSTANCE_NORM, DEFAULT_USE_INSTANCE_NORM)
        self.use_styles = config.get(USE_STYLES, DEFAULT_USE_STYLES)
        self.truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
        self.G_fused_scale = config.get(G_FUSED_SCALE, DEFAULT_G_FUSED_SCALE)
        self.G_kernel_size = config.get(G_KERNEL_SIZE, DEFAULT_G_KERNEL_SIZE)
        self.G_fmap_base = config.get(G_FMAP_BASE, DEFAULT_FMAP_BASE)
        self.G_fmap_decay = config.get(G_FMAP_DECAY, DEFAULT_FMAP_DECAY)
        self.G_fmap_max = config.get(G_FMAP_MAX, DEFAULT_FMAP_MAX)
        self.G_act_name = config.get(G_ACTIVATION, DEFAULT_G_ACTIVATION)

        self.mapping_layers = config.get(MAPPING_LAYERS, DEFAULT_MAPPING_LAYERS)
        self.mapping_units = config.get(MAPPING_UNITS, DEFAULT_MAPPING_UNITS)
        self.mapping_lrmul = config.get(MAPPING_LRMUL, DEFAULT_MAPPING_LRMUL)
        self.mapping_act_name = config.get(MAPPING_ACTIVATION, DEFAULT_MAPPING_ACTIVATION)
        self.mapping_gain = GAIN_ACTIVATION_FUNS_DICT[self.mapping_act_name]

        self.use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
        self.policy = mixed_precision.Policy('mixed_float16') if self.use_mixed_precision else 'float32'
        self.compute_dtype = self.policy.compute_dtype if self.use_mixed_precision else 'float32'

        self.weights_init_mode = config.get(G_WEIGHTS_INIT_MODE, None)
        if self.weights_init_mode is None:
            self.gain = GAIN_ACTIVATION_FUNS_DICT[self.G_act_name]
        else:
            self.gain = GAIN_INIT_MODE_DICT[self.weights_init_mode]

        self.override_projecting_gain = config.get(OVERRIDE_G_PROJECTING_GAIN, DEFAULT_OVERRIDE_G_PROJECTING_GAIN)
        # Gain is overridden to match the original ProGAN implementation
        # sqrt(2) / 4 was used with He init
        self.projecting_gain_correction = 4. if self.override_projecting_gain else 1.
        self.projecting_gain = self.gain / self.projecting_gain_correction

        # This constant is taken from the original implementation
        self.projecting_mult = 4
        if self.data_format == NCHW_FORMAT:
            self.z_dim = (self.latent_size, 1, 1)
            self.projecting_target_shape = (-1, self.G_n_filters(1), self.projecting_mult, self.projecting_mult)
        elif self.data_format == NHWC_FORMAT:
            self.z_dim = (1, 1, self.latent_size)
            self.projecting_target_shape = (-1, self.projecting_mult, self.projecting_mult, self.G_n_filters(1))
        self.projecting_units = np.prod(self.projecting_target_shape)

        self.num_layers = self.resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if self.use_styles else 1
        self.batch_sizes = to_int_dict(config[BATCH_SIZES])

        self.create_model_layers()
        self.G_models = {}

    def G_n_filters(self, stage):
        return n_filters(stage, self.G_fmap_base, self.G_fmap_decay, self.G_fmap_max)

    def G_output_shape(self, res):
        nf = self.G_n_filters(res - 1)
        if self.data_format == NCHW_FORMAT:
            return [nf, 2 ** res, 2 ** res]
        elif self.data_format == NHWC_FORMAT:
            return [2 ** res, 2 ** res, nf]

    def create_model_layers(self):
        self.toRGB_layers = {
            res: self.to_rgb_layer(res) for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.latents = Input(self.z_dim, dtype=self.compute_dtype, name='Latents')
        self.create_G_mapping()

    def to_rgb_layer(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'ToRGB_lod{lod}'

        with tf.name_scope(block_name) as scope:
            x = Input(self.G_output_shape(res))
            y = self.apply_bias(self.conv2d(x, fmaps=3, kernel_size=1, gain=1., scope=scope), scope=scope)

        return tf.keras.Model(x, y, name=block_name)

    def to_rgb(self, x, res):
        return self.toRGB_layers[res](x)

    def blur(self, x, scope=''):
        return blur_layer(x, scope=scope, config=self.config)

    def dense(self, x, units, gain=None, lrmul=None, scope=''):
        if gain is None: gain = self.gain
        if lrmul is None: lrmul = LRMUL
        return dense_layer(x, units, lrmul=lrmul, gain=gain, config=self.config, scope=scope)

    def conv2d(self, x, fmaps, kernel_size=None, gain=None, fused_up=False, scope=''):
        if kernel_size is None: kernel_size = self.G_kernel_size
        if gain is None: gain = self.gain
        return conv2d_layer(
            x, fmaps, kernel_size=kernel_size, gain=gain,
            fused_up=fused_up, scope=scope, config=self.config
        )

    def upscale2d(self, x):
        return upscale2d_layer(x, factor=2, config=self.config)

    def upscale2d_conv2d(self, x, fmaps, scope=''):
        if self.G_fused_scale:
            x = self.conv2d(x, fmaps=fmaps, fused_up=True, scope=scope)
        else:
            x = self.upscale2d(x)
            x = self.conv2d(x, fmaps=fmaps, scope=scope)
        return x

    def act(self, x, act_name=None, scope=''):
        if act_name is None: act_name = self.G_act_name
        return act_layer(x, act_name, scope=scope, config=self.config)

    def apply_bias(self, x, lrmul=None, scope=''):
        if lrmul is None: lrmul=LRMUL
        return bias_layer(x, lrmul, scope=scope, config=self.config) if self.use_bias else x

    def layer_epilogue(self, x, layer_idx, scope):
        if self.use_noise:
            x = noise_layer(x, scope=scope, config=self.config)
        x = self.apply_bias(x, scope=scope)
        x = self.act(x, scope=scope)
        if self.use_pixel_norm:
            x = pixel_norm_layer(x, scope=scope, config=self.config)
        if self.use_instance_norm:
            x = instance_norm_layer(x, scope=scope, config=self.config)
        if self.use_styles:
            x = style_mod_layer(x, self.dlatents[:, layer_idx], scope=scope, config=self.config)
        return x

    def create_G_mapping(self):
        x = self.latents
        if self.normalize_latents:
            #with tf.name_scope('Latents_normalizer') as scope:
            x = pixel_norm_layer(x, config=self.config)

        with tf.name_scope('G_mapping'):
            for layer_idx in range(self.mapping_layers):
                with tf.name_scope('Dense%d' % layer_idx) as scope:
                    units = self.dlatent_size if layer_idx == self.mapping_layers - 1 else self.mapping_units
                    x = self.dense(x, units, gain=self.mapping_gain, lrmul=self.mapping_lrmul, scope=scope)
                    x = self.apply_bias(x, lrmul=self.mapping_lrmul, scope=scope)
                    x = self.act(x, act_name=self.mapping_act_name, scope=scope)

            with tf.name_scope('Broadcast'):
                x = tf.tile(x[:, np.newaxis], [1, self.num_styles, 1])

        self.G_mapping = tf.keras.Model(self.latents, tf.identity(x, name='dlatents'), name='G_mapping')
        self.dlatents = self.G_mapping(self.latents)

    def input_block(self):
        with tf.name_scope('4x4'):
            if self.const_input_layer:
                with tf.name_scope('Const') as scope:
                    x = const_layer(self.latents, self.latent_size, scope=scope, config=self.config)
                    x = self.layer_epilogue(x, 0, scope=scope)
            else:
                with tf.name_scope('Dense') as scope:
                    x = self.dense(
                        self.dlatents[:, 0], units=self.projecting_units,
                        gain=self.projecting_gain, scope=scope
                    )
                    x = self.layer_epilogue(tf.reshape(x, self.projecting_target_shape), 0, scope=scope)
            with tf.name_scope('Conv1') as scope:
                x = self.layer_epilogue(self.conv2d(x, fmaps=self.G_n_filters(1), scope=scope), 1, scope=scope)
        return x

    def G_block(self, x, res):
        # res = 3 ... resolution_log2
        with tf.name_scope(f'{2**res}x{2**res}'):
            with tf.name_scope('Conv0_up') as scope:
                x = self.blur(self.upscale2d_conv2d(x, fmaps=self.G_n_filters(res - 1), scope=scope), scope=scope)
                x = self.layer_epilogue(x, res * 2 - 4, scope=scope)
            with tf.name_scope('Conv1') as scope:
                x = self.layer_epilogue(self.conv2d(x, fmaps=self.G_n_filters(res - 1), scope=scope), res * 2 - 3, scope=scope)
        return x

    def create_G_model(self, model_res, mode=STABILIZATION_MODE):
        # Note: model_res is should be log2(image_res)
        # TODO: for debugging, remove later
        print(f'Calling create G model: res={model_res}')
        assert mode in [TRANSITION_MODE, STABILIZATION_MODE], 'Mode ' + mode + ' is not supported'
        model_type_key = create_model_type_key(model_res, mode)
        if model_type_key not in self.G_models.keys():
            if model_res == 2:
                x = self.input_block()
                images_out = self.to_rgb(x, model_res)
            else:
                x = self.input_block()
                for res in range(3, model_res, 1):
                    x = self.G_block(x, res)
                if mode == STABILIZATION_MODE:
                    x = self.G_block(x, model_res)
                    images_out = self.to_rgb(x, model_res)
                else:
                    # Last output layers
                    images1 = self.to_rgb(x, model_res - 1)
                    images1 = self.upscale2d(images1)
                    # Introduce new layers
                    images2 = self.G_block(x, model_res)
                    images2 = self.to_rgb(images2, model_res)
                    lod = level_of_details(model_res, self.resolution_log2)
                    wsum_name = f'G_{WSUM_NAME}_lod{lod}'
                    images_out = WeightedSum(dtype=self.policy, name=wsum_name)([images1, images2])

            self.G_models[model_type_key] = tf.keras.Model(
                self.latents, tf.identity(images_out, name='images_out'), name='G_style'
            )

        return self.G_models[model_type_key]

    def initialize_G_model(self, model_res=None, mode=None):
        if model_res is not None:
            batch_size = self.batch_sizes[2 ** model_res]
            latents = tf.zeros(
                shape=(batch_size,) + self.z_dim, dtype=self.compute_dtype
            )
            G_model = self.create_G_model(model_res, mode=mode)
            _ = G_model(latents)
        else:
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
                batch_size = self.batch_sizes[2 ** res]
                latents = tf.zeros(
                    shape=(batch_size,) + self.z_dim, dtype=self.compute_dtype
                )
                G_model = self.create_G_model(res, mode=TRANSITION_MODE)
                _ = G_model(latents)

        print('G model built')

    def save_G_weights_in_class(self, G_model):
        self.G_weights_dict = weights_to_dict(G_model)

    def load_G_weights_from_class(self, G_model):
        return load_model_weights_from_dict(G_model, self.G_weights_dict)

    def trace_G_graphs(self, summary_writers, writers_dirs):
        G_prefix = 'Generator_'
        trace_G_input = tf.zeros(shape=(1,) + self.z_dim, dtype=self.compute_dtype)
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            writer = summary_writers[res]
            profiler_dir = writers_dirs[res]
            if res == self.start_resolution_log2:
                trace_G_model = tf.function(self.create_G_model(res, mode=STABILIZATION_MODE))
                with writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_G_model(trace_G_input)
                    tf.summary.trace_export(
                        name=G_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()
                    writer.flush()
            else:
                trace_G_model1 = tf.function(self.create_G_model(res, mode=TRANSITION_MODE))
                trace_G_model2 = tf.function(self.create_G_model(res, mode=STABILIZATION_MODE))
                with writer.as_default():
                    # Transition model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_G_model1(trace_G_input)
                    tf.summary.trace_export(
                        name=G_prefix + TRANSITION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()
                    # Stabilization model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_G_model2(trace_G_input)
                    tf.summary.trace_export(
                        name=G_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()
                    writer.flush()

        print('All Generator models traced!')


class Discriminator:

    def __init__(self, config):
        self.config = config

        self.target_resolution = config[TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.use_bias = config.get(USE_BIAS, DEFAULT_USE_BIAS)
        self.use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
        self.truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
        self.mbstd_group_size = config[MBSTD_GROUP_SIZE]
        self.D_fused_scale = config.get(D_FUSED_SCALE, DEFAULT_D_FUSED_SCALE)
        self.D_kernel_size = config.get(D_KERNEL_SIZE, DEFAULT_D_KERNEL_SIZE)
        self.D_fmap_base = config.get(D_FMAP_BASE, DEFAULT_FMAP_BASE)
        self.D_fmap_decay = config.get(D_FMAP_DECAY, DEFAULT_FMAP_DECAY)
        self.D_fmap_max = config.get(D_FMAP_MAX, DEFAULT_FMAP_MAX)
        self.D_act_name = config.get(D_ACTIVATION, DEFAULT_D_ACTIVATION)

        self.use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
        self.policy = mixed_precision.Policy('mixed_float16') if self.use_mixed_precision else 'float32'
        self.compute_dtype = self.policy.compute_dtype if self.use_mixed_precision else 'float32'

        self.weights_init_mode = config.get(D_WEIGHTS_INIT_MODE, None)
        if self.weights_init_mode is None:
            self.gain = GAIN_ACTIVATION_FUNS_DICT[self.D_act_name]
        else:
            self.gain = GAIN_INIT_MODE_DICT[self.weights_init_mode]

        # Might be useful to override number of units in projecting layer
        # in case latent size is not 512 to make models have almost the same number
        # of trainable params
        self.projecting_nf = config.get(D_PROJECTING_NF, self.D_n_filters(2 - 2))

        self.batch_sizes = to_int_dict(config[BATCH_SIZES])

        self.create_model_layers()
        self.D_models = {}

    def D_n_filters(self, stage):
        return n_filters(stage, self.D_fmap_base, self.D_fmap_decay, self.D_fmap_max)

    def D_input_shape(self, res):
        if self.data_format == NCHW_FORMAT:
            return (3, 2 ** res, 2 ** res)
        elif self.data_format == NHWC_FORMAT:
            return (2 ** res, 2 ** res, 3)

    def create_model_layers(self):
        self.D_input_layers = {
            res: Input(
                shape=self.D_input_shape(res), dtype=self.compute_dtype, name=f'Images_{2**res}x{2**res}'
            ) for res in range(2, self.resolution_log2 + 1)
        }
        self.fromRGB_layers = {
            res: self.from_rgb_layer(res) for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }

    def from_rgb_layer(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'FromRGB_lod{lod}'

        with tf.name_scope(block_name) as scope:
            x = Input(self.D_input_shape(res))
            y = self.apply_bias(self.conv2d(x, fmaps=self.D_n_filters(res - 1), kernel_size=1, scope=scope), scope=scope)

        return tf.keras.Model(x, y, name=block_name)

    def from_rgb(self, x, res):
        return self.fromRGB_layers[res](x)

    def blur(self, x, scope=''):
        return blur_layer(x, scope=scope, config=self.config)

    def dense(self, x, units, gain=None, scope=''):
        if gain is None: gain = self.gain
        return dense_layer(x, units, gain=gain, config=self.config, scope=scope)

    def conv2d(self, x, fmaps, kernel_size=None, gain=None, fused_down=False, scope=''):
        if kernel_size is None: kernel_size = self.D_kernel_size
        if gain is None: gain = self.gain
        return conv2d_layer(
            x, fmaps, kernel_size=kernel_size, gain=gain,
            fused_down=fused_down, scope=scope, config=self.config
        )

    def downscale2d(self, x):
        return downscale2d_layer(x, factor=2, config=self.config)

    def conv2d_downscale2d(self, x, fmaps, scope=''):
        if self.D_fused_scale:
            x = self.conv2d(x, fmaps=fmaps, fused_down=True, scope=scope)
        else:
            x = self.conv2d(x, fmaps=fmaps, scope=scope)
            x = self.downscale2d(x)
        return x

    def act(self, x, scope=''):
        return act_layer(x, self.D_act_name, scope=scope, config=self.config)

    def apply_bias(self, x, scope=''):
        return bias_layer(x, scope=scope, config=self.config) if self.use_bias else x

    def D_block(self, x, res):
        with tf.name_scope(f'{2**res}x{2**res}') as top_scope:
            if res >= 3: # 8x8 and up
                with tf.name_scope('Conv') as scope:
                    x = self.act(self.apply_bias(self.conv2d(x, fmaps=self.D_n_filters(res - 1), scope=scope), scope), scope)
                with tf.name_scope('Conv1_down') as scope:
                    x = self.conv2d_downscale2d(self.blur(x, scope), fmaps=self.D_n_filters(res - 2), scope=scope)
                    x = self.act(self.apply_bias(x, scope), scope)
            else: # 4x4
                if self.mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, scope=top_scope, config=self.config)
                with tf.name_scope('Conv') as scope:
                    x = self.act(self.apply_bias(self.conv2d(x, fmaps=self.D_n_filters(res - 1), scope=scope), scope), scope)
                with tf.name_scope('Dense0') as scope:
                    x = self.act(self.apply_bias(self.dense(x, units=self.D_n_filters(res - 2), scope=scope), scope), scope)
                with tf.name_scope('Dense1') as scope:
                    x = self.apply_bias(self.dense(x, units=1, gain=1., scope=scope), scope)
            return x

    def create_D_model(self, model_res, mode=STABILIZATION_MODE):
        # Note: model_res is should be log2(image_res)
        # TODO: for debugging, remove later
        print(f'Calling create D model: res={model_res}')
        assert mode in [TRANSITION_MODE, STABILIZATION_MODE], 'Mode ' + mode + ' is not supported'
        model_type_key = create_model_type_key(model_res, mode)
        if model_type_key not in self.D_models.keys():
            if model_res >= 3:
                inputs = self.D_input_layers[model_res]
                if mode == STABILIZATION_MODE:
                    x = self.from_rgb(inputs, model_res)
                    x = self.D_block(x, model_res)
                elif mode == TRANSITION_MODE:
                    # Last input layers
                    x1 = self.downscale2d(inputs)
                    x1 = self.from_rgb(x1, model_res - 1)
                    # Introduce new layers
                    x2 = self.from_rgb(inputs, model_res)
                    x2 = self.D_block(x2, model_res)
                    lod = level_of_details(model_res, self.resolution_log2)
                    wsum_name = f'D_{WSUM_NAME}_lod{lod}'
                    x = WeightedSum(dtype=self.policy, name=wsum_name)([x1, x2])

                for res in range(model_res - 1, 2 - 1, -1):
                    x = self.D_block(x, res)
            else:
                inputs = self.D_input_layers[2]
                x = self.from_rgb(inputs, 2)
                x = self.D_block(x, 2)

            self.D_models[model_type_key] = tf.keras.Model(
                inputs, tf.identity(x, name='scores_out'), name='D_style'
            )

        return self.D_models[model_type_key]

    def initialize_D_model(self, model_res=None, mode=None):
        if model_res is not None:
            batch_size = self.batch_sizes[2 ** model_res]
            images = tf.zeros(
                shape=(batch_size,) + self.D_input_shape(model_res), dtype=self.compute_dtype
            )
            D_model = self.create_D_model(model_res, mode=mode)
            _ = D_model(images)
        else:
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
                batch_size = self.batch_sizes[2 ** res]
                images = tf.zeros(
                    shape=(batch_size,) + self.D_input_shape(res), dtype=self.compute_dtype
                )
                D_model = self.create_D_model(res, mode=TRANSITION_MODE)
                _ = D_model(images)

        print('D model built')

    def save_D_weights_in_class(self, D_model):
        self.D_weights_dict = weights_to_dict(D_model)

    def load_D_weights_from_class(self, D_model):
        return load_model_weights_from_dict(D_model, self.D_weights_dict)

    def trace_D_graphs(self, summary_writers, writers_dirs):
        D_prefix = 'Discriminator_'
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            writer = summary_writers[res]
            # TODO: Change it later!
            profiler_dir = writers_dirs[res]
            trace_D_input = tf.zeros((1,) + self.D_input_shape(res), dtype=self.compute_dtype)

            if res == self.start_resolution_log2:
                trace_D_model = tf.function(self.create_D_model(res, mode=STABILIZATION_MODE))
                with writer.as_default():
                    # Transition model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_D_model(trace_D_input)
                    tf.summary.trace_export(
                        name=D_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()
                    writer.flush()
            else:
                trace_D_model1 = tf.function(self.create_D_model(res, mode=TRANSITION_MODE))
                trace_D_model2 = tf.function(self.create_D_model(res, mode=STABILIZATION_MODE))
                with writer.as_default():
                    # Fade-in model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_D_model1(trace_D_input)
                    tf.summary.trace_export(
                        name=D_prefix + TRANSITION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()
                    # Stabilization model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_D_model2(trace_D_input)
                    tf.summary.trace_export(
                        name=D_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()
                    writer.flush()

        print('All Discriminator models traced!')
