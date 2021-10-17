import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from config import Config as cfg
from custom_layers import WeightedSum, layer_dtype,\
    dense_layer, conv2d_layer, fused_bias_act_layer, bias_act_layer, const_layer, noise_layer, blur_layer,\
    pixel_norm_layer, instance_norm_layer, style_mod_layer, downscale2d_layer, upscale2d_layer, minibatch_stddev_layer
from checkpoint_utils import weights_to_dict, load_model_weights_from_dict
from utils import level_of_details, validate_data_format, create_model_type_key, to_int_dict,\
    get_start_fp16_resolution, should_use_fp16, adjust_clamp,\
    NHWC_FORMAT, NCHW_FORMAT, TRANSITION_MODE, STABILIZATION_MODE
from tf_utils import generate_latents, get_compute_dtype, lerp,\
    DEFAULT_DATA_FORMAT, WSUM_NAME, GAIN_INIT_MODE_DICT, GAIN_ACTIVATION_FUNS_DICT


def n_filters(stage, fmap_base, fmap_decay, fmap_max):
    """
    fmap_base  Overall multiplier for the number of feature maps.
    fmap_decay log2 feature map reduction when doubling the resolution.
    fmap_max   Maximum number of feature maps in any layer.
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class ModelConfig:

    def __init__(self, config):
        self.config = config

        self.target_resolution = config[cfg.TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(cfg.START_RESOLUTION, cfg.DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        # Computations
        self.fused_bias_act             = config.get(cfg.FUSED_BIAS_ACT, cfg.DEFAULT_FUSED_BIAS_ACT)
        self.use_mixed_precision        = config.get(cfg.USE_MIXED_PRECISION, cfg.DEFAULT_USE_MIXED_PRECISION)
        self.num_fp16_resolutions       = config.get(cfg.NUM_FP16_RESOLUTIONS, cfg.DEFAULT_NUM_FP16_RESOLUTIONS)
        self.start_fp16_resolution_log2 =\
            get_start_fp16_resolution(self.num_fp16_resolutions, self.start_resolution_log2, self.resolution_log2)
        self.model_compute_dtype        = get_compute_dtype(self.use_mixed_precision)
        self.use_xla                    = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
        self.conv_clamp                 = config.get(cfg.CONV_CLAMP, cfg.DEFAULT_CONV_CLAMP)

        self.batch_sizes = to_int_dict(config[cfg.BATCH_SIZES])
        self.final_batch_size = config.get(cfg.FINAL_BATCH_SIZE, cfg.DEFAULT_FINAL_BATCH_SIZE)
        # If not provided, take value from common dict
        if self.final_batch_size is None:
            self.final_batch_size = self.batch_sizes[self.target_resolution]


class GeneratorMapping(ModelConfig):

    def __init__(self, config):
        super(GeneratorMapping, self).__init__(config)

        self.latent_size       = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.dlatent_size      = config.get(cfg.DLATENT_SIZE, cfg.DEFAULT_DLATENT_SIZE)
        self.normalize_latents = config.get(cfg.NORMALIZE_LATENTS, cfg.DEFAULT_NORMALIZE_LATENTS)
        self.use_styles        = config.get(cfg.USE_STYLES, cfg.DEFAULT_USE_STYLES)
        self.mapping_layers    = config.get(cfg.MAPPING_LAYERS, cfg.DEFAULT_MAPPING_LAYERS)
        self.mapping_units     = config.get(cfg.MAPPING_UNITS, cfg.DEFAULT_MAPPING_UNITS)
        self.mapping_lrmul     = config.get(cfg.MAPPING_LRMUL, cfg.DEFAULT_MAPPING_LRMUL)
        self.mapping_act_name  = config.get(cfg.MAPPING_ACTIVATION, cfg.DEFAULT_MAPPING_ACTIVATION)
        self.mapping_gain      = GAIN_ACTIVATION_FUNS_DICT[self.mapping_act_name]
        self.mapping_use_bias  = config.get(cfg.MAPPING_USE_BIAS, cfg.DEFAULT_MAPPING_USE_BIAS)

        # Note: now mapping network is built for target resolution, so when current resolution is lower that that,
        # some broadcast outputs aren't connected to any part of the graph. It's fine, but may look strange.
        self.num_layers = self.resolution_log2 * 2 - 2
        # self.num_layers = self.model_res * 2 - 2
        self.num_styles = self.num_layers if self.use_styles else 1

    def dense(self, x, units, use_fp16=None, scope=''):
        return dense_layer(x, units, lrmul=self.mapping_lrmul, gain=self.mapping_gain, use_fp16=use_fp16, scope=scope, config=self.config)

    def bias_act(self, x, use_fp16=None, scope=''):
        kwargs = {
            'x': x,
            'act_name': self.mapping_act_name,
            'use_bias': self.mapping_use_bias,
            'lrmul': self.mapping_lrmul,
            'use_fp16': use_fp16,
            'scope': scope,
            'config': self.config
        }
        return fused_bias_act_layer(**kwargs) if self.fused_bias_act else bias_act_layer(**kwargs)

    def create_G_mapping(self):
        # TODO: think about fp16 for this network
        use_fp16 = self.use_mixed_precision

        self.latents = Input([self.latent_size], dtype=self.model_compute_dtype, name='Latents')
        x = self.latents
        if self.normalize_latents:
            #with tf.name_scope('Latents_normalizer') as scope:
            x = pixel_norm_layer(x, use_fp16=use_fp16, config=self.config)

        with tf.name_scope('G_mapping'):
            for layer_idx in range(self.mapping_layers):
                with tf.name_scope(f'Dense{layer_idx}') as scope:
                    units = self.dlatent_size if layer_idx == self.mapping_layers - 1 else self.mapping_units
                    x = self.dense(x, units, use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, use_fp16=use_fp16, scope=scope)

        with tf.name_scope('Broadcast'):
            x = tf.tile(x[:, tf.newaxis], [1, self.num_styles, 1])

        self.G_mapping = tf.keras.Model(self.latents, tf.identity(x, name='dlatents'), name='G_mapping')


class GeneratorStyle(tf.keras.Model, ModelConfig):

    def __init__(self, G_mapping, G_synthesis, model_res, config):
        ModelConfig.__init__(self, config)
        tf.keras.Model.__init__(self, name='G_style')

        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.model_res = model_res

        self.latent_size         = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.dlatent_size        = config.get(cfg.DLATENT_SIZE, cfg.DEFAULT_DLATENT_SIZE)
        self.randomize_noise     = config.get(cfg.RANDOMIZE_NOISE, cfg.DEFAULT_RANDOMIZE_NOISE)
        self.truncation_psi      = config.get(cfg.TRUNCATION_PSI, cfg.DEFAULT_TRUNCATION_PSI)
        self.truncation_cutoff   = config.get(cfg.TRUNCATION_CUTOFF, cfg.DEFAULT_TRUNCATION_CUTOFF)
        self.dlatent_avg_beta    = config.get(cfg.DLATENT_AVG_BETA, cfg.DEFAULT_DLATENT_AVG_BETA)
        self.style_mixing_prob   = config.get(cfg.STYLE_MIXING_PROB, cfg.DEFAULT_STYLE_MIXING_PROB)

        self.num_layers = self.resolution_log2 * 2 - 2
        self.res_num_layers = self.model_res * 2 - 2

        self.validate_call_params()
        self.initialize_variables()

    def validate_call_params(self):
        def validate_range(value, min_val, max_val):
            if value is not None:
                assert min_val <= value <= max_val
        validate_range(self.truncation_psi, 0.0, 1.0)
        # validate_range(self.truncation_cutoff, 0, self.num_layers)
        validate_range(self.dlatent_avg_beta, 0.0, 1.0)
        validate_range(self.style_mixing_prob, 0.0, 1.0)

    def initialize_variables(self):
        self.dlatent_avg = self.add_weight(
            name='dlatent_avg',
            shape=[self.dlatent_size],
            dtype=self.model_compute_dtype,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        # Prepare value for style mixing and truncation
        self.layer_idx = tf.range(self.num_layers)[tf.newaxis, :, tf.newaxis]

        if self.style_mixing_prob is not None:
            self.mixing_cur_layers = self.res_num_layers

        if (self.truncation_psi is not None) and (self.truncation_cutoff is not None):
            ones = tf.ones(self.layer_idx.shape, dtype=self.model_compute_dtype)
            self.truncation_coefs = tf.where(self.layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)

    def update_dlatent_avg(self, dlatents):
        batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
        self.dlatent_avg.assign(
            lerp(batch_avg, self.dlatent_avg, self.dlatent_avg_beta)
        )

    def generate_latents(self, batch_size):
        return generate_latents(batch_size, self.latent_size, self.model_compute_dtype)

    def apply_style_mixing(self, dlatents):
        latents2 = self.generate_latents(tf.shape(dlatents)[0])
        # Styles can only be mixed during training
        dlatents2 = self.G_mapping(latents2, training=True)
        mixing_cutoff = tf.cond(
            tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob,
            lambda: tf.random.uniform([], 1, self.mixing_cur_layers, dtype=tf.int32),
            lambda: self.mixing_cur_layers
        )
        dlatents = tf.where(tf.broadcast_to(self.layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
        return dlatents

    def apply_truncation_trick(self, dlatents):
        return lerp(self.dlatent_avg, dlatents, self.truncation_coefs)

    def call(self, latents, training=True, validation=False, truncation_psi=None, truncation_cutoff=None, *args, **kwargs):
        # 1. Decide which actions to perform based on training/testing/validation.
        # Validation is used for metrics evaluation. Testing for generation of images
        assert not (training and validation), "Model can't use training and validation modes at the same time"
        if training or validation:
            truncation_psi = None
            truncation_cutoff = None
        else:
            truncation_psi = self.truncation_psi if truncation_psi is None else truncation_psi
            truncation_cutoff = self.truncation_cutoff if truncation_cutoff is None else truncation_cutoff

        should_update_dlatent_avg = (self.dlatent_avg_beta is not None) and training
        should_apply_style_mixing = (self.style_mixing_prob is not None) and training
        should_apply_truncation_trick = (truncation_psi is not None) and (truncation_cutoff is not None)

        # 2. Evaluate dlatents, output shape: (batch, num_layers, dlatent_size)
        dlatents = self.G_mapping(latents, training=training)

        # 3. Update moving average of W
        with tf.name_scope('DlatentAvg'):
            if should_update_dlatent_avg:
                self.update_dlatent_avg(dlatents)

        # 4. Perform mixing style regularization
        with tf.name_scope('StyleMixing'):
            if should_apply_style_mixing:
                dlatents = self.apply_style_mixing(dlatents)

        # 5. Apply truncation trick
        with tf.name_scope('Truncation'):
            if should_apply_truncation_trick:
                dlatents = self.apply_truncation_trick(dlatents)

        # 6. Evaluate synthesis network
        images_out = self.G_synthesis(dlatents, training=training)
        return images_out


class Generator(ModelConfig):

    def __init__(self, config):
        super(Generator, self).__init__(config)

        self.latent_size       = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.dlatent_size      = config.get(cfg.DLATENT_SIZE, cfg.DEFAULT_DLATENT_SIZE)
        self.normalize_latents = config.get(cfg.NORMALIZE_LATENTS, cfg.DEFAULT_NORMALIZE_LATENTS)
        self.const_input_layer = config.get(cfg.CONST_INPUT_LAYER, cfg.DEFAULT_CONST_INPUT_LAYER)
        self.use_noise         = config.get(cfg.USE_NOISE, cfg.DEFAULT_USE_NOISE)
        self.randomize_noise   = config.get(cfg.RANDOMIZE_NOISE, cfg.DEFAULT_RANDOMIZE_NOISE)
        self.use_bias          = config.get(cfg.USE_BIAS, cfg.DEFAULT_USE_BIAS)
        self.use_pixel_norm    = config.get(cfg.USE_PIXEL_NORM, cfg.DEFAULT_USE_PIXEL_NORM)
        self.use_instance_norm = config.get(cfg.USE_INSTANCE_NORM, cfg.DEFAULT_USE_INSTANCE_NORM)
        self.use_styles        = config.get(cfg.USE_STYLES, cfg.DEFAULT_USE_STYLES)
        self.G_fused_scale     = config.get(cfg.G_FUSED_SCALE, cfg.DEFAULT_G_FUSED_SCALE)
        self.G_kernel_size     = config.get(cfg.G_KERNEL_SIZE, cfg.DEFAULT_G_KERNEL_SIZE)
        self.G_fmap_base       = config.get(cfg.G_FMAP_BASE, cfg.DEFAULT_FMAP_BASE)
        self.G_fmap_decay      = config.get(cfg.G_FMAP_DECAY, cfg.DEFAULT_FMAP_DECAY)
        self.G_fmap_max        = config.get(cfg.G_FMAP_MAX, cfg.DEFAULT_FMAP_MAX)
        self.G_act_name        = config.get(cfg.G_ACTIVATION, cfg.DEFAULT_G_ACTIVATION)
        self.blur_filter       = config.get(cfg.BLUR_FILTER, cfg.DEFAULT_BLUR_FILTER)

        self.G_weights_init_mode = config.get(cfg.G_WEIGHTS_INIT_MODE, None)
        if self.G_weights_init_mode is None:
            self.gain = GAIN_ACTIVATION_FUNS_DICT[self.G_act_name]
        else:
            self.gain = GAIN_INIT_MODE_DICT[self.G_weights_init_mode]

        self.override_projecting_gain = config.get(cfg.OVERRIDE_G_PROJECTING_GAIN, cfg.DEFAULT_OVERRIDE_G_PROJECTING_GAIN)
        # Gain is overridden to match the original ProGAN implementation (sqrt(2) / 4 was used with He init)
        self.projecting_gain_correction = 4. if self.override_projecting_gain else 1.
        self.projecting_gain = self.gain / self.projecting_gain_correction

        # This constant is taken from the original implementation
        self.projecting_mult = 4
        if self.data_format == NCHW_FORMAT:
            self.projecting_target_shape = [-1, self.G_n_filters(1), self.projecting_mult, self.projecting_mult]
        else: # self.data_format == NHWC_FORMAT:
            self.projecting_target_shape = [-1, self.projecting_mult, self.projecting_mult, self.G_n_filters(1)]
        self.projecting_units = np.prod(self.projecting_target_shape)

        self.num_layers = self.resolution_log2 * 2 - 2

        self.create_model_layers()
        self.G_models = {}

    def G_n_filters(self, stage):
        return n_filters(stage, self.G_fmap_base, self.G_fmap_decay, self.G_fmap_max)

    def G_output_shape(self, res):
        nf = self.G_n_filters(res - 1)
        if self.data_format == NCHW_FORMAT:
            return [nf, 2 ** res, 2 ** res]
        else: # self.data_format == NHWC_FORMAT:
            return [2 ** res, 2 ** res, nf]

    def create_model_layers(self):
        # TODO: change start resolution (problems with graphs tracing)
        self.toRGB_layers = {
            res: self.to_rgb_layer(res) for res in range(self.start_resolution_log2 - 1, self.resolution_log2 + 1)
        }
        self.G_mapping_object = GeneratorMapping(self.config)
        self.G_mapping_object.create_G_mapping()
        self.latents = self.G_mapping_object.latents
        self.G_mapping = self.G_mapping_object.G_mapping
        # Use mapping network to get shape of dlatents
        self.dlatents = Input(self.G_mapping.output_shape[1:], dtype=self.model_compute_dtype, name='Dlatents')

    def to_rgb_layer(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'ToRGB_lod{lod}'
        use_fp16 = should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)

        with tf.name_scope(block_name) as scope:
            x = Input(self.G_output_shape(res))
            y = self.conv2d(x, fmaps=3, kernel_size=1, gain=1., use_fp16=use_fp16, scope=scope)
            y = self.bias_act(y, act_name='linear', clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)

        return tf.keras.Model(x, y, name=block_name)

    def to_rgb(self, x, res):
        return self.toRGB_layers[res](x)

    def blur(self, x, use_fp16=None, scope=''):
        return blur_layer(x, use_fp16=use_fp16, scope=scope, config=self.config) if self.blur_filter is not None else x

    def dense(self, x, units, gain=None, use_fp16=None, scope=''):
        if gain is None: gain = self.gain
        return dense_layer(x, units, gain=gain, use_fp16=use_fp16, scope=scope, config=self.config)

    def conv2d(self, x, fmaps, kernel_size=None, gain=None, fused_up=False, use_fp16=None, scope=''):
        if kernel_size is None: kernel_size = self.G_kernel_size
        if gain is None: gain = self.gain
        return conv2d_layer(
            x, fmaps, kernel_size=kernel_size, gain=gain, fused_up=fused_up,
            use_fp16=use_fp16, scope=scope, config=self.config
        )

    def upscale2d(self, x, use_fp16=None):
        return upscale2d_layer(x, factor=2, use_fp16=use_fp16, config=self.config)

    def upscale2d_conv2d(self, x, fmaps, use_fp16=None, scope=''):
        if self.G_fused_scale:
            x = self.conv2d(x, fmaps=fmaps, fused_up=True, use_fp16=use_fp16, scope=scope)
        else:
            x = self.upscale2d(x, use_fp16=use_fp16)
            x = self.conv2d(x, fmaps=fmaps, use_fp16=use_fp16, scope=scope)
        return x

    def bias_act(self, x, act_name=None, clamp=None, use_fp16=None, scope=''):
        if act_name is None: act_name = self.G_act_name
        clamp = adjust_clamp(clamp, use_fp16)
        kwargs = {
            'x': x,
            'act_name': act_name,
            'use_bias': self.use_bias,
            'clamp': clamp,
            'use_fp16': use_fp16,
            'scope': scope,
            'config': self.config
        }
        return fused_bias_act_layer(**kwargs) if self.fused_bias_act else bias_act_layer(**kwargs)

    def layer_epilogue(self, x, layer_idx, use_fp16=None, scope=''):
        if self.use_noise:
            x = noise_layer(x, use_fp16=use_fp16, scope=scope, config=self.config)
        x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
        if self.use_pixel_norm:
            x = pixel_norm_layer(x, use_fp16=use_fp16, scope=scope, config=self.config)
        if self.use_instance_norm:
            x = instance_norm_layer(x, use_fp16=use_fp16, scope=scope, config=self.config)
        if self.use_styles:
            x = style_mod_layer(x, self.dlatents[:, layer_idx], use_fp16=use_fp16, scope=scope, config=self.config)
        return x

    def input_block(self):
        use_fp16 = should_use_fp16(2, self.start_fp16_resolution_log2, self.use_mixed_precision)
        with tf.name_scope('4x4'):
            if self.const_input_layer:
                with tf.name_scope('Const') as scope:
                    # Note: input to layer should be dlatents to allow having separate mapping and synthesis networks,
                    # though number of channels is taken according to latents
                    x = const_layer(self.dlatents[:, 0], self.latent_size, use_fp16=use_fp16, scope=scope, config=self.config)
                    x = self.layer_epilogue(x, 0, use_fp16=use_fp16, scope=scope)
            else:
                with tf.name_scope('Dense') as scope:
                    x = self.dense(
                        self.dlatents[:, 0], units=self.projecting_units, gain=self.projecting_gain,
                        use_fp16=use_fp16, scope=scope
                    )
                    x = self.layer_epilogue(tf.reshape(x, self.projecting_target_shape), 0, use_fp16=use_fp16, scope=scope)
            with tf.name_scope('Conv1') as scope:
                x = self.layer_epilogue(self.conv2d(x, fmaps=self.G_n_filters(1), use_fp16=use_fp16, scope=scope), 1, use_fp16=use_fp16, scope=scope)
        return x

    def G_block(self, x, res):
        # res = 3 ... resolution_log2
        use_fp16 = should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)
        with tf.name_scope(f'{2**res}x{2**res}'):
            with tf.name_scope('Conv0_up') as scope:
                x = self.blur(self.upscale2d_conv2d(x, fmaps=self.G_n_filters(res - 1), use_fp16=use_fp16, scope=scope), use_fp16, scope)
                x = self.layer_epilogue(x, res * 2 - 4, use_fp16=use_fp16, scope=scope)
            with tf.name_scope('Conv1') as scope:
                x = self.conv2d(x, fmaps=self.G_n_filters(res - 1), use_fp16=use_fp16, scope=scope)
                x = self.layer_epilogue(x, res * 2 - 3, use_fp16=use_fp16, scope=scope)
        return x

    def create_G_model(self, model_res, mode=STABILIZATION_MODE):
        # Note: model_res is should be log2(image_res)
        assert mode in [TRANSITION_MODE, STABILIZATION_MODE], 'Mode ' + mode + ' is not supported'
        model_type_key = create_model_type_key(model_res, mode)
        if model_type_key not in self.G_models.keys():
            print(f' ...Creating G model for res = {model_res} and mode = {mode}...')
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
                    use_fp16 = should_use_fp16(model_res, self.start_fp16_resolution_log2, self.use_mixed_precision)
                    images1 = self.to_rgb(x, model_res - 1)
                    images1 = self.upscale2d(images1, use_fp16=use_fp16)
                    # Introduce new layers
                    images2 = self.G_block(x, model_res)
                    images2 = self.to_rgb(images2, model_res)
                    # Merge images
                    lod = level_of_details(model_res, self.resolution_log2)
                    wsum_name = f'G_{WSUM_NAME}_lod{lod}'
                    wsum_dtype = layer_dtype(WSUM_NAME, use_fp16)
                    images_out = WeightedSum(dtype=wsum_dtype, name=wsum_name)([images1, images2])

            self.G_synthesis = tf.keras.Model(
                self.dlatents, tf.identity(images_out, name='images_out'), name='G_synthesis'
            )
            # Create full G model
            self.G_models[model_type_key] = GeneratorStyle(self.G_mapping, self.G_synthesis, model_res, self.config)
        else:
            print(f' ...Taking G model for res = {model_res} and mode = {mode} from cache...')

        return self.G_models[model_type_key]

    def initialize_G_model(self, model_res=None, mode=None):
        if model_res is not None:
            batch_size = self.batch_sizes[2 ** model_res]
            latents = tf.zeros(shape=[batch_size, self.dlatent_size], dtype=self.model_compute_dtype)
            G_model = self.create_G_model(model_res, mode=mode)
            _ = G_model(latents, training=False)
        else:
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
                batch_size = self.batch_sizes[2 ** res]
                latents = tf.zeros(shape=[batch_size, self.dlatent_size], dtype=self.model_compute_dtype)
                G_model = self.create_G_model(res, mode=TRANSITION_MODE)
                _ = G_model(latents, training=False)

        print(f' ...Built G model for res = {model_res} and mode = {mode}...')

    def save_G_weights_in_class(self, G_model):
        self.G_weights_dict = weights_to_dict(G_model)

    def load_G_weights_from_class(self, G_model):
        return load_model_weights_from_dict(G_model, self.G_weights_dict)

    def trace_G_graphs(self, summary_writers, writers_dirs):
        G_prefix = 'Generator_'
        trace_G_input = tf.zeros(shape=[1, self.dlatent_size], dtype=self.model_compute_dtype)
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


class Discriminator(ModelConfig):

    def __init__(self, config):
        super(Discriminator, self).__init__(config)

        self.use_bias         = config.get(cfg.USE_BIAS, cfg.DEFAULT_USE_BIAS)
        self.mbstd_group_size = config[cfg.MBSTD_GROUP_SIZE]
        self.D_fused_scale    = config.get(cfg.D_FUSED_SCALE, cfg.DEFAULT_D_FUSED_SCALE)
        self.D_kernel_size    = config.get(cfg.D_KERNEL_SIZE, cfg.DEFAULT_D_KERNEL_SIZE)
        self.D_fmap_base      = config.get(cfg.D_FMAP_BASE, cfg.DEFAULT_FMAP_BASE)
        self.D_fmap_decay     = config.get(cfg.D_FMAP_DECAY, cfg.DEFAULT_FMAP_DECAY)
        self.D_fmap_max       = config.get(cfg.D_FMAP_MAX, cfg.DEFAULT_FMAP_MAX)
        self.D_act_name       = config.get(cfg.D_ACTIVATION, cfg.DEFAULT_D_ACTIVATION)
        self.blur_filter      = config.get(cfg.BLUR_FILTER, cfg.DEFAULT_BLUR_FILTER)

        self.D_weights_init_mode = config.get(cfg.D_WEIGHTS_INIT_MODE, None)
        if self.D_weights_init_mode is None:
            self.gain = GAIN_ACTIVATION_FUNS_DICT[self.D_act_name]
        else:
            self.gain = GAIN_INIT_MODE_DICT[self.D_weights_init_mode]

        # Might be useful to override number of units in projecting layer in case latent size is not 512
        # to make models have almost the same number of trainable params
        self.projecting_nf = config.get(cfg.D_PROJECTING_NF, self.D_n_filters(2 - 2))

        self.create_model_layers()
        self.D_models = {}

    def D_n_filters(self, stage):
        return n_filters(stage, self.D_fmap_base, self.D_fmap_decay, self.D_fmap_max)

    def D_input_shape(self, res):
        if self.data_format == NCHW_FORMAT:
            return [3, 2 ** res, 2 ** res]
        else: # self.data_format == NHWC_FORMAT:
            return [2 ** res, 2 ** res, 3]

    def create_model_layers(self):
        self.D_input_layers = {
            res: Input(
                shape=self.D_input_shape(res), dtype=self.model_compute_dtype, name=f'Images_{2**res}x{2**res}'
            ) for res in range(2, self.resolution_log2 + 1)
        }
        # TODO: change start resolution (problems with graphs tracing)
        self.fromRGB_layers = {
            res: self.from_rgb_layer(res) for res in range(self.start_resolution_log2 - 1, self.resolution_log2 + 1)
        }

    def from_rgb_layer(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'FromRGB_lod{lod}'
        use_fp16 = should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)

        with tf.name_scope(block_name) as scope:
            x = Input(self.D_input_shape(res))
            y = self.conv2d(x, fmaps=self.D_n_filters(res - 1), kernel_size=1, use_fp16=use_fp16, scope=scope)
            y = self.bias_act(y, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)

        return tf.keras.Model(x, y, name=block_name)

    def from_rgb(self, x, res):
        return self.fromRGB_layers[res](x)

    def blur(self, x, use_fp16=None, scope=''):
        return blur_layer(x, use_fp16=use_fp16, scope=scope, config=self.config) if self.blur_filter is not None else x

    def dense(self, x, units, gain=None, use_fp16=None, scope=''):
        if gain is None: gain = self.gain
        return dense_layer(x, units, gain=gain, use_fp16=use_fp16, scope=scope, config=self.config)

    def conv2d(self, x, fmaps, kernel_size=None, gain=None, fused_down=False, use_fp16=None, scope=''):
        if kernel_size is None: kernel_size = self.D_kernel_size
        if gain is None: gain = self.gain
        return conv2d_layer(
            x, fmaps, kernel_size=kernel_size, gain=gain, fused_down=fused_down,
            use_fp16=use_fp16, scope=scope, config=self.config
        )

    def downscale2d(self, x, use_fp16=None):
        return downscale2d_layer(x, factor=2, use_fp16=use_fp16, config=self.config)

    def conv2d_downscale2d(self, x, fmaps, use_fp16=None, scope=''):
        if self.D_fused_scale:
            x = self.conv2d(x, fmaps=fmaps, fused_down=True, use_fp16=use_fp16, scope=scope)
        else:
            x = self.conv2d(x, fmaps=fmaps, use_fp16=use_fp16, scope=scope)
            x = self.downscale2d(x, use_fp16=use_fp16)
        return x

    def bias_act(self, x, act_name=None, clamp=None, use_fp16=None, scope=''):
        if act_name is None: act_name = self.D_act_name
        clamp = adjust_clamp(clamp, use_fp16)
        kwargs = {
            'x': x,
            'act_name': act_name,
            'use_bias': self.use_bias,
            'clamp': clamp,
            'use_fp16': use_fp16,
            'scope': scope,
            'config': self.config
        }
        return fused_bias_act_layer(**kwargs) if self.fused_bias_act else bias_act_layer(**kwargs)

    def D_block(self, x, res):
        use_fp16 = should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)
        with tf.name_scope(f'{2**res}x{2**res}') as top_scope:
            if res >= 3: # 8x8 and up
                with tf.name_scope('Conv') as scope:
                    x = self.conv2d(x, fmaps=self.D_n_filters(res - 1), use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
                with tf.name_scope('Conv1_down') as scope:
                    x = self.conv2d_downscale2d(self.blur(x, use_fp16, scope), fmaps=self.D_n_filters(res - 2), use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
            else: # 4x4
                if self.mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, use_fp16, scope=top_scope, config=self.config)
                with tf.name_scope('Conv') as scope:
                    x = self.conv2d(x, fmaps=self.D_n_filters(res - 1), use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
                with tf.name_scope('Dense0') as scope:
                    x = self.dense(x, units=self.D_n_filters(res - 2), use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, use_fp16=use_fp16, scope=scope)
                with tf.name_scope('Dense1') as scope:
                    x = self.dense(x, units=1, gain=1., use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, act_name='linear', use_fp16=use_fp16, scope=scope)
            return x

    def create_D_model(self, model_res, mode=STABILIZATION_MODE):
        # Note: model_res is should be log2(image_res)
        assert mode in [TRANSITION_MODE, STABILIZATION_MODE], 'Mode ' + mode + ' is not supported'
        model_type_key = create_model_type_key(model_res, mode)
        if model_type_key not in self.D_models.keys():
            print(f' ...Creating D model for res = {model_res} and mode = {mode}...')
            if model_res >= 3:
                inputs = self.D_input_layers[model_res]
                if mode == STABILIZATION_MODE:
                    x = self.from_rgb(inputs, model_res)
                    x = self.D_block(x, model_res)
                elif mode == TRANSITION_MODE:
                    # Last input layers
                    use_fp16 = should_use_fp16(model_res - 1, self.start_fp16_resolution_log2, self.use_mixed_precision)
                    x1 = self.downscale2d(inputs, use_fp16=use_fp16)
                    x1 = self.from_rgb(x1, model_res - 1)
                    # Introduce new layers
                    x2 = self.from_rgb(inputs, model_res)
                    x2 = self.D_block(x2, model_res)
                    # Merge features
                    lod = level_of_details(model_res, self.resolution_log2)
                    wsum_name = f'D_{WSUM_NAME}_lod{lod}'
                    wsum_use_fp16 = should_use_fp16(model_res, self.start_fp16_resolution_log2, self.use_mixed_precision)
                    wsum_dtype = layer_dtype(WSUM_NAME, wsum_use_fp16)
                    x = WeightedSum(dtype=wsum_dtype, name=wsum_name)([x1, x2])

                for res in range(model_res - 1, 2 - 1, -1):
                    x = self.D_block(x, res)
            else:
                inputs = self.D_input_layers[2]
                x = self.from_rgb(inputs, 2)
                x = self.D_block(x, 2)

            self.D_models[model_type_key] = tf.keras.Model(
                inputs, tf.identity(x, name='scores_out'), name='D_style'
            )
        else:
            print(f' ...Taking D model for res = {model_res} and mode = {mode} from cache...')

        return self.D_models[model_type_key]

    def initialize_D_model(self, model_res=None, mode=None):
        if model_res is not None:
            batch_size = self.batch_sizes[2 ** model_res]
            images = tf.zeros(shape=[batch_size] + self.D_input_shape(model_res), dtype=self.model_compute_dtype)
            D_model = self.create_D_model(model_res, mode=mode)
            _ = D_model(images, training=False)
        else:
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
                batch_size = self.batch_sizes[2 ** res]
                images = tf.zeros(shape=[batch_size] + self.D_input_shape(res), dtype=self.model_compute_dtype)
                D_model = self.create_D_model(res, mode=TRANSITION_MODE)
                _ = D_model(images, training=False)

        print(f' ...Built D model for res = {model_res} and mode = {mode}...')

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
            trace_D_input = tf.zeros([1] + self.D_input_shape(res), dtype=self.model_compute_dtype)
            if res == self.start_resolution_log2:
                trace_D_model = tf.function(self.create_D_model(res, mode=STABILIZATION_MODE))
                with writer.as_default():
                    # Stabilization model
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
                    # Transition model
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
