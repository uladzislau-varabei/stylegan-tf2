import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation
from tensorflow.keras import mixed_precision

from utils import DEFAULT_DTYPE, DEFAULT_DATA_FORMAT, NCHW_FORMAT, NHWC_FORMAT,\
    validate_data_format, HE_GAIN, DEFAULT_USE_WSCALE, DEFAULT_TRUNCATE_WEIGHTS, DEFAULT_USE_XLA

# Config constants
from utils import ACTIVATION_FUNS_DICT, FP32_ACTIVATIONS, USE_MIXED_PRECISION, USE_WSCALE, TRUNCATE_WEIGHTS,\
    DATA_FORMAT, RANDOMIZE_NOISE, BLUR_FILTER, USE_BIAS, MBSTD_GROUP_SIZE, MBSTD_NUM_FEATURES, \
    DEFAULT_USE_MIXED_PRECISION, DEFAULT_RANDOMIZE_NOISE, DEFAULT_BLUR_FILTER, DEFAULT_USE_BIAS,\
    DEFAULT_MBSTD_NUM_FEATURES


LRMUL = 1.

WEIGHTS_NAME = 'weights'
BIASES_NAME = 'biases'


#----------------------------------------------------------------------------
# Utils.

biases_init = tf.zeros_initializer()


def weights_init(std):
    return tf.random_normal_initializer(stddev=std, seed=42)


def truncated_weights_init(std):
    return tf.initializers.TruncatedNormal(stddev=std, seed=42)


def select_initializer(truncate_weights, std):
    if truncate_weights:
        return truncated_weights_init(std)
    else:
        return weights_init(std)


def weights_std(gain, fan_in):
    # He/LeCun init
    return gain / np.sqrt(fan_in)


# Equalized learning rate and custom learning rate multiplier
def weights_coeffs(use_wscale, std, lrmul):
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = std * lrmul
    else:
        init_std = std / lrmul
        runtime_coef = lrmul
    return init_std, runtime_coef


# Remove layer name from name scope
# To be used as a second level of scoping in build with
# ... tf.name_scope(clean_name_scope(scope)) as final_scope: ...
def clean_name_scope(name_scope):
    return name_scope.split('/', 1)[1] if '/' in name_scope else name_scope


# Make layer name to structure graph in tensorboard
def make_layer_name(input_name, input_scope, class_scope):
    return input_name if input_name is not None else input_scope + class_scope


#----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.

@tf.function
def _blur2d(x, f=None, normalize=True, flip=False, stride=1, data_format=DEFAULT_DATA_FORMAT):
    assert isinstance(stride, int) and stride >= 1
    validate_data_format(data_format)

    # Finalize filter kernel
    if f is None: f = [1, 2, 1]
    f = np.array(f, dtype=x.dtype.name)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    if normalize:
        f /= np.sum(f)
    assert f.ndim == 2
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]

    if data_format == NCHW_FORMAT:
        f = np.tile(f, [1, 1, int(x.shape[1]), 1])
        strides = [1, 1, stride, stride]
    elif data_format == NHWC_FORMAT:
        f = np.tile(f, [1, 1, int(x.shape[3]), 1])
        strides = [1, stride, stride, 1]

    # No op => early exit
    if f.shape == (1, 1) and abs(f[0, 0] - 1.) <= 1e-3:
        return x

    x = tf.nn.depthwise_conv2d(x, f, strides, padding='SAME', data_format=data_format)
    return x


@tf.function
def _upscale2d(x, factor, gain=1, data_format=DEFAULT_DATA_FORMAT):
    assert isinstance(factor, int) and factor >= 1
    validate_data_format(data_format)

    if gain != 1:
        x *= gain

    # No op => early exit
    if factor == 1:
        return x

    if data_format == NCHW_FORMAT:
        _, c, h, w = x.shape
        pretile_shape = [-1, c, h, 1, w, 1]
        tile_mults = [1, 1, 1, factor, 1, factor]
        output_shape = [-1, c, h * factor, w * factor]
    elif data_format == NHWC_FORMAT:
        _, h, w, c = x.shape
        pretile_shape = [-1, h, 1, w, 1, c]
        tile_mults = [1, 1, factor, 1, factor, 1]
        output_shape = [-1, h * factor, w * factor, c]

    x = tf.reshape(x, pretile_shape)
    x = tf.tile(x, tile_mults)
    x = tf.reshape(x, output_shape)
    return x


@tf.function
def _downscale2d(x, factor, gain=1, data_format=DEFAULT_DATA_FORMAT):
    assert isinstance(factor, int) and factor >= 1
    validate_data_format(data_format)

    if factor == 2 and x.dtype == tf.float32:
        f = [float(np.sqrt(gain) / factor)] * factor
        return _blur2d(x, f, normalize=False, stride=factor, data_format=data_format)

    if gain != 1:
        x *= gain

    # No op => early exit
    if factor == 1:
        return x

    if data_format == NCHW_FORMAT:
        ksize = [1, 1, factor, factor]
    elif data_format == NHWC_FORMAT:
        ksize = [1, factor, factor, 1]

    # Note: soft ops placements should be enabled
    return tf.nn.avg_pool2d(x, ksize, strides=ksize, padding='VALID', data_format=data_format)


#----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

@tf.function
def blur2d(x, filter=None, normalize=True, data_format=DEFAULT_DATA_FORMAT):
    if filter is None: filter = [1, 2, 1]
    @tf.custom_gradient
    def func(x):
        y = _blur2d(x, filter, normalize, data_format=data_format)
        @tf.custom_gradient
        def grad(dy):
            dx = _blur2d(dy, filter, normalize, flip=True, data_format=data_format)
            return dx, lambda ddx: _blur2d(ddx, filter, normalize, data_format=data_format)
        return y, grad
    return func(x)


@tf.function
def upscale2d(x, factor=2, data_format=DEFAULT_DATA_FORMAT):
    @tf.custom_gradient
    def func(x):
        y = _upscale2d(x, factor, data_format=data_format)
        @tf.custom_gradient
        def grad(dy):
            dx = _downscale2d(dy, factor,  gain=factor**2, data_format=data_format)
            return dx, lambda ddx: _upscale2d(ddx, factor,  data_format=data_format)
        return y, grad
    return func(x)


@tf.function
def downscale2d(x, factor=2, data_format=DEFAULT_DATA_FORMAT):
    @tf.custom_gradient
    def func(x):
        y = _downscale2d(x, factor, data_format=data_format)
        @tf.custom_gradient
        def grad(dy):
            dx = _upscale2d(dy, factor, gain=1/factor**2, data_format=data_format)
            return dx, lambda ddx: _downscale2d(ddx, factor, data_format=data_format)
        return y, grad
    return func(x)


#----------------------------------------------------------------------------
# Layers.

class ScaledConv2d(Layer):

    def __init__(self, fmaps, kernel_size, stride=1, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Conv2d')
        super(ScaledConv2d, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.stride = stride
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.scope = scope

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, self.stride, self.stride]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, self.stride, self.stride, 1]

        self.wshape = [self.kernel_size, self.kernel_size, self.channels_in, self.fmaps]
        self.fan_in = np.prod(self.wshape[:-1])
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=self.wshape,
                initializer=initializer,
                trainable=True
            )

    @tf.function
    def call(self, x, *args, **kwargs):
        return tf.nn.conv2d(
            x, self.runtime_coef * self.w, strides=self.strides, padding='SAME', data_format=self.data_format
        )


class ScaledLinear(Layer):

    def __init__(self, units, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Linear')
        super(ScaledLinear, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.units = units
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.scope = scope

    def build(self, input_shape):
        self.fan_in = np.prod(input_shape[1:])
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=[self.fan_in, self.units],
                initializer=initializer,
                trainable=True
            )

    @tf.function
    def call(self, x, *args, **kwargs):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, self.fan_in])
        return tf.linalg.matmul(x, self.runtime_coef * self.w)


class Bias(Layer):
    def __init__(self, lrmul=LRMUL, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Bias')
        super(Bias, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.lrmul = lrmul
        self.use_xla = use_xla
        self.scope = scope

    def build(self, input_shape):
        self.is_linear_bias = len(input_shape) == 2

        if self.is_linear_bias:
            self.units = input_shape[1]
        else:
            if self.data_format == NCHW_FORMAT:
                self.bias_target_shape = [1, -1, 1, 1]
                self.units = input_shape[1]
            elif self.data_format == NHWC_FORMAT:
                self.bias_target_shape = [1, 1, 1, -1]
                self.units = input_shape[-1]

        with tf.name_scope(self.scope):
            self.b = self.add_weight(
                name=BIASES_NAME,
                shape=[self.units],
                initializer=biases_init,
                trainable=True
            )

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.is_linear_bias:
            return x + self.lrmul * self.b
        else:
            # Note: keep reshaping to allow easy weights decay between cpu and gpu models
            return x + tf.reshape(self.lrmul * self.b, self.bias_target_shape)


class Blur2d(Layer):

    def __init__(self, filter, dtype=DEFAULT_DTYPE, data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Blur2d')
        super(Blur2d, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.filter = filter

    def call(self, x, *args, **kwargs):
        return blur2d(x, filter=self.filter, data_format=self.data_format)


class Upscale2d(Layer):

    def __init__(self, factor, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Upscale2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.factor = factor
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        return upscale2d(x, factor=self.factor, data_format=self.data_format)


class Downscale2d(Layer):

    def __init__(self, factor, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Downscale2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        if self.data_format == NCHW_FORMAT:
            self.ksize = [1, 1, factor, factor]
        elif self.data_format == NHWC_FORMAT:
            self.ksize = [1, factor, factor, 1]
        self.factor = factor
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        return downscale2d(x, factor=self.factor, data_format=self.data_format)


class PixelNorm(Layer):

    def __init__(self, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'PixelNorm')
        super(PixelNorm, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        if self.data_format == NCHW_FORMAT:
            self.channel_axis = 1
        elif self.data_format == NHWC_FORMAT:
            self.channel_axis = 3
        self.epsilon = 1e-8 if self._dtype_policy.compute_dtype == 'float32' else 1e-4
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=self.channel_axis, keepdims=True) + self.epsilon
        )


class InstanceNorm(Layer):

    def __init__(self, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'InstanceNorm')
        super(InstanceNorm, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        if self.data_format == NCHW_FORMAT:
            self.hw_axes = [2, 3]
        elif self.data_format == NHWC_FORMAT:
            self.hw_axes = [1, 2]
        # self.epsilon = 1e-8 if self._dtype_policy.compute_dtype == 'float32' else 1e-4
        # Epsilon can be constant as call inputs are casted to fp32
        self.epsilon = 1e-8
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=self.hw_axes, keepdims=True)
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=self.hw_axes, keepdims=True) + self.epsilon
        )


class StyleMod(Layer):

    def __init__(self, use_bias=True,
                 use_wscale=DEFAULT_USE_WSCALE,  truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'StyleMod')
        super(StyleMod, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.scope = scope + 'StyleMod/'

    def build(self, input_shape):
        x_shape, style_shape = input_shape

        if self.data_format == NCHW_FORMAT:
            self.fc_units = x_shape[1] * 2
            self.style_target_shape = [-1, 2, x_shape[1]] + [1] * (len(x_shape) - 2)
        elif self.data_format == NHWC_FORMAT:
            self.fc_units = x_shape[3] * 2
            self.style_target_shape = [-1, 2] + [1] * (len(x_shape) - 2) + [x_shape[3]]

        self.fc = ScaledLinear(
            units=self.fc_units, gain=1,
            use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
            dtype=self._dtype_policy, use_xla=self.use_xla,
            data_format=self.data_format, scope=self.scope
        )
        if self.use_bias:
            self.bias = Bias(
                dtype=self._dtype_policy, use_xla=self.use_xla, scope=self.scope, data_format=self.data_format
            )

    def apply_bias(self, x):
        return self.bias(x) if self.use_bias else x

    @tf.function
    def call(self, x, *args, **kwargs):
        # x: [inputs, dlatents]
        style = self.apply_bias(self.fc(x[1]))
        style = tf.reshape(style, self.style_target_shape)
        return x[0] * (style[:, 0] + 1) + style[:, 1]


class Noise(Layer):

    def __init__(self, randomize_noise, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = name if name is not None else scope + 'Noise'
        super(Noise, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.randomize_noise = randomize_noise
        self.tf_zero = tf.constant(0., dtype=self._dtype_policy.compute_dtype)
        self.tf_randomize_noise = tf.constant(1. if randomize_noise else -1., dtype=self._dtype_policy.compute_dtype)
        self.use_xla = use_xla
        self.scope = scope #+ 'Noise'

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.w_target_shape = [1, -1, 1, 1]
            self.noise_tail_shape = [1, input_shape[2], input_shape[3]]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.w_target_shape = [1, 1, 1, -1]
            self.noise_tail_shape = [input_shape[1], input_shape[2], 1]

        self.wshape = [self.channels_in]

        with tf.name_scope(self.scope) as scope:
            #with tf.name_scope(clean_name_scope(scope)) as final_scope:
            self.w = self.add_weight(
                name='noise',
                shape=self.wshape,
                initializer=tf.zeros_initializer(),
                trainable=True
            )
            # Always create non-random noise to allow easy testing
            # TODO: think how to handle batch dim (when building layer input_shape[0] is None)
            self.const_noise = self.add_weight(
                name='const_noise',
                shape=[1] + self.noise_tail_shape,
                initializer=tf.random_normal_initializer(),
                trainable=False
            )

    @tf.function
    def call(self, x, *args, **kwargs):
        # One can change layer weights (tf_randomize_noise) to switch between random and non random noise
        noise = tf.cond(
            tf.greater(self.tf_randomize_noise, self.tf_zero),
            lambda: tf.random.normal([tf.shape(x)[0]] + self.noise_tail_shape, dtype=self._dtype_policy.compute_dtype),
            lambda: tf.tile(self.const_noise, [tf.shape(x)[0], 1, 1, 1])
        )
        return x + noise * tf.reshape(self.w, self.w_target_shape)


class Const(Layer):

    def __init__(self, channel_size, dtype=DEFAULT_DTYPE,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Const')
        super(Const, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.channel_size = channel_size
        # Taken from the original implementation
        self.hw_size = 4
        self.scope = scope

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.shape = [1, self.channel_size, self.hw_size, self.hw_size]
        elif self.data_format == NHWC_FORMAT:
            self.shape = [1, self.hw_size, self.hw_size, self.channel_size]

        with tf.name_scope(self.scope):
            self.const_input = self.add_weight(
                name='const',
                shape=self.shape,
                initializer=tf.initializers.ones(),
                trainable=True
            )

    def call(self, x, *args, **kwargs):
        # x - latents
        return tf.tile(self.const_input, [tf.shape(x)[0], 1, 1, 1])


class MinibatchStdDev(Layer):

    def __init__(self, group_size=4, num_new_features=1, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'MinibatchStddev')
        super(MinibatchStdDev, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.group_size = group_size
        self.num_new_features = num_new_features
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.data_format == NCHW_FORMAT:
            _, c, h, w = x.shape
            n = tf.shape(x)[0]
            group_size = tf.math.minimum(self.group_size, n)     # Minibatch must be divisible or smaller than batch size
            # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c
            y = tf.reshape(x, [group_size, -1, self.num_new_features, c // self.num_new_features, h, w])
            y = tf.cast(y, tf.float32)                           # [GMncHW] Cast to fp32
            y -= tf.math.reduce_mean(y, axis=0, keepdims=True)   # [GMncHW] Subtract mean over group
            y = tf.reduce_mean(tf.square(y), axis=0)             # [MncHW] Variance over group
            y = tf.sqrt(y + 1e-8)                                # [MncHW] Stddev over group
            y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True) # [Mn111] Average over fmaps and pixels
            y = tf.reduce_mean(y, axis=[2])                      # [Mn11]
            y = tf.cast(y, x.dtype)                              # [Mn11] Cast back to original dtype
            y = tf.tile(y, [group_size, 1, h, w])                # [NnHW] Replicate over group and pixels
            return tf.concat([x, y], axis=1)
        elif self.data_format == NHWC_FORMAT:
            _, h, w, c = x.shape
            n = tf.shape(x)[0]
            group_size = tf.math.minimum(self.group_size, n)     # Minibatch must be divisible or smaller than batch size
            # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c
            y = tf.reshape(x, [group_size, -1, h, w, self.num_new_features, c // self.num_new_features])
            y = tf.cast(y, tf.float32)                           # [GMHWnc] Cast to fp32
            y -= tf.math.reduce_mean(y, axis=0, keepdims=True)   # [GMHWnc] Subtract mean over group
            y = tf.reduce_mean(tf.square(y), axis=0)             # [MHWnc] Variance over group
            y = tf.sqrt(y + 1e-8)                                # [MHWnc] Stddev over group
            y = tf.reduce_mean(y, axis=[1, 2, 4], keepdims=True) # [M11n1] Average over fmaps and pixels
            y = tf.reduce_mean(y, axis=[4])                      # [M11n]
            y = tf.cast(y, x.dtype)                              # [M11n] Cast back to original dtype
            y = tf.tile(y, [group_size, h, w, 1])                # [NHWn] Replicate over group and pixels
            return tf.concat([x, y], axis=3)


class WeightedSum(Layer):

    def __init__(self, dtype=DEFAULT_DTYPE, name=None):
        super(WeightedSum, self).__init__(dtype=dtype, name=name)
        # Note: for mixed precision training constants can have float16 dtype
        self.alpha =self.add_weight(
            name='alpha',
            initializer=tf.constant_initializer(0.),
            trainable=False,
            dtype=self._dtype_policy.compute_dtype,
            experimental_autocast=False
        )
        self.one = tf.constant(1., dtype=self._dtype_policy.compute_dtype, name='One')

    # Avoid using tf.function or alpha will be compiled (if it not set as non trainable weight)
    def call(self, inputs, *args, **kwargs):
        return (self.one - self.alpha) * inputs[0] + self.alpha * inputs[1]


# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.
class Fused_Upscale2d_ScaledConv2d(Layer):

    def __init__(self, fmaps, kernel_size, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Upscale2d_Conv2d')
        super(Fused_Upscale2d_ScaledConv2d, self).__init__(dtype=dtype, name=layer_name)
        assert kernel_size >= 1 and kernel_size % 2 == 1
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.scope = scope

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, 2, 2]
            self.os_tail = [self.fmaps, input_shape[2] * 2, input_shape[3] * 2]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, 2, 2, 1]
            self.os_tail = [input_shape[1] * 2, input_shape[2] * 2, self.fmaps]

        # Wshape is different from Scaled_Conv2d
        self.wshape = [self.kernel_size, self.kernel_size, self.fmaps, self.channels_in]
        self.fan_in = (self.kernel_size ** 2) * self.channels_in
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=self.wshape,
                initializer=initializer,
                trainable=True
            )

    @tf.function
    def call(self, x, *args, **kwargs):
        w = self.runtime_coef * self.w

        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])

        os = [tf.shape(x)[0]] + self.os_tail

        return tf.nn.conv2d_transpose(
            x, w, os, strides=self.strides, padding='SAME', data_format=self.data_format
        )


# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.
class Fused_ScaledConv2d_Downscale2d(Layer):

    def __init__(self, fmaps, kernel_size, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 use_xla=DEFAULT_USE_XLA, dtype=DEFAULT_DTYPE,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Conv2d_Downscale2d')
        super(Fused_ScaledConv2d_Downscale2d, self).__init__(dtype=dtype, name=layer_name)
        assert kernel_size >= 1 and kernel_size % 2 == 1
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.scope = scope

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, 2, 2]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, 2, 2, 1]

        self.wshape = [self.kernel_size, self.kernel_size, self.channels_in, self.fmaps]
        self.fan_in = np.prod(self.wshape[:-1])
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=self.wshape,
                initializer=initializer,
                trainable=True
            )

    @tf.function
    def call(self, x, *args, **kwargs):
        w = self.runtime_coef * self.w

        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25

        return tf.nn.conv2d(x, w, strides=self.strides, padding='SAME', data_format=self.data_format)


#----------------------------------------------------------------------------
# Layers as functions.

def layer_dtype(config, layer_type, act_name=None):
    use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        act_dtype = 'float32' if act_name in FP32_ACTIVATIONS else policy
        compute_dtype = policy.compute_dtype
    else:
        policy = 'float32'
        act_dtype = 'float32'
        compute_dtype = 'float32'

    if layer_type in ['conv2d', 'dense', 'bias', 'style_mod', 'noise', 'const']:
        return policy
    elif layer_type == 'act':
        return act_dtype
    elif layer_type in [
        'pixel_norm', 'instance_norm', 'blur2d', 'upscale2d', 'downscale2d', 'minibatch_stddev'
    ]:
        return compute_dtype
    else:
        assert False, 'Unknown layer type'


def dense_layer(x, units, gain, lrmul=LRMUL, scope='', config=None):
    use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
    truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(config, 'dense')
    return ScaledLinear(
        units=units, gain=gain,
        use_wscale=use_wscale, lrmul=lrmul, truncate_weights=truncate_weights,
        dtype=policy, data_format=data_format, scope=scope
    )(x)


def conv2d_layer(x, fmaps, kernel_size, gain, lrmul=LRMUL,
                 fused_up=False, fused_down=False, scope='', config=None):
    assert not (fused_up and fused_down)

    use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
    truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(config, 'conv2d')

    if fused_up:
        return Fused_Upscale2d_ScaledConv2d(
            fmaps=fmaps, kernel_size=kernel_size, gain=gain,
            use_wscale=use_wscale, lrmul=lrmul, truncate_weights=truncate_weights,
            dtype=policy, data_format=data_format, scope=scope
        )(x)
    elif fused_down:
        return Fused_ScaledConv2d_Downscale2d(
            fmaps=fmaps, kernel_size=kernel_size, gain=gain,
            use_wscale=use_wscale, lrmul=lrmul, truncate_weights=truncate_weights,
            dtype=policy, data_format=data_format, scope=scope
        )(x)
    else:
        return ScaledConv2d(
            fmaps=fmaps, kernel_size=kernel_size, gain=gain,
            use_wscale=use_wscale, lrmul=lrmul, truncate_weights=truncate_weights,
            dtype=policy, data_format=data_format, scope=scope
        )(x)


def bias_layer(x, lrmul=LRMUL, scope='', config=None):
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(config, 'bias')
    return Bias(lrmul=lrmul, dtype=policy, data_format=data_format, scope=scope)(x)


def act_layer(x, act_name, scope='', config=None):
    if act_name in ACTIVATION_FUNS_DICT.keys():
        act = ACTIVATION_FUNS_DICT[act_name]
    else:
        assert False, f"Activation '{act_name}' not supported. See ACTIVATION_FUNS_DICT"
    dtype = layer_dtype(config, 'act', act_name)
    return Activation(act, dtype=dtype, name=scope + act_name)(x)


def const_layer(x, channel_size, scope='', config=None):
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(config, 'const')
    return Const(channel_size=channel_size, dtype=policy, data_format=data_format, scope=scope)(x)


def noise_layer(x, scope='', config=None):
    randomize_noise = config.get(RANDOMIZE_NOISE, DEFAULT_RANDOMIZE_NOISE)
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(config, 'noise')
    return Noise(randomize_noise=randomize_noise, dtype=policy, data_format=data_format, scope=scope)(x)


def blur_layer(x, scope='', config=None):
    blur_filter = config.get(BLUR_FILTER, DEFAULT_BLUR_FILTER)
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(config, 'noise')
    return Blur2d(filter=blur_filter, dtype=dtype, scope=scope, data_format=data_format)(x) if blur_filter is not None else x


def pixel_norm_layer(x, scope='', config=None):
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(config, 'pixel_norm')
    return PixelNorm(dtype=dtype, data_format=data_format, scope=scope)(x)


def instance_norm_layer(x, scope='', config=None):
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(config, 'instance_norm')
    return InstanceNorm(dtype=dtype, data_format=data_format, scope=scope)(x)


def style_mod_layer(x, dlatents, scope='', config=None):
    use_bias = config.get(USE_BIAS, DEFAULT_USE_BIAS)
    use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
    truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(config, 'style_mod')
    return StyleMod(
        use_bias=use_bias, use_wscale=use_wscale, truncate_weights=truncate_weights,
        dtype=policy, data_format=data_format, scope=scope
    )([x, dlatents])


def downscale2d_layer(x, factor, config=None):
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(config, 'downscale2d')
    return Downscale2d(factor=factor, dtype=dtype, data_format=data_format)(x)


def upscale2d_layer(x, factor, config=None):
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(config, 'upscale2d')
    return Upscale2d(factor=factor, dtype=dtype, data_format=data_format)(x)


def minibatch_stddev_layer(x, scope='', config=None):
    group_size = config.get(MBSTD_GROUP_SIZE, 4)
    num_new_features = config.get(MBSTD_NUM_FEATURES, DEFAULT_MBSTD_NUM_FEATURES)
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(config, 'minibatch_stddev')
    return MinibatchStdDev(
        group_size=group_size, num_new_features=num_new_features,
        dtype=dtype, data_format=data_format, scope=scope
    )(x)
