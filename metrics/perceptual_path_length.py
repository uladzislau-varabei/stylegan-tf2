import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from .lpips_models.lpips_tensorflow import learned_perceptual_metric_model
from utils import generate_latents, toNHWC_AXIS, toNCHW_AXIS


#----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    # Note: the original implementation doesn't seem to work at least for me in Tf2.5.
    # If the function should normalize each vector in a batch separately, then it should be fixed in the given way
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=[1, 2, 3], keepdims=True))


# Linear interpolation
def lerp(a, b, t):
    # TODO: think how to remove tf.cast
    return a + (b - a) * tf.cast(t, a.dtype)


# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    # Make sure acos inputs have right boundaries (due to numeric rounds)
    d = tf.clip_by_value(d, -1., 1.)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)


#----------------------------------------------------------------------------

class PPL:

    def __init__(self, image_size, num_samples, epsilon, space, sampling, crop_face=False, **kwargs):
        assert space in ['w', 'z']
        assert sampling in ['full', 'end']
        assert isinstance(num_samples, int)
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.norm_constant = 1. / (epsilon ** 2)
        self.space = space
        self.sampling = sampling
        self.crop_face = crop_face
        # Required field for each metric class (used in tensorboard)
        self.name = f'PPL_{space}_{sampling}_{num_samples // 1000}k'

        # Min size for the network is 32x32.
        # TODO: think to which resolution should images be upsampled if size is smaller than that?
        # self.min_size = 32
        # Always upscale to 256
        self.min_size = 256
        self.input_image_size = image_size if image_size > self.min_size else self.min_size
        print('Input image size for PPL metric:', self.input_image_size)
        # Tf 2.x port of vgg16_zhang_perceptual
        vgg_ckpt_fn = os.path.join('metrics', 'lpips_models' ,'vgg', 'exported')
        lin_ckpt_fn = os.path.join('metrics', 'lpips_models', 'lin', 'exported')
        # Note: input images should be in NHWC format in range (0, 255)
        self.lpips_model = tf.function(learned_perceptual_metric_model(self.input_image_size, vgg_ckpt_fn, lin_ckpt_fn))

    @tf.function
    def process_images(self, images):
        # Crop only the face region (images are in NCHW format).
        if self.crop_face:
            c = int(images.shape[2] // 8)
            images = images[:, :, c * 3: c * 7, c * 2: c * 6]

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if images.shape[2] > 256:
            factor = images.shape[2] // 256
            images_target_shape = [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor]
            images = tf.reshape(images, images_target_shape)
            images = tf.reduce_mean(images, axis=[3, 5])
        elif images.shape[2] < self.min_size:
            s = images.shape
            factor = self.min_size // images.shape[2]
            # TODO: think about usage of upscale fun and data format (now works only with NCHW)
            images = tf.reshape(images, [-1, s[1], s[2], 1, s[3], 1])
            images = tf.tile(images, [1, 1, 1, factor, 1, factor])
            images = tf.reshape(images, [-1, s[1], s[2] * factor, s[3] * factor])

        # Scale dynamic range from [-1,1] to [0,255] for VGG.
        images = (images + 1) * (255 / 2)

        # Convert images to network format (NHWC).
        images = tf.transpose(images, toNHWC_AXIS)

        # TODO: check if this is necessary
        # Cast to fp32.
        images = tf.cast(images, tf.float32)

        return images

    def evaluate_distance_for_batch(self, batch_size, G_synthesis: tf.keras.Model, G_mapping: tf.keras.Model):
        # TODO: implement non-random noise (the same noise for the whole batch)

        # Generate random latents and interpolation t-values.
        # TODO: looks like it doesn't help
        dtype = G_mapping.compute_dtype
        lat_t01 = generate_latents(batch_size * 2, tuple(G_mapping.input_shape[1:]),  dtype)
        lerp_t = tf.random.uniform([batch_size], 0.0, 1.0 if self.sampling == 'full' else 0.0, dtype=dtype)

        # Interpolate in W or Z.
        if self.space == 'w':
            dlat_t01 = G_mapping(lat_t01, training=False)
            dlat_t0, dlat_t1 = dlat_t01[0::2], dlat_t01[1::2]
            dlat_e0 = lerp(dlat_t0, dlat_t1, lerp_t[:, tf.newaxis, tf.newaxis])
            dlat_e1 = lerp(dlat_t0, dlat_t1, lerp_t[:, tf.newaxis, tf.newaxis] + self.epsilon)
            dlat_e01 = tf.reshape(tf.stack([dlat_e0, dlat_e1], axis=1), dlat_t01.shape)
        else:  # space == 'z'
            lat_t0, lat_t1 = lat_t01[0::2], lat_t01[1::2]
            # Note: lerp_t shape is different from that in the original implementation
            # (otherwise error due to incompatible shape is raised)
            lat_e0 = slerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis, tf.newaxis, tf.newaxis])
            lat_e1 = slerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis, tf.newaxis, tf.newaxis] + self.epsilon)
            lat_e01 = tf.reshape(tf.stack([lat_e0, lat_e1], axis=1), lat_t01.shape)
            dlat_e01 = G_mapping(lat_e01, training=False)

        # Synthesize images.
        images = G_synthesis(dlat_e01, training=False)
        images = self.process_images(images)

        # Evaluate perceptual distance.
        img_e0, img_e1 = images[0::2], images[1::2]
        batch_distance = self.lpips_model([img_e0, img_e1]) * self.norm_constant
        return batch_distance

    def run_metric(self, input_batch_size, G_synthesis: tf.keras.Model, G_mapping: tf.keras.Model):
        # Sampling loop.
        all_distances = []
        # Max batch size for 256 resolution is 32
        batch_size = min(input_batch_size, 32) if self.min_size == 256 else input_batch_size
        for _ in tqdm(range(0, self.num_samples, batch_size), desc='PPL metric steps'):
        #for _ in range(0, self.num_samples, batch_size):
            all_distances += self.evaluate_distance_for_batch(batch_size, G_synthesis, G_mapping).numpy().tolist()

        all_distances = np.array(all_distances)

        # Reject outliers.
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        return np.mean(filtered_distances)
