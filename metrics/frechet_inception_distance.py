import os
from tqdm import tqdm

import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_input

from config import Config as cfg
from custom_layers import naive_upsample
from dataloader_utils import create_training_dataset
from utils import generate_latents, CACHE_DIR, toNHWC_AXIS, NCHW_FORMAT, validate_data_format,\
    enable_mixed_precision_policy, disable_mixed_precision_policy


FID_DIR = 'fid'
MU_REAL_KEY = 'mu_real'
SIGMA_REAL_KEY = 'sigma_real'


class FID:

    def __init__(self, image_size, num_samples, model_name, dataset_params, use_fp16, use_xla, **kwargs):
        assert isinstance(num_samples, int)
        # Inception model was trained on 299x299 images. So set minimal size to 256x256
        self.min_size = 256
        self.image_size = image_size if image_size > self.min_size else self.min_size
        self.image_res_log2 = int(np.log2(self.image_size))
        self.num_samples = num_samples
        self.model_name = model_name
        self.dataset_params = dataset_params
        self.data_format = dataset_params[cfg.DATA_FORMAT]
        validate_data_format(self.data_format)
        self.use_fp16 = use_fp16
        self.use_xla = use_xla
        # Required field for each metric class (used in tensorboard)
        self.name = f'FID_{num_samples // 1000}k'

        self.cache_dir = os.path.join(CACHE_DIR, model_name)
        self.cache_file = os.path.join(self.cache_dir, self.name + '.npz')
        if self.use_fp16:
            enable_mixed_precision_policy()
        # With pooling specified output shape will always be (None, 2048), so no need to provide input shape
        self.inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        self.activations_shape = [self.num_samples, self.inception_model.output_shape[1]]
        self.activations_dtype = np.float16 if self.use_fp16 else np.float32
        self.inception_model = tf.function(self.inception_model, jit_compile=self.use_xla)
        if self.use_fp16:
            disable_mixed_precision_policy()

    def create_images_dataset(self, batch_size):
        self.dataset_params.update(res=self.image_res_log2, batch_size=batch_size, cache=False)
        return create_training_dataset(**self.dataset_params)

    @tf.function
    def process_images(self, images):
        # TODO: think about face cropping (like for PPL metric)
        # TODO: think about downsampling images (like for PPL metric)
        # Upsample image to 256x256 if it's smaller than that. Inception was built for 256x256 images.
        if self.data_format == NCHW_FORMAT:
            shape_idx = 2
        else: # data_format == NHWC_FORMAT:
            shape_idx = 1
        if images.shape[shape_idx] < self.min_size:
            factor = self.min_size // images.shape[shape_idx]
            images = naive_upsample(images, factor, data_format=self.data_format)

        # Scale dynamic range from [-1,1] to [0,255] for Inception.
        images = (images + 1) * (255 / 2)

        # Convert images to network format (NHWC).
        if self.data_format == NCHW_FORMAT:
            images = tf.transpose(images, toNHWC_AXIS)

        # TODO: check if this is necessary
        # Cast to fp32.
        images = tf.cast(images, tf.float32)

        # Prepare images for Inception model.
        images = preprocess_inception_input(images)

        return images

    def compute_activations_stats(self, activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate_on_reals(self, batch_size):
        if os.path.exists(self.cache_file):
            data = np.load(self.cache_file)
            mu_real, sigma_real = data[MU_REAL_KEY], data[SIGMA_REAL_KEY]
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            images_dataset = iter(self.create_images_dataset(batch_size))
            activations = np.empty(self.activations_shape, dtype=self.activations_dtype)
            for idx in tqdm(range(0, self.num_samples, batch_size), 'FID metric reals steps'):
                start = idx * batch_size
                end = min(start + batch_size, self.num_samples)
                real_images = next(images_dataset)
                real_images = self.process_images(real_images)
                activations[start:end] = self.inception_model(real_images).numpy()[:(end-start)]
            mu_real, sigma_real = self.compute_activations_stats(activations)
            np.savez(self.cache_file, **{MU_REAL_KEY: mu_real, SIGMA_REAL_KEY: sigma_real})
        return mu_real, sigma_real

    def evaluate_on_fakes(self, batch_size, G_model):
        activations = np.empty(self.activations_shape, dtype=self.activations_dtype)
        z_dim = G_model.z_dim
        latents_dtype = G_model.latents_dtype
        for idx in tqdm(range(0, self.num_samples, batch_size), 'FID metric fakes steps'):
            start = idx * batch_size
            end = min(start + batch_size, self.num_samples)
            latents = generate_latents(batch_size, z_dim, latents_dtype)
            fake_images = G_model(latents, training=False, validation=True)
            fake_images = self.process_images(fake_images)
            activations[start:end] = self.inception_model(fake_images).numpy()[:(end-start)]
        mu_fake, sigma_fake = self.compute_activations_stats(activations)
        return mu_fake, sigma_fake

    def calculate_fid(self, real_stats, fake_stats):
        mu_real, sigma_real = real_stats
        mu_fake, sigma_fake = fake_stats
        m = np.square(mu_fake - mu_real).sum()
        s, _ = sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2.0 * s.real)
        return dist

    def run_metric(self, input_batch_size, G_model):
        # TODO: determine max batch size. For now just always use 32
        batch_size = min(max(input_batch_size, 32), 32)
        mu_real, sigma_real = self.evaluate_on_reals(batch_size)
        mu_fake, sigma_fake = self.evaluate_on_fakes(batch_size, G_model)
        dist = self.calculate_fid((mu_real, sigma_real), (mu_fake, sigma_fake))
        return dist
