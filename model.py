import os
import logging
import shutil
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from config import Config as cfg
from dataloader_utils import create_training_dataset
from losses import select_G_loss_fn, select_D_loss_fn
from metrics.metrics_utils import setup_metrics
from networks import Generator, Discriminator
# Utils imports
from checkpoint_utils import save_model, load_model, save_optimizer_loss_scale, load_optimizer_loss_scale,\
    remove_old_models
from utils import compute_alpha,\
    get_start_fp16_resolution, should_use_fp16,\
    create_images_dir_path, create_images_grid_title,\
    format_time, to_int_dict, validate_data_format, to_z_dim, mult_by_zero, is_last_step, should_write_summary,\
    load_images_paths, fast_save_grid
from utils import NHWC_FORMAT, NCHW_FORMAT,\
    DEFAULT_MODE, TRAIN_MODE, INFERENCE_MODE, BENCHMARK_MODE,\
    GENERATOR_NAME, DISCRIMINATOR_NAME, TRANSITION_MODE, STABILIZATION_MODE, SMOOTH_POSTFIX, OPTIMIZER_POSTFIX,\
    CACHE_DIR, DATASET_CACHE_DIR, TF_LOGS_DIR
from tf_utils import generate_latents, update_wsum_alpha,\
    get_compute_dtype, is_finite_grad,\
    trace_vars, get_gpu_memory_usage,\
    maybe_scale_loss, maybe_unscale_grads, is_optimizer_ready, set_optimizer_iters, set_tf_logging,\
    smooth_model_weights, convert_outputs_to_images, smooth_crossfade_images, run_model_on_batches
from tf_utils import DEFAULT_DATA_FORMAT, toNCHW_AXIS, toNHWC_AXIS


set_tf_logging(debug_mode=False)


MIXED_PRECISION_MAX_INIT_OPTIMIZER_ITERS = 20


"""
How to load model:
1) Create model
2) Initialize model
3) Load weights
"""


def show_vars_stats(vars):
    for idx, var in enumerate(vars):
        mean = tf.math.reduce_mean(var).numpy()
        std = tf.math.reduce_std(var).numpy()
        print(f'{idx}) {var.name}: mean={mean:.3f}, std={std:.3f}')


CPU_DEVICE = '/CPU:0'

D_KEY  = 'D'
G_KEY  = 'G'
GS_KEY = 'Gs'

FIRST_STEP_COND_KEY          = 'first_step_cond'
LAST_STEP_COND_KEY           = 'last_step_cond'
STAGE_IMAGES_KEY             = 'stage_images'
TRAINING_FINISHED_IMAGES_KEY = 'training_finished_images'
WRITE_SCALARS_SUMMARY_KEY    = 'write_scalars_summary'
WRITE_HISTS_SUMMARY_KEY      = 'write_hists_summary'
WRITE_LOSS_SCALE_SUMMARY_KEY = 'write_loss_scale_summary'
RUN_METRICS_KEY              = 'run_metrics'
SAVE_MODELS_KEY              = 'save_models'
SAVE_VALID_IMAGES_KEY        = 'save_valid_images'
SMOOTH_G_WEIGHTS_KEY         = 'smooth_G_weights'


class Scheduler:

    def __init__(self, config):
        self.target_resolution = config[cfg.TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(cfg.START_RESOLUTION, cfg.DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        # Training images and batches
        self.batch_sizes                = to_int_dict(config[cfg.BATCH_SIZES])
        self.final_batch_size           =             config.get(cfg.FINAL_BATCH_SIZE, cfg.DEFAULT_FINAL_BATCH_SIZE)
        # If not provided, take value from common dict
        if self.final_batch_size is None:
            self.final_batch_size = self.batch_sizes[self.target_resolution]
        self.total_kimages              =             config.get(cfg.TOTAL_KIMAGES, cfg.DEFAULT_TOTAL_KIMAGES)
        self.transition_kimages         =             config.get(cfg.TRANSITION_KIMAGES, cfg.DEFAULT_TRANSITION_KIMAGES)
        self.transition_kimages_dict    = to_int_dict(config.get(cfg.TRANSITION_KIMAGES_DICT, cfg.DEFAULT_TRANSITION_KIMAGES_DICT))
        self.stabilization_kimages      =             config.get(cfg.STABILIZATION_KIMAGES, cfg.DEFAULT_STABILIZATION_KIMAGES)
        self.stabilization_kimages_dict = to_int_dict(config.get(cfg.STABILIZATION_KIMAGES_DICT, cfg.DEFAULT_STABILIZATION_KIMAGES_DICT))

        self.Gs_beta              = config.get(cfg.G_SMOOTHING_BETA, cfg.DEFAULT_G_SMOOTHING_BETA)
        self.Gs_beta_kimgs        = config.get(cfg.G_SMOOTHING_BETA_KIMAGES, cfg.DEFAULT_G_SMOOTHING_BETA_KIMAGES)
        # Resolution-specific betas for Gs
        self.Gs_betas = {}

        self.reset_opt_state_for_new_lod = config.get(cfg.RESET_OPT_STATE_FOR_NEW_LOD, cfg.DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD)

        self.validate_config()

    def validate_config(self):
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            if 2 ** res not in self.batch_sizes.keys():
                assert False, f'Missing batch size for res={2**res}'
        self.get_n_steps_for_last_stage()

    def validate_res_and_mode_combination(self, res, mode):
        assert self.start_resolution_log2 <= res <= self.resolution_log2
        assert res > self.start_resolution_log2 or mode == STABILIZATION_MODE, \
            'For start resolution only stabilization stage is run'

    def compute_optimizer_steps(self, res, stage):
        # Note: res and stage refer to the last used model not the current one
        n_steps = 0
        if self.reset_opt_state_for_new_lod:
            assert stage == TRANSITION_MODE, \
                'If optimizers states should be reset for each resolution, ' \
                'then optimizers weights should only be loaded from transition stage'
            n_steps = self.get_n_steps_for_stage(res, stage)
        else:
            # Optimizer state is not reset, so sum all previous number of iters
            for r in range(self.start_resolution_log2, res + 1):
                if r > self.start_resolution_log2:
                    n_steps += self.get_n_steps_for_stage(r, TRANSITION_MODE)
                if r < res or (r == res and stage == STABILIZATION_MODE):
                    n_steps += self.get_n_steps_for_stage(r, STABILIZATION_MODE)
        return np.array(n_steps)

    def get_n_steps_for_last_stage(self):
        batch_size = self.final_batch_size
        # Simple case when total number of training images is not provided
        if self.total_kimages is None:
            n_kimages = self.stabilization_kimages_dict.get(self.target_resolution, self.stabilization_kimages)
            return int(np.ceil(1000 * n_kimages / batch_size))
        # Advanced case when total number of training images is not provided
        n_kimages = 0
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            if res > self.start_resolution_log2:
                n_kimages += self.transition_kimages_dict.get(2 ** res, self.transition_kimages)
                # TODO: for debugging, remove later
                # print(f'After res={res}, mode={TRANSITION_MODE}', n_kimages)
            if res < self.resolution_log2:
                n_kimages += self.stabilization_kimages_dict.get(2 ** res, self.stabilization_kimages)
                # TODO: for debugging, remove later
                # print(f'After res={res}, mode={STABILIZATION_MODE}', n_kimages)
        n_kimages = self.total_kimages - n_kimages
        # TODO: for debugging, remove later
        # print(f'n_kimages for the last stage: {n_kimages}')
        logging.info(f'n_kimages for the last stage: {n_kimages}')
        assert n_kimages > 0, 'Total number of images is greater than total number of images for all stages'
        return int(np.ceil(1000 * n_kimages / batch_size))

    def get_n_steps_for_stage(self, res, mode):
        """
        Returns number of training steps for provided res and mode
        """
        self.validate_res_and_mode_combination(res, mode)
        images_res = 2 ** res
        batch_size = self.batch_sizes[images_res] # batch sze can only be different for the last stage
        if mode == TRANSITION_MODE:
            n_kimages = self.transition_kimages_dict.get(images_res, self.transition_kimages)
            return int(np.ceil(1000 * n_kimages / batch_size))
        elif mode == STABILIZATION_MODE:
            if res < self.resolution_log2:
                n_kimages = self.stabilization_kimages_dict.get(images_res, self.stabilization_kimages)
                return int(np.ceil(1000 * n_kimages / batch_size))
            else:
                # Last training stage, which usually uses more images
                return self.get_n_steps_for_last_stage()

    def get_stage_start_processed_images(self, res, mode):
        """
        Returns number of processed images when training for provided res and mode is started
        """
        self.validate_res_and_mode_combination(res, mode)
        n_kimages = 0
        # Iterate for all previous resolutions
        for r in range(self.start_resolution_log2, res + 1):
            images_res = 2 ** r
            if r == res:
                if mode == STABILIZATION_MODE and r > self.start_resolution_log2:
                    n_kimages += self.transition_kimages_dict.get(images_res, self.transition_kimages)
            else:
                # The first resolution doesn't have transition stage
                if r > self.start_resolution_log2:
                    n_kimages += self.transition_kimages_dict.get(images_res, self.transition_kimages)
                n_kimages += self.stabilization_kimages_dict.get(images_res, self.stabilization_kimages)
        return int(1000 * n_kimages)

    def get_stage_end_processed_images(self, res, mode):
        if mode == TRANSITION_MODE:
            stage_kimages = self.transition_kimages_dict.get(2 ** res, self.transition_kimages)
        else:  # mode == STABILIZATION_MODE
            stage_kimages = self.stabilization_kimages_dict.get(2 ** res, self.stabilization_kimages)
        stage_images = int(1000 * stage_kimages) + self.get_stage_start_processed_images(res, mode)
        return stage_images

    def get_previous_res_and_stage(self, res, stage):
        if stage == STABILIZATION_MODE:
            load_res = res
            load_stage = TRANSITION_MODE
        else:
            load_res = res - 1
            load_stage = STABILIZATION_MODE
        return load_res, load_stage


class StyleGAN:

    def __init__(self, config, mode=DEFAULT_MODE, images_paths=None, res=None, stage=None,
                 single_process_training=False):

        self.target_resolution = config[cfg.TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(cfg.START_RESOLUTION, cfg.DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.latent_size = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.z_dim = to_z_dim(self.latent_size, self.data_format)

        # Training images and batches
        self.scheduler = Scheduler(config)
        self.batch_sizes = to_int_dict(config[cfg.BATCH_SIZES])
        self.final_batch_size = config.get(cfg.FINAL_BATCH_SIZE, cfg.DEFAULT_FINAL_BATCH_SIZE)
        # If not provided, take value from common dict
        if self.final_batch_size is None:
            self.final_batch_size = self.batch_sizes[self.target_resolution]
        self.batch_repeats = config.get(cfg.BATCH_REPEATS, cfg.DEFAULT_BATCH_REPEATS)

        # Computations
        self.use_mixed_precision  = config.get(cfg.USE_MIXED_PRECISION, cfg.DEFAULT_USE_MIXED_PRECISION)
        self.num_fp16_resolutions = config.get(cfg.NUM_FP16_RESOLUTIONS, cfg.DEFAULT_NUM_FP16_RESOLUTIONS)
        self.start_fp16_resolution_log2 =\
            get_start_fp16_resolution(self.num_fp16_resolutions, self.start_resolution_log2, self.resolution_log2)
        self.compute_dtype        = get_compute_dtype(self.use_mixed_precision)
        self.use_xla              = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
        self.use_Gs               = config.get(cfg.USE_G_SMOOTHING, cfg.DEFAULT_USE_G_SMOOTHING)
        self.use_gpu_for_Gs       = config.get(cfg.USE_GPU_FOR_GS, cfg.DEFAULT_USE_GPU_FOR_GS)
        self.Gs_beta              = config.get(cfg.G_SMOOTHING_BETA, cfg.DEFAULT_G_SMOOTHING_BETA)
        self.Gs_beta_kimgs        = config.get(cfg.G_SMOOTHING_BETA_KIMAGES, cfg.DEFAULT_G_SMOOTHING_BETA_KIMAGES)
        # Resolution-specific betas for Gs
        self.Gs_betas = {}
        # Used for training in a single process
        self.clear_session_for_new_model = True

        # Dataset
        self.dataset_hw_ratio      = config.get(cfg.DATASET_HW_RATIO, cfg.DEFAULT_DATASET_HW_RATIO)
        self.dataset_max_cache_res = config.get(cfg.DATASET_MAX_CACHE_RES, cfg.DEFAULT_DATASET_MAX_CACHE_RES)
        if images_paths is None:
            images_paths = load_images_paths(config)
        # These options are used for metrics
        self.dataset_params = {
            'fpaths'              : images_paths,
            'hw_ratio'            : self.dataset_hw_ratio,
            'mirror_augment'      : config.get(cfg.MIRROR_AUGMENT, cfg.DEFAULT_MIRROR_AUGMENT),
            'shuffle_dataset'     : config.get(cfg.SHUFFLE_DATASET, cfg.DEFAULT_SHUFFLE_DATASET),
            'data_format'         : self.data_format,
            'use_fp16'            : self.use_mixed_precision,
            'n_parallel_calls'    : config.get(cfg.DATASET_N_PARALLEL_CALLS, cfg.DEFAULT_DATASET_N_PARALLEL_CALLS),
            'n_prefetched_batches': config.get(cfg.DATASET_N_PREFETCHED_BATCHES, cfg.DEFAULT_DATASET_N_PREFETCHED_BATCHES)
        }

        # Losses
        self.G_loss_fn_name = config.get(cfg.G_LOSS_FN, cfg.DEFAULT_G_LOSS_FN)
        self.D_loss_fn_name = config.get(cfg.D_LOSS_FN, cfg.DEFAULT_D_LOSS_FN)
        self.G_loss_fn = select_G_loss_fn(self.G_loss_fn_name)
        self.D_loss_fn = select_D_loss_fn(self.D_loss_fn_name)
        self.G_loss_params = config.get(cfg.G_LOSS_FN_PARAMS, cfg.DEFAULT_G_LOSS_FN_PARAMS)
        self.D_loss_params = config.get(cfg.D_LOSS_FN_PARAMS, cfg.DEFAULT_D_LOSS_FN_PARAMS)

        # Optimizers options
        self.G_learning_rate = config.get(cfg.G_LEARNING_RATE, cfg.DEFAULT_G_LEARNING_RATE)
        self.D_learning_rate = config.get(cfg.D_LEARNING_RATE, cfg.DEFAULT_D_LEARNING_RATE)
        self.G_learning_rate_dict = to_int_dict(config.get(cfg.G_LEARNING_RATE_DICT, cfg.DEFAULT_G_LEARNING_RATE_DICT))
        self.D_learning_rate_dict = to_int_dict(config.get(cfg.D_LEARNING_RATE_DICT, cfg.DEFAULT_D_LEARNING_RATE_DICT))
        self.beta1 = config.get(cfg.ADAM_BETA1, cfg.DEFAULT_ADAM_BETA1)
        self.beta2 = config.get(cfg.ADAM_BETA2, cfg.DEFAULT_ADAM_BETA2)
        self.reset_opt_state_for_new_lod = config.get(cfg.RESET_OPT_STATE_FOR_NEW_LOD, cfg.DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD)

        # Valid images options
        self.valid_grid_nrows = config.get(cfg.VALID_GRID_NROWS, cfg.DEFAULT_VALID_GRID_NROWS)
        self.valid_grid_ncols = config.get(cfg.VALID_GRID_NCOLS, cfg.DEFAULT_VALID_GRID_NCOLS)
        self.valid_grid_padding = 2
        self.min_target_single_image_size = config.get(cfg.VALID_MIN_TARGET_SINGLE_IMAGE_SIZE, cfg.DEFAULT_VALID_MIN_TARGET_SINGLE_IMAGE_SIZE)
        if self.min_target_single_image_size < 0:
           self.min_target_single_image_size = max(2 ** (self.resolution_log2 - 1), 2 ** 7)
        self.max_png_res = config.get(cfg.VALID_MAX_PNG_RES, cfg.DEFAULT_VALID_MAX_PNG_RES)

        # Summaries
        self.model_name            =            config[cfg.MODEL_NAME]
        self.metrics               =            config.get(cfg.METRICS_DICT, cfg.DEFAULT_METRICS_DICT)
        self.storage_path          =            config.get(cfg.STORAGE_PATH, cfg.DEFAULT_STORAGE_PATH)
        self.max_models_to_keep    =            config.get(cfg.MAX_MODELS_TO_KEEP, cfg.DEFAULT_MAX_MODELS_TO_KEEP)
        self.run_metrics_every     = int(1000 * config.get(cfg.RUN_METRICS_EVERY_KIMAGES, cfg.DEFAULT_RUN_METRICS_EVERY_KIMAGES))
        self.summary_scalars_every = int(1000 * config.get(cfg.SUMMARY_SCALARS_EVERY_KIMAGES, cfg.DEFAULT_SUMMARY_SCALARS_EVERY_KIMAGES))
        self.summary_hists_every   = int(1000 * config.get(cfg.SUMMARY_HISTS_EVERY_KIMAGES, cfg.DEFAULT_SUMMARY_HISTS_EVERY_KIMAGES))
        self.save_model_every      = int(1000 * config.get(cfg.SAVE_MODEL_EVERY_KIMAGES, cfg.DEFAULT_SAVE_MODEL_EVERY_KIMAGES))
        self.save_images_every     = int(1000 * config.get(cfg.SAVE_IMAGES_EVERY_KIMAGES, cfg.DEFAULT_SAVE_IMAGES_EVERY_KIMAGES))
        self.logs_path = os.path.join(TF_LOGS_DIR, self.model_name)
        self.writers_dirs = {
            res: os.path.join(self.logs_path, f'{2**res}x{2**res}')
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.summary_writers = {
            res: tf.summary.create_file_writer(self.writers_dirs[res])
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.valid_latents = self.initialize_valid_latents()

        self.G_object = Generator(config)
        self.D_object = Discriminator(config)
        # Maybe create smoothed generator
        if self.use_Gs:
            Gs_config = config
            self.Gs_valid_latents = self.valid_latents
            self.Gs_device = '/GPU:0' if self.use_gpu_for_Gs else '/CPU:0'
            if not self.use_gpu_for_Gs:
                Gs_config[cfg.DATA_FORMAT] = NHWC_FORMAT
                self.Gs_valid_latents = tf.transpose(self.valid_latents, toNHWC_AXIS)
            self.Gs_object = Generator(Gs_config)

        self.initialize_models(res, stage)
        if mode == INFERENCE_MODE:
            print('Ready for inference')
        else:
            self.setup_Gs_betas(res)
            self.create_images_datasets(res, stage)
            # TODO: metrics result in a process crash for high resolutions with large batch sizes (which work for benchmark)
            self.metrics_objects = setup_metrics(2 ** res,
                                                 hw_ratio=self.dataset_hw_ratio,
                                                 dataset_params=self.dataset_params,
                                                 use_fp16=self.use_mixed_precision,
                                                 use_xla=self.use_xla,
                                                 model_name=self.model_name,
                                                 metrics=self.metrics,
                                                 benchmark_mode=(mode == BENCHMARK_MODE))
            if mode == BENCHMARK_MODE:
                self.initialize_optimizers(create_all_variables=False, benchmark=True)
            elif mode == TRAIN_MODE:
                if single_process_training:
                    # TODO: think about using metrics and updating optimizers scales
                    self.initialize_optimizers(create_all_variables=True)
                else:
                    self.initialize_optimizers(create_all_variables=False, res=res, stage=stage)

    def initialize_valid_latents(self):
        latents_dir  = os.path.join(CACHE_DIR, self.model_name)
        latents_path = os.path.join(latents_dir, 'latents.npy')
        if os.path.exists(latents_path):
            latents = tf.constant(np.load(latents_path, allow_pickle=False))
            logging.info('Loaded valid latents from file')
        else:
            os.makedirs(latents_dir, exist_ok=True)
            latents = self.generate_latents(self.valid_grid_nrows * self.valid_grid_ncols)
            np.save(latents_path, latents.numpy(), allow_pickle=False)
            logging.info('Valid latents not found. Created and saved new samples')
        return latents

    def initialize_models(self, model_res=None, stage=None):
        if self.use_mixed_precision:
            logging.info(f'Start fp16 resolution: {self.start_fp16_resolution_log2}')
        self.G_object.initialize_G_model(model_res=model_res, mode=stage)
        self.D_object.initialize_D_model(model_res=model_res, mode=stage)
        if self.use_Gs:
            with tf.device(self.Gs_device):
                self.Gs_object.initialize_G_model(model_res=model_res, mode=stage)
                # Case for training in single process
                if model_res is None and stage is None:
                    for r in range(self.start_resolution_log2, self.resolution_log2 + 1):
                        self.Gs_object.toRGB_layers[r].set_weights(self.G_object.toRGB_layers[r].get_weights())
                    model_res = self.resolution_log2
                    stage = STABILIZATION_MODE
                G_model = self.G_object.create_G_model(model_res=model_res, mode=stage)
                Gs_model = self.Gs_object.create_G_model(model_res=model_res, mode=stage)
                Gs_model.set_weights(G_model.get_weights())

    def is_last_stage(self, res, stage):
        return (res == self.resolution_log2) and (stage == STABILIZATION_MODE)

    def get_batch_size(self, res, stage):
        if self.is_last_stage(res, stage):
            batch_size = self.final_batch_size
        else:
            batch_size = self.batch_sizes[2 ** res]
        return batch_size

    def setup_Gs_betas(self, res=None):
        def get_res_beta(self, res, stage=None):
            if self.Gs_beta is None:
                batch_size = self.get_batch_size(res, stage)
                beta = tf.constant(0.5 ** (batch_size / (1000.0 * self.Gs_beta_kimgs)), dtype='float32')
            else:
                beta = tf.constant(self.Gs_beta, dtype='float32')
            logging.info(f'Gs beta for res={res} and stage={stage}: {beta}')
            return beta

        if res is None:
            for r in range(self.start_resolution_log2, self.resolution_log2 + 1):
                self.Gs_betas[r] = get_res_beta(self, r)
        else:
            self.Gs_betas[res] = get_res_beta(self, res)

        self.final_Gs_beta = get_res_beta(self, self.resolution_log2, STABILIZATION_MODE)

    def get_Gs_beta(self, res, stage):
        if self.is_last_stage(res, stage):
            return self.final_Gs_beta
        else:
            return self.Gs_betas[res]

    def trace_graphs(self):
        self.G_object.trace_G_graphs(self.summary_writers, self.writers_dirs)
        self.D_object.trace_D_graphs(self.summary_writers, self.writers_dirs)

        self.G_object.initialize_G_model(model_res=self.resolution_log2, mode=TRANSITION_MODE)
        G_model = self.G_object.create_G_model(model_res=self.resolution_log2, mode=TRANSITION_MODE)
        logging.info('\nThe biggest Generator:\n')
        G_model.summary(print_fn=logging.info)

        self.D_object.initialize_D_model(model_res=self.resolution_log2, mode=TRANSITION_MODE)
        D_model = self.D_object.create_D_model(model_res=self.resolution_log2, mode=TRANSITION_MODE)
        logging.info('\nThe biggest Discriminator:\n')
        D_model.summary(print_fn=logging.info)

    def initialize_main_vars_for_G_optimizer(self, n_iters):
        # The purpose of this function is to create all variables needed for optimizer
        # when using mixed precision training.
        # Usually if grads are not finite the variables have prefix 'cond_1' (maybe it depends on Tensorflow version)
        step = tf.Variable(0, trainable=False, dtype=tf.int64)
        write_summary = tf.Variable(False, trainable=False, dtype=tf.bool)

        res = self.resolution_log2
        batch_size = self.batch_sizes[2 ** res]

        G_model = self.G_object.create_G_model(res, mode=TRANSITION_MODE)
        D_model = self.D_object.create_D_model(res, mode=TRANSITION_MODE)
        G_vars = G_model.trainable_variables

        for i in range(n_iters):
            # TODO: for debugging, remove later
            print('i=', i)
            latents = self.generate_latents(batch_size)
            with tf.GradientTape(watch_accessed_variables=False) as G_tape:
                G_tape.watch(G_vars)
                G_loss = self.G_loss_fn(G_model, D_model, self.G_optimizer,
                                        latents=latents,
                                        write_summary=write_summary,
                                        step=step,
                                        **self.G_loss_params)
                G_loss = maybe_scale_loss(G_loss, self.G_optimizer)
                #print('G loss computed')

            # No need to update weights!
            # TODO: before it was mult by zero, but results may be strange for mixed precision
            G_grads = (G_tape.gradient(G_loss, G_vars))
            G_grads = maybe_unscale_grads(G_grads, self.G_optimizer)
            G_grads = [mult_by_zero(g) for g in G_grads]
            #print('G gradients obtained')

            self.G_optimizer.apply_gradients(zip(G_grads, G_vars))
            #var_names = [var.name for var in self.G_optimizer.weights]
            #print('G vars before output:\n', var_names)
            #print('G gradients applied')

            # TODO: for debugging, remove later
            if is_optimizer_ready(self.G_optimizer):
                print(f'Optimizer is ready in {i} steps')
                print('Vars:\n', [var.name for var in self.G_optimizer.weights])
                break

    def initialize_additional_vars_for_G_optimizer(self):
        logging.info('Creating slots for intermediate output layers...')
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            batch_size = self.batch_sizes[2 ** res]
            # toRGB layers
            to_layer = self.G_object.toRGB_layers[res]
            to_input_shape = [batch_size] + list(to_layer.input_shape[1:])
            to_inputs = tf.zeros(shape=to_input_shape, dtype=self.compute_dtype)

            with tf.GradientTape() as tape:
                # Use outputs as loss values
                loss = maybe_scale_loss(to_layer(to_inputs), self.G_optimizer)

            G_vars = to_layer.trainable_variables
            # No need to update weights!
            # TODO: before it was mult by zero, but results may be strange for mixed precision
            G_grads = mult_by_zero(tape.gradient(loss, G_vars))
            G_grads = maybe_unscale_grads(G_grads, self.G_optimizer)
            self.G_optimizer.apply_gradients(zip(G_grads, G_vars))

        logging.info('G optimizer slots created!')

    def initialize_G_optimizer(self, res, stage, create_all_variables: bool = False, benchmark: bool = False):
        self.G_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.G_learning_rate,
            beta_1=self.beta1, beta_2=self.beta2,
            epsilon=1e-8,
            name='G_Adam'
        )
        if self.use_mixed_precision:
            initial_scale, dynamic = self.get_optimizer_initial_loss_scale(GENERATOR_NAME, res, stage, benchmark)
            self.G_optimizer = LossScaleOptimizer(self.G_optimizer, dynamic=dynamic, initial_scale=initial_scale)
        self.G_optimizer.use_mixed_precision = self.use_mixed_precision

        if not create_all_variables:
            # Variables will be automatically created (only valid if optimizer state is not restored)
            return

        # 1-st step: create optimizer states for all internal and final output layers
        init_iters = MIXED_PRECISION_MAX_INIT_OPTIMIZER_ITERS if self.use_mixed_precision else 1
        self.initialize_main_vars_for_G_optimizer(init_iters)
        # 2-nd step: create optimizer states for all remaining output layers
        self.initialize_additional_vars_for_G_optimizer()

        # TODO: for debugging, remove later
        var_names = sorted([var.name for var in self.G_optimizer.weights])
        print('\nG vars after initialization:')
        for idx, var in enumerate(var_names, 1):
            print(f'{idx}: {var}')

    def initialize_main_vars_for_D_optimizer(self, n_iters):
        # The purpose of this function is to create all variables needed for optimizer
        # when using mixed precision training.
        # Usually if grads are not finite the variables have prefix 'cond_1' (maybe it depends on Tensorflow version)
        step = tf.Variable(0, trainable=False, dtype=tf.int64)
        write_summary = tf.Variable(False, trainable=False, dtype=tf.bool)

        res = self.resolution_log2
        batch_size = self.batch_sizes[2 ** res]

        G_model = self.G_object.create_G_model(res, mode=TRANSITION_MODE)
        D_model = self.D_object.create_D_model(res, mode=TRANSITION_MODE)
        D_vars = D_model.trainable_variables
        D_input_shape = self.D_object.D_input_shape(res)

        for i in range(n_iters):
            # TODO: for debugging, remove later
            print('i=', i)
            latents = self.generate_latents(batch_size)
            # It's probably better to use ranges which are similar to the ones in real images
            images = tf.random.uniform(
                shape=[batch_size] + D_input_shape, minval=-1.0, maxval=1.0, dtype=self.compute_dtype
            )
            with tf.GradientTape(watch_accessed_variables=False) as D_tape:
                D_tape.watch(D_vars)
                D_loss = self.D_loss_fn(G_model, D_model, self.D_optimizer,
                                        latents=latents,
                                        real_images=images,
                                        write_summary=write_summary,
                                        step=step,
                                        **self.D_loss_params)
                D_loss = maybe_scale_loss(D_loss, self.D_optimizer)
                #print('D loss computed')

            # No need to update weights!
            # TODO: before it was mult by zero, but results may be strange for mixed precision
            D_grads = (D_tape.gradient(D_loss, D_vars))
            D_grads = maybe_unscale_grads(D_grads, self.D_optimizer)
            #print('D gradients obtained')

            self.D_optimizer.apply_gradients(zip(D_grads, D_vars))
            #var_names = [var.name for var in self.D_optimizer.weights]
            #print('D vars before output:\n', var_names)

            # TODO: for debugging, remove later
            if is_optimizer_ready(self.D_optimizer):
                print(f'Optimizer is ready in {i} steps')
                print('Vars:\n', [var.name for var in self.D_optimizer.weights])
                break

    def initialize_additional_vars_for_D_optimizer(self):
        logging.info('Creating slots for intermediate output layers...')
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            batch_size = self.batch_sizes[2 ** res]
            # fromRGB layers
            from_layer = self.D_object.fromRGB_layers[res]
            from_input_shape = [batch_size] + list(from_layer.input_shape[1:])
            from_inputs = tf.zeros(shape=from_input_shape, dtype=self.compute_dtype)

            with tf.GradientTape() as tape:
                # Use outputs as loss values
                loss = maybe_scale_loss(from_layer(from_inputs), self.D_optimizer)

            D_vars = from_layer.trainable_variables
            # No need to update weights!
            # TODO: before it was mult by zero, but results may be strange for mixed precision
            D_grads = (tape.gradient(loss, D_vars))
            D_grads = maybe_unscale_grads(D_grads, self.D_optimizer)
            D_grads = [mult_by_zero(g) for g in D_grads]
            self.D_optimizer.apply_gradients(zip(D_grads, D_vars))

        logging.info('D optimizer slots created!')

    def initialize_D_optimizer(self, res, stage, create_all_variables: bool = False, benchmark: bool = False):
        self.D_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.D_learning_rate,
            beta_1=self.beta1, beta_2=self.beta2,
            epsilon=1e-8,
            name='D_Adam'
        )
        if self.use_mixed_precision:
            initial_scale, dynamic = self.get_optimizer_initial_loss_scale(DISCRIMINATOR_NAME, res, stage, benchmark)
            self.D_optimizer = LossScaleOptimizer(self.D_optimizer, dynamic=dynamic, initial_scale=initial_scale)
        self.D_optimizer.use_mixed_precision = self.use_mixed_precision

        if not create_all_variables:
            # Variables will be automatically created (only valid if optimizer state is not restored)
            return

        # 1-st step: create optimizer states for all internal and final input layer
        init_iters = MIXED_PRECISION_MAX_INIT_OPTIMIZER_ITERS if self.use_mixed_precision else 1
        self.initialize_main_vars_for_D_optimizer(init_iters)
        # 2-nd step: create optimizer states for all remaining input layers
        self.initialize_additional_vars_for_D_optimizer()

        # TODO: for debugging, remove later
        var_names = sorted([var.name for var in self.D_optimizer.weights])
        print('\nD vars after initialization:')
        for idx, var in enumerate(var_names, 1):
            print(f'{idx}: {var}')

    def get_optimizer_initial_loss_scale(self, model_type, res, stage, benchmark: bool = False):
        if benchmark:
            # Use default values for loss scale optimizer
            return 2 ** 15, True

        use_fp16 = should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)
        if not use_fp16:
            logging.info(f"Res {res} doesn't use fp16, so constant loss scale for {model_type} optimizer is set to 1")
            return 1., False

        # Note: res and stage refer to the current train stage, so they should be adjusted
        if (res is None) and (stage is None):
            logging.info(f'Default initial loss scale is used for {model_type} optimizer as res and stage are not provided')
            return None, True

        if (res <= self.start_fp16_resolution_log2) or (res == self.start_resolution_log2):
            logging.info(f'Default initial loss scale is used for {model_type} optimizer '
                         f'for resolution lower then or equal to start_fp16_resolution')
            return None, True

        load_res, load_stage = self.scheduler.get_previous_res_and_stage(res, stage)
        kwargs = {
            'model_name': self.model_name,
            'model_type': model_type,
            'res': load_res,
            'stage': load_stage,
            'step': None,
            'storage_path': self.storage_path
        }
        last_loss_scale = load_optimizer_loss_scale(**kwargs)
        # Note: init default loss scale is 2 ** 15. Start with a value which is several steps higher than the last one
        init_loss_scale = min(last_loss_scale * (2 ** 4), 2 ** 15)
        logging.info(f'Loss scale for {model_type} optimizer for res={res}, stage={stage}: '
                     f'init_loss_scale = {init_loss_scale}, last_loss_scale = {last_loss_scale}')
        return init_loss_scale, True

    def reset_optimizers_state(self):
        G_optimizer_new_weights = mult_by_zero(self.G_optimizer.get_weights())
        self.G_optimizer.set_weights(G_optimizer_new_weights)

        D_optimizer_new_weights = mult_by_zero(self.D_optimizer.get_weights())
        self.D_optimizer.set_weights(D_optimizer_new_weights)

        logging.info('Optimizers states reset')

    def initialize_optimizers(self, res=None, stage=None, create_all_variables: bool = False, benchmark: bool = False):
        start_time = time.time()
        logging.info('Initializing optimizers...')

        # 1: create optimizer states for all internal and final output layers
        # 2: create optimizer states for all intermediate output layers
        # 3: set slots of optimizer to zero
        self.initialize_G_optimizer(res, stage, create_all_variables, benchmark)
        self.initialize_D_optimizer(res, stage, create_all_variables, benchmark)

        if create_all_variables:
            self.reset_optimizers_state()

        total_time = time.time() - start_time
        logging.info(f'Optimizers initialized in {total_time:.3f} seconds!')

    def adjust_optimizers_learning_rate(self, res):
        G_lr = self.G_learning_rate_dict.get(2 ** res, self.G_learning_rate)
        self.G_optimizer.learning_rate.assign(G_lr)

        D_lr = self.D_learning_rate_dict.get(2 ** res, self.D_learning_rate)
        self.D_optimizer.learning_rate.assign(D_lr)

        logging.info(f'Optimizers learning rates for res={res}: '
                     f'G_lr={self.G_optimizer.learning_rate.numpy():.4f}, '
                     f'D_lr={self.D_optimizer.learning_rate.numpy():.4f}')

    def zero_optimizers_learning_rate(self):
        self.G_optimizer.learning_rate.assign(0.)
        self.D_optimizer.learning_rate.assign(0.)

    def restore_optimizers_state(self, res, stage):
        # Note: res and stage refer to the last used model, not the current one
        if self.reset_opt_state_for_new_lod:
            assert stage == TRANSITION_MODE, \
                'If optimizers states should be reset for each resolution, ' \
                'then optimizers weights should only be loaded from transition stage'
        logging.info(f'Restoring optimizer state for res={res}, stage={stage}...')
        shared_kwargs = {
            'model_name': self.model_name,
            'res': res,
            'stage': stage,
            'step': None,
            'storage_path': self.storage_path
        }

        MODEL_ARG = 'model'
        MODEL_TYPE_ARG = 'model_type'

        D_optimizer_kwargs = {
            MODEL_ARG: self.D_optimizer._optimizer if self.use_mixed_precision else self.D_optimizer,
            MODEL_TYPE_ARG: DISCRIMINATOR_NAME + OPTIMIZER_POSTFIX,
            **shared_kwargs
        }
        G_optimizer_kwargs = {
            MODEL_ARG: self.G_optimizer._optimizer if self.use_mixed_precision else self.G_optimizer,
            MODEL_TYPE_ARG: GENERATOR_NAME + OPTIMIZER_POSTFIX,
            **shared_kwargs
        }

        if self.use_mixed_precision:
            self.D_optimizer._optimizer = load_model(**D_optimizer_kwargs)
            self.G_optimizer._optimizer = load_model(**G_optimizer_kwargs)
        else:
            self.D_optimizer = load_model(**D_optimizer_kwargs)
            self.G_optimizer = load_model(**G_optimizer_kwargs)

        optimizer_steps = self.scheduler.compute_optimizer_steps(res, stage)
        set_optimizer_iters(self.D_optimizer, optimizer_steps)
        set_optimizer_iters(self.G_optimizer, optimizer_steps)

        # TODO: for debugging, remove later
        print('D opt iters:', self.D_optimizer.iterations)
        print('G opt iters:', self.G_optimizer.iterations)

    def create_images_datasets(self, res=None, stage=None):
        start_time = time.time()
        logging.info('Initializing images datasets...')

        if res is None:
            start_res = self.start_resolution_log2
            end_res = self.resolution_log2
        else:
            start_res = res
            end_res = res

        self.images_datasets = {}
        for ds_res in tqdm(range(start_res, end_res + 1), desc='Dataset res'):
            self.images_datasets[ds_res] = self.create_images_dataset(ds_res, None, messages=True)

        self.final_images_dataset = None
        if (res == self.resolution_log2 and stage == STABILIZATION_MODE) or (res is None):
            self.final_images_dataset = self.create_images_dataset(self.resolution_log2, STABILIZATION_MODE, messages=True)

        total_time = time.time() - start_time
        logging.info(f'Images datasets initialized in {total_time:.3f} seconds!')

    def create_images_dataset(self, res, stage, messages=True):
        if messages:
            start_time = time.time()
            logging.info(f'Initializing images dataset for {2**res}x{2**res} model...')

        # No caching by default
        cache = False
        if self.dataset_max_cache_res is not None:
            if res <= self.dataset_max_cache_res:
                cache = os.path.join(self.storage_path or '', DATASET_CACHE_DIR, self.model_name)
                os.makedirs(cache, exist_ok=True)

        batch_size = self.get_batch_size(res, stage)
        res_kwargs = {'res': res, 'batch_size': batch_size, 'cache': cache}
        images_dataset = create_training_dataset(**{**res_kwargs, **self.dataset_params})

        if messages:
            total_time = time.time() - start_time
            logging.info(f'Images dataset initialized in {total_time:.3f} seconds!')

        return images_dataset

    def get_images_dataset(self, res, stage):
        if res == self.resolution_log2 and stage == STABILIZATION_MODE:
            return iter(self.final_images_dataset)
        else:
            return iter(self.images_datasets[res])

    def create_models(self, res, mode, load_prev_stage_weights=False):
        # All models should be initialized before calling this function
        D_model = self.D_object.create_D_model(res, mode=mode)
        G_model = self.G_object.create_G_model(res, mode=mode)
        if self.use_Gs:
            Gs_model = self.Gs_object.create_G_model(res, mode=mode)
            Gs_model.set_weights(G_model.get_weights())
        else:
            Gs_model = None

        # Log D model
        D_model.summary(print_fn=logging.info)
        # Log G model (mapping and synthesis networks)
        self.G_object.G_mapping.summary(print_fn=logging.info)
        self.G_object.G_synthesis.summary(print_fn=logging.info)
        G_model.summary(print_fn=logging.info)

        if load_prev_stage_weights:
            load_res, load_mode = self.scheduler.get_previous_res_and_stage(res, mode)
            D_model, G_model, Gs_model = self.load_trained_models(
                {D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model}, res=load_res, mode=load_mode
            )

        return D_model, G_model, Gs_model

    def update_models_weights(self, models):
        self.D_object.save_D_weights_in_class(models[D_KEY])
        self.G_object.save_G_weights_in_class(models[G_KEY])
        if self.use_Gs:
            self.Gs_object.save_G_weights_in_class(models[GS_KEY])

    def load_models_from_class(self, models):
        D_model = self.D_object.load_D_weights_from_class(models[D_KEY])
        G_model = self.G_object.load_G_weights_from_class(models[G_KEY])
        if self.use_Gs:
            Gs_model = self.Gs_object.load_G_weights_from_class(models[GS_KEY])
        else:
            Gs_model = None
        return D_model, G_model, Gs_model

    def load_trained_models(self, models, res, mode):
        logging.info(f'Loading models for res={res} and mode={STABILIZATION_MODE}...')

        D_model, G_model, Gs_model = models[D_KEY], models[G_KEY], models[GS_KEY]
        step = self.scheduler.get_stage_end_processed_images(res, mode)

        # TODO: for debugging, remove later
        # print('\nD stats after init:')
        # show_vars_stats(D_model.trainable_variables)
        # print('\nG stats after init:')
        # show_vars_stats(G_model.trainable_variables)

        D_model = load_model(
            D_model, self.model_name, DISCRIMINATOR_NAME,
            res=res, stage=mode, step=step, storage_path=self.storage_path
        )
        G_model = load_model(
            G_model, self.model_name, GENERATOR_NAME,
            res=res, stage=mode, step=step, storage_path=self.storage_path
        )
        if Gs_model is not None:
            Gs_model = load_model(
                Gs_model, self.model_name, GENERATOR_NAME + SMOOTH_POSTFIX,
                res=res, stage=mode, step=step, storage_path=self.storage_path
            )

        # TODO: for debugging, remove later
        # print('\nD stats after loading:')
        # show_vars_stats(D_model.trainable_variables)
        # print('\nG stats after loading:')
        # show_vars_stats(G_model.trainable_variables)

        logging.info(f'Loaded model weights from res={res}, mode={mode}')
        return D_model, G_model, Gs_model

    def save_models(self, models, res, mode, step):
        D_model, G_model, Gs_model = models[D_KEY], models[G_KEY], models[GS_KEY]
        shared_kwargs = {
            'model_name': self.model_name,
            'res': res,
            'stage': mode,
            'step': step,
            'storage_path': self.storage_path
        }
        save_model(model=D_model, model_type=DISCRIMINATOR_NAME, **shared_kwargs)
        save_model(model=G_model, model_type=GENERATOR_NAME, **shared_kwargs)
        if Gs_model is not None:
            save_model(model=Gs_model, model_type=GENERATOR_NAME + SMOOTH_POSTFIX, **shared_kwargs)

    def save_optimizers_weights(self, res, stage, step=None):
        shared_kwargs = {
            'model_name': self.model_name,
            'res': res,
            'stage': stage,
            'step': step,
            'storage_path': self.storage_path
        }

        MODEL_ARG = 'model'
        MODEL_TYPE_ARG = 'model_type'

        D_model_type = DISCRIMINATOR_NAME + OPTIMIZER_POSTFIX
        G_model_type = GENERATOR_NAME + OPTIMIZER_POSTFIX

        D_optimizer_kwargs = {
            MODEL_ARG: self.D_optimizer._optimizer if self.use_mixed_precision else self.D_optimizer,
            MODEL_TYPE_ARG: D_model_type,
            **shared_kwargs
        }
        G_optimizer_kwargs = {
            MODEL_ARG: self.G_optimizer._optimizer if self.use_mixed_precision else self.G_optimizer,
            MODEL_TYPE_ARG: G_model_type,
            **shared_kwargs
        }

        save_model(**D_optimizer_kwargs)
        save_model(**G_optimizer_kwargs)

        use_fp16 = should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)
        if use_fp16:
            OPTIMIZER_ARG = 'optimizer'
            save_optimizer_loss_scale(**{OPTIMIZER_ARG: self.D_optimizer, MODEL_TYPE_ARG: D_model_type, **shared_kwargs})
            save_optimizer_loss_scale(**{OPTIMIZER_ARG: self.G_optimizer, MODEL_TYPE_ARG: G_model_type, **shared_kwargs})

    def save_valid_images(self, models, training_finished_images, res, stage, smoothed=False):
        G_model, Gs_model = models[G_KEY], models[GS_KEY]
        dir_stage = stage
        if smoothed:
            dir_stage += SMOOTH_POSTFIX

        digits_in_number = 8 # Total number of training images is 25000k for resolution 1024
        fname = ('%0' + str(digits_in_number) + 'd') % training_finished_images

        valid_images_dir = create_images_dir_path(self.model_name, res, dir_stage)
        use_grid_title = False
        if use_grid_title:
            valid_images_grid_title = create_images_grid_title(res, dir_stage, training_finished_images)
        else:
            valid_images_grid_title = None

        model_kwargs = {'training': False}
        batch_size = 2 * self.get_batch_size(res, stage)
        if smoothed:
            valid_images = run_model_on_batches(Gs_model, model_kwargs, self.Gs_valid_latents, batch_size)
            if not self.use_gpu_for_Gs:
                valid_images = tf.transpose(valid_images, toNCHW_AXIS)
        else:
            valid_images = run_model_on_batches(G_model, model_kwargs, self.valid_latents, batch_size)
        valid_images = convert_outputs_to_images(
            valid_images, max(2 ** res, self.min_target_single_image_size),
            hw_ratio=self.dataset_hw_ratio, data_format=self.data_format
        ).numpy()

        save_in_jpg = res > self.max_png_res
        fast_save_grid(
            out_dir=valid_images_dir,
            fname=fname,
            images=valid_images,
            title=valid_images_grid_title,
            nrows=self.valid_grid_nrows,
            ncols=self.valid_grid_ncols,
            padding=self.valid_grid_padding,
            save_in_jpg=save_in_jpg
        )

    def smooth_crossfade_images(self, images, alpha):
        # TODO: thing about compiling this function. Alpha must be a tensor
        return smooth_crossfade_images(images, alpha, self.data_format)

    @tf.function
    def generate_latents(self, batch_size):
        return generate_latents(batch_size, self.z_dim, self.compute_dtype)

    @tf.function
    def G_train_step(self, G_model, D_model, latents, write_scalars_summary, write_hists_summary, step):
        G_vars = G_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(G_vars, 'Generator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as G_tape:
            G_tape.watch(G_vars)
            G_loss = self.G_loss_fn(G_model, D_model, self.G_optimizer,
                                    latents=latents,
                                    write_summary=write_scalars_summary,
                                    step=step,
                                    **self.G_loss_params)
            G_loss = maybe_scale_loss(G_loss, self.G_optimizer)

        G_grads = G_tape.gradient(G_loss, G_vars)
        G_grads = maybe_unscale_grads(G_grads, self.G_optimizer)
        self.G_optimizer.apply_gradients(zip(G_grads, G_vars))

        # TODO: for debugging, remove later
        #for grad, var in zip(G_grads, G_vars):
            #nans = tf.math.count_nonzero(~tf.math.is_finite(grad))
            #nums = tf.math.count_nonzero(tf.math.is_finite(grad))
            # tf.print('G, step =', step, f'{var.name}: nans =', nans, ',', tf.math.round(100 * nans / (nans + nums)), '%')

        # self.process_hists(G_model, G_grads, 'G', write_hists_summary, step)

        return G_grads

    @tf.function
    def D_train_step(self, G_model, D_model, latents, images, write_scalars_summary, write_hists_summary, step):
        D_vars = D_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(D_vars, 'Discriminator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as D_tape:
            D_tape.watch(D_vars)
            D_loss = self.D_loss_fn(G_model, D_model, self.D_optimizer,
                                    latents=latents,
                                    real_images=images,
                                    write_summary=write_scalars_summary,
                                    step=step,
                                    **self.D_loss_params)
            D_loss = maybe_scale_loss(D_loss, self.D_optimizer)

        D_grads = D_tape.gradient(D_loss, D_vars)
        D_grads = maybe_unscale_grads(D_grads, self.D_optimizer)
        self.D_optimizer.apply_gradients(zip(D_grads, D_vars))

        # TODO: for debugging, remove later
        #for grad, var in zip(D_grads, D_vars):
            #nans = tf.math.count_nonzero(~tf.math.is_finite(grad))
            #nums = tf.math.count_nonzero(tf.math.is_finite(grad))
            # tf.print('D, step =', step, f'{var.name}: nans =', nans, ',', tf.math.round(100 * nans / (nans + nums)), '%')

        # self.process_hists(D_model, D_grads, 'D', write_hists_summary, step)

        return D_grads

    def process_hists(self, model, grads, model_name, write_hists_summary, step):
        if write_hists_summary:
            print('Writing hists summaries...')
            start_time = time.time()

        # Note: it's important to have cond for summaries after name scope definition,
        # otherwise all hists will have the same prefix, e.g. 'cond1'.
        # It holds at least for summaries inside tf.function
        vars = model.trainable_variables
        with tf.device(CPU_DEVICE):
            # Write gradients
            with tf.name_scope(f'{model_name}-grads'):
                if write_hists_summary:
                    for grad, var in zip(grads, vars):
                        hist_grad = tf.cond(is_finite_grad(grad), lambda: grad, lambda: tf.zeros(grad.shape, grad.dtype))
                        tf.summary.histogram(var.name, hist_grad, step=step)
            # Write weights
            with tf.name_scope(f'{model_name}-weights'):
                if write_hists_summary:
                    for var in vars:
                        tf.summary.histogram(var.name, var, step=step)

        if write_hists_summary:
            total_time = time.time() - start_time
            print(f'Hists written in {total_time:.3f} seconds')

    def train_step(self, G_model, D_model, G_latents, D_latents,
                   images, write_scalars_summary, write_hists_summary, step):
        # Note: explicit use of G and D models allows to make sure that
        # tf.function doesn't compile models (can they be?). Additionally tracing is used
        # (previously for res=3 and mode=transition G model used variables only from res=2)
        D_grads = self.D_train_step(G_model, D_model, D_latents, images, write_scalars_summary, write_hists_summary, step)
        G_grads = self.G_train_step(G_model, D_model, G_latents, write_scalars_summary, write_hists_summary, step)

        # Note: processing hists summaries inside train_step functions leads to a process crash for large resolutions.
        # Maybe due to compilation. To write hists to TensorBoard they need to be processed separately.
        # This also allows larger batch sizes. Rear OOM warnings don't seem to affect performance
        self.process_hists(D_model, D_grads, 'D', write_hists_summary, step)
        self.process_hists(G_model, G_grads, 'G', write_hists_summary, step)

    def add_resources_summary(self, training_finished_images):
        for device, memory_stats in get_gpu_memory_usage().items():
            for k, v in memory_stats.items():
                tf.summary.scalar(f'Resources/{device}/{k}(Mbs)', v, step=training_finished_images)

    def add_timing_summary(self, training_finished_images):
        # 1. Get time info and update last used value
        # One tick is number of images after each scalar summaries is updated
        cur_update_time = time.time()
        tick_time = cur_update_time - self.last_update_time
        self.last_update_time = cur_update_time
        kimg_denom = self.summary_scalars_every / 1000
        total_time = cur_update_time - self.start_time
        # 2. Summary tick time
        # Note: picks when metrics are evaluated and on first call (graph compilation)
        tf.summary.scalar(f'Timing/Tick(s)', tick_time, step=training_finished_images)
        tf.summary.scalar(f'Timing/Kimg(s)', tick_time / kimg_denom, step=training_finished_images)
        # 3. Summary total time
        tf.summary.scalar(f'Timing/Total(hours)', total_time / (60.0 * 60.0), step=training_finished_images)
        tf.summary.scalar(f'Timing/Total(days)', total_time / (24.0 * 60.0 * 60.0), step=training_finished_images)

    def post_train_step_actions(self, models, res, mode, summary_options, summary_writer):
        D_model, G_model, Gs_model = models[D_KEY], models[G_KEY], models[GS_KEY]
        # Adjust number of processed images if method is called for last step
        last_step_cond           = summary_options[LAST_STEP_COND_KEY]
        training_finished_images = summary_options[TRAINING_FINISHED_IMAGES_KEY]
        if last_step_cond:
            training_finished_images = self.scheduler.get_stage_end_processed_images(res, mode)

        if summary_options[SMOOTH_G_WEIGHTS_KEY]:
            if Gs_model is not None:
                smooth_model_weights(
                    sm_model=Gs_model, src_model=G_model, beta=self.get_Gs_beta(res, mode), device=self.Gs_device
                )

        if summary_options[RUN_METRICS_KEY]:
            self.run_metrics(res, mode, training_finished_images)

        if summary_options[WRITE_LOSS_SCALE_SUMMARY_KEY]:
            tf.summary.scalar('LossScale/D_optimizer', self.D_optimizer.loss_scale, step=training_finished_images)
            tf.summary.scalar('LossScale/G_optimizer', self.G_optimizer.loss_scale, step=training_finished_images)

        # TODO: think how to flush summaries outside of this function
        if summary_options[WRITE_SCALARS_SUMMARY_KEY]:
            self.add_resources_summary(training_finished_images)
            self.add_timing_summary(training_finished_images)
            summary_writer.flush()

        if summary_options[SAVE_MODELS_KEY]:
            self.save_models(models=models, res=res, mode=mode, step=training_finished_images)
            if self.is_last_stage(res, mode):
                self.save_optimizers_weights(res=res, stage=mode, step=training_finished_images)

        if summary_options[SAVE_VALID_IMAGES_KEY]:
            self.save_valid_images(models, training_finished_images, res=res, stage=mode)
            if Gs_model is not None:
                self.save_valid_images(models, training_finished_images, res=res, stage=mode, smoothed=True)

    def summary_options(self, step, stage_steps, n_finished_images, batch_size):
        first_step_cond          = step == 0
        last_step_cond           = is_last_step(step, stage_steps)
        stage_images             = (step + 1) * batch_size
        write_scalars_summary    = should_write_summary(self.summary_scalars_every, stage_images, batch_size) or last_step_cond
        write_loss_scale_summary = self.use_mixed_precision and write_scalars_summary and (not first_step_cond) # The first step usually uses very high scale

        # TODO: should summaries use stage_images or training_finished_images?
        return {
            FIRST_STEP_COND_KEY         : first_step_cond,
            LAST_STEP_COND_KEY          : last_step_cond,
            STAGE_IMAGES_KEY            : stage_images,
            TRAINING_FINISHED_IMAGES_KEY: stage_images + n_finished_images,
            WRITE_LOSS_SCALE_SUMMARY_KEY: write_loss_scale_summary,
            WRITE_SCALARS_SUMMARY_KEY   : write_scalars_summary,
            WRITE_HISTS_SUMMARY_KEY     : should_write_summary(self.summary_hists_every, stage_images, batch_size) or last_step_cond,
            RUN_METRICS_KEY             : should_write_summary(self.run_metrics_every, stage_images + n_finished_images, batch_size) or last_step_cond,
            SAVE_MODELS_KEY             : should_write_summary(self.save_model_every, stage_images + n_finished_images, batch_size) or last_step_cond,
            SAVE_VALID_IMAGES_KEY       : should_write_summary(self.save_images_every, stage_images + n_finished_images, batch_size) or last_step_cond,
            SMOOTH_G_WEIGHTS_KEY        : self.use_Gs and (not first_step_cond) # For the first step optimizers learning rates are zeros
        }

    def init_training_time(self):
        self.start_time = time.time()
        self.last_update_time = time.time()

    def train(self):
        # TODO: refactor this function, and make it consistent with training for each separate stage
        self.init_training_time()

        tf_step                  = tf.Variable(0, trainable=False, dtype=tf.int64)
        tf_write_scalars_summary = tf.Variable(True, trainable=False, dtype=tf.bool)
        tf_write_hists_summary   = tf.Variable(True, trainable=False, dtype=tf.bool)

        for res in tqdm(range(self.start_resolution_log2, self.resolution_log2 + 1), desc='Training res'):
            logging.info(f'Training {2**res}x{2**res} model...')
            res_start_time = time.time()

            if self.reset_opt_state_for_new_lod:
                self.reset_optimizers_state()
            self.adjust_optimizers_learning_rate(res)

            summary_writer      = self.summary_writers[res]
            n_finished_images   = self.scheduler.get_stage_start_processed_images(res, TRANSITION_MODE)
            transition_steps    = self.scheduler.get_n_steps_for_stage(res, TRANSITION_MODE)
            stabilization_steps = self.scheduler.get_n_steps_for_stage(res, STABILIZATION_MODE)

            with summary_writer.as_default():
                # The first resolution doesn't use alpha parameter,
                # but has usual number of steps for stabilization phase
                if res > self.start_resolution_log2:
                    # Transition stage
                    transition_stage_start_time = time.time()

                    if self.clear_session_for_new_model:
                        logging.info('Clearing session...')
                        tf.keras.backend.clear_session()

                    D_model, G_model, Gs_model = self.create_models(res, mode=TRANSITION_MODE)
                    D_model, G_model, Gs_model = self.load_models_from_class({D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model})

                    images_dataset = self.get_images_dataset(res, TRANSITION_MODE)
                    batch_size = self.get_batch_size(res, TRANSITION_MODE)
                    tf_step.assign(n_finished_images)
                    tf_write_scalars_summary.assign(True)
                    tf_write_hists_summary.assign(True)

                    desc = f'{2**res}x{2**res} model, transition steps'
                    for step in tqdm(range(transition_steps), desc=desc):
                        summary_options = self.summary_options(step, transition_steps, n_finished_images, batch_size)
                        tf_write_scalars_summary.assign(summary_options[WRITE_SCALARS_SUMMARY_KEY])
                        tf_write_hists_summary.assign(summary_options[WRITE_HISTS_SUMMARY_KEY])
                        tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                        if step % self.batch_repeats == 0:
                            alpha = compute_alpha(step, transition_steps)
                            D_model = update_wsum_alpha(D_model, alpha)
                            G_model = update_wsum_alpha(G_model, alpha)
                            if Gs_model is not None:
                                Gs_model = update_wsum_alpha(Gs_model, alpha)

                        if summary_options[WRITE_SCALARS_SUMMARY_KEY]:
                            tf.summary.scalar('Alpha', alpha, step=summary_options[TRAINING_FINISHED_IMAGES_KEY])

                        G_latents = self.generate_latents(batch_size)
                        D_latents = self.generate_latents(batch_size)
                        real_images = self.smooth_crossfade_images(next(images_dataset), alpha)
                        self.train_step(
                            G_model=G_model, D_model=D_model,
                            G_latents=G_latents, D_latents=D_latents, images=real_images,
                            write_scalars_summary=tf_write_scalars_summary, write_hists_summary=tf_write_hists_summary, step=tf_step
                        )
                        self.post_train_step_actions(
                            models={D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model},
                            res=res, mode=TRANSITION_MODE, summary_options=summary_options, summary_writer=summary_writer
                        )

                    self.update_models_weights({D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model})
                    remove_old_models(
                        self.model_name, res=res, stage=TRANSITION_MODE,
                        max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
                    )

                    transition_stage_total_time = time.time() - transition_stage_start_time
                    logging.info(f'Transition stage took {format_time(transition_stage_total_time)}')

                # Stabilization stage
                stabilization_stage_start_time = time.time()

                if self.clear_session_for_new_model:
                    logging.info('Clearing session...')
                    tf.keras.backend.clear_session()

                D_model, G_model, Gs_model = self.create_models(res, mode=STABILIZATION_MODE)
                D_model, G_model, Gs_model = self.load_models_from_class({D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model})

                n_finished_images += transition_steps * batch_size if res > self.start_resolution_log2 else 0
                images_dataset = self.get_images_dataset(res, STABILIZATION_MODE)
                batch_size = self.get_batch_size(res, STABILIZATION_MODE)
                tf_step.assign(n_finished_images)
                tf_write_scalars_summary.assign(True)
                tf_write_hists_summary.assign(True)

                desc = f'{2**res}x{2**res} model, stabilization steps'
                for step in tqdm(range(stabilization_steps), desc=desc):
                    summary_options = self.summary_options(step, stabilization_steps, n_finished_images, batch_size)
                    tf_write_scalars_summary.assign(summary_options[WRITE_SCALARS_SUMMARY_KEY])
                    tf_write_hists_summary.assign(summary_options[WRITE_HISTS_SUMMARY_KEY])
                    tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                    G_latents = self.generate_latents(batch_size)
                    D_latents = self.generate_latents(batch_size)
                    real_images = next(images_dataset)
                    self.train_step(
                        G_model=G_model, D_model=D_model,
                        G_latents=G_latents, D_latents=D_latents, images=real_images,
                        write_scalars_summary=tf_write_scalars_summary, write_hists_summary=tf_write_hists_summary, step=tf_step
                    )
                    self.post_train_step_actions(
                        models={D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model},
                        res=res, mode=STABILIZATION_MODE, summary_options=summary_options, summary_writer=summary_writer
                    )

                self.update_models_weights({D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model})
                remove_old_models(
                    self.model_name, res=res, stage=STABILIZATION_MODE,
                    max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
                )

                stabilization_stage_total_time = time.time() - stabilization_stage_start_time
                logging.info(f'Stabilization stage took {format_time(stabilization_stage_total_time)}')

                res_total_time = time.time() - res_start_time
                logging.info(f'Training of {2**res}x{2**res} model took {format_time(res_total_time)}')
                logging.info(f'----------------------------------------------------------------------')
                logging.info('')

        train_total_time = time.time() - self.start_time
        logging.info(f'Training finished in {format_time(train_total_time)}!')

    def run_transition_stage(self, res):
        self.init_training_time()

        D_model, G_model, Gs_model = self.create_models(res, mode=TRANSITION_MODE, load_prev_stage_weights=True)
        self.zero_optimizers_learning_rate()

        images_dataset           = self.get_images_dataset(res, TRANSITION_MODE)
        batch_size               = self.get_batch_size(res, TRANSITION_MODE)
        summary_writer           = self.summary_writers[res]
        n_finished_images        = self.scheduler.get_stage_start_processed_images(res, TRANSITION_MODE)
        transition_steps         = self.scheduler.get_n_steps_for_stage(res, TRANSITION_MODE)
        tf_step                  = tf.Variable(n_finished_images, trainable=False, dtype=tf.int64)
        tf_write_scalars_summary = tf.Variable(True, trainable=False, dtype=tf.bool)
        tf_write_hists_summary   = tf.Variable(True, trainable=False, dtype=tf.bool)

        with summary_writer.as_default():
            desc = f'{2**res}x{2**res} model, transition steps'
            for step in tqdm(range(transition_steps), desc=desc):
                summary_options = self.summary_options(step, transition_steps, n_finished_images, batch_size)
                tf_write_scalars_summary.assign(summary_options[WRITE_SCALARS_SUMMARY_KEY])
                tf_write_hists_summary.assign(summary_options[WRITE_HISTS_SUMMARY_KEY])
                tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                if step % self.batch_repeats == 0:
                    alpha = compute_alpha(step, transition_steps)
                    D_model = update_wsum_alpha(D_model, alpha)
                    G_model = update_wsum_alpha(G_model, alpha)
                    if Gs_model is not None:
                        Gs_model = update_wsum_alpha(Gs_model, alpha)

                if summary_options[WRITE_SCALARS_SUMMARY_KEY]:
                    tf.summary.scalar('Alpha', alpha, step=summary_options[TRAINING_FINISHED_IMAGES_KEY])

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                real_images = self.smooth_crossfade_images(next(images_dataset), alpha)
                self.train_step(
                    G_model=G_model, D_model=D_model,
                    G_latents=G_latents, D_latents=D_latents, images=real_images,
                    write_scalars_summary=tf_write_scalars_summary, write_hists_summary=tf_write_hists_summary, step=tf_step
                )

                if summary_options[FIRST_STEP_COND_KEY]:
                    if not self.reset_opt_state_for_new_lod:
                        self.restore_optimizers_state(res - 1, STABILIZATION_MODE)
                    # Always adjust learning rates
                    self.adjust_optimizers_learning_rate(res)

                self.post_train_step_actions(
                    models={D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model},
                    res=res, mode=TRANSITION_MODE, summary_options=summary_options, summary_writer=summary_writer
                )

        remove_old_models(
            self.model_name, res=res, stage=TRANSITION_MODE,
            max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        # Save states after extra weights are removed
        self.save_optimizers_weights(res, stage=TRANSITION_MODE)

        transition_stage_total_time = time.time() - self.start_time
        logging.info(f'Transition stage took {format_time(transition_stage_total_time)}')

    def run_stabilization_stage(self, res):
        self.init_training_time()

        load_pres_stage_weights = res > self.start_resolution_log2
        D_model, G_model, Gs_model = self.create_models(res, mode=STABILIZATION_MODE, load_prev_stage_weights=load_pres_stage_weights)
        self.zero_optimizers_learning_rate()

        images_dataset           = self.get_images_dataset(res, STABILIZATION_MODE)
        batch_size               = self.get_batch_size(res, STABILIZATION_MODE)
        summary_writer           = self.summary_writers[res]
        n_finished_images        = self.scheduler.get_stage_start_processed_images(res, STABILIZATION_MODE)
        stabilization_steps      = self.scheduler.get_n_steps_for_stage(res, STABILIZATION_MODE)
        tf_step                  = tf.Variable(n_finished_images, trainable=False, dtype=tf.int64)
        tf_write_scalars_summary = tf.Variable(True, trainable=False, dtype=tf.bool)
        tf_write_hists_summary   = tf.Variable(True, trainable=False, dtype=tf.bool)

        with summary_writer.as_default():
            desc = f'{2**res}x{2**res} model, stabilization steps'
            for step in tqdm(range(stabilization_steps), desc=desc):
                summary_options = self.summary_options(step, stabilization_steps, n_finished_images, batch_size)
                tf_write_scalars_summary.assign(summary_options[WRITE_SCALARS_SUMMARY_KEY])
                tf_write_hists_summary.assign(summary_options[WRITE_HISTS_SUMMARY_KEY])
                tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                real_images = next(images_dataset)
                self.train_step(
                    G_model=G_model, D_model=D_model,
                    G_latents=G_latents, D_latents=D_latents, images=real_images,
                    write_scalars_summary=tf_write_scalars_summary, write_hists_summary=tf_write_hists_summary, step=tf_step
                )

                if summary_options[FIRST_STEP_COND_KEY]:
                    if res > self.start_resolution_log2:
                        self.restore_optimizers_state(res, stage=TRANSITION_MODE)
                    # Always adjust learning rates
                    self.adjust_optimizers_learning_rate(res)

                self.post_train_step_actions(
                    models={D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model},
                    res=res, mode=STABILIZATION_MODE, summary_options=summary_options, summary_writer=summary_writer
                )

        remove_old_models(
            self.model_name, res=res, stage=STABILIZATION_MODE,
            max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        # Save states after extra weights are removed
        self.save_optimizers_weights(res, stage=STABILIZATION_MODE)

        stabilization_stage_total_time = time.time() - self.start_time
        logging.info(f'Stabilization stage took {format_time(stabilization_stage_total_time)}')

    def run_metrics(self, res, mode, training_finished_images, summary_writer=None):
        if self.use_Gs:
            G_model = self.Gs_object.create_G_model(res, mode)
        else:
            G_model = self.G_object.create_G_model(res, mode)

        batch_size = self.get_batch_size(res, mode)
        if summary_writer is None:
            summary_writer = self.summary_writers[res]
        with summary_writer.as_default():
            metrics_start_time = time.time()
            for idx, metric_object in enumerate(self.metrics_objects):
                metric_name = metric_object.name

                start_time = time.time()
                metric_value = metric_object.run_metric(batch_size, G_model)
                total_time = time.time() - start_time

                tf.summary.scalar(f'Metric/{metric_name}', metric_value, step=training_finished_images)
                tf.summary.scalar(f'Metric/{metric_name}/Time(s)', total_time, step=training_finished_images)
                summary_writer.flush()
                logging.info(f'Evaluated {metric_name} metric in {format_time(total_time)}')

            metrics_total_time = time.time() - metrics_start_time
            tf.summary.scalar(f'Metric/TotalRunTime/Time(s)', metrics_total_time, step=training_finished_images)
            summary_writer.flush()

    def run_benchmark_stage(self, res, mode, images, run_metrics):
        stage_start_time = time.time()

        D_model, G_model, Gs_model = self.create_models(res, mode=mode)
        self.adjust_optimizers_learning_rate(res)

        images_dataset           = self.get_images_dataset(res, mode)
        batch_size               = self.get_batch_size(res, mode)
        benchmark_steps          = images // batch_size
        n_finished_images        = 0
        tf_step                  = tf.Variable(n_finished_images, trainable=False, dtype=tf.int64)
        # TODO: usage of a tensor for hists leads to a process crash
        tf_write_scalars_summary = tf.Variable(True, trainable=False, dtype=tf.bool)
        tf_write_hists_summary   = tf.Variable(True, trainable=False, dtype=tf.bool)

        # TODO: add temp summary writer and remove logs after execution
        benchmark_dir = os.path.join(TF_LOGS_DIR, 'temp_dir')
        summary_writer = tf.summary.create_file_writer(benchmark_dir)
        with summary_writer.as_default():
            desc = f'Benchmark {2**res}x{2**res} model, {mode} steps'
            for step in tqdm(range(benchmark_steps), desc=desc):
                # Note: on the 1st step model is compiled, so don't count this time
                if step == 1:
                    stage_start_time = time.time()
                    metrics_time = 0.

                summary_options = self.summary_options(step, benchmark_steps, n_finished_images, batch_size)
                tf_write_scalars_summary.assign(summary_options[WRITE_SCALARS_SUMMARY_KEY])
                tf_write_hists_summary.assign(summary_options[WRITE_HISTS_SUMMARY_KEY])
                # tf_write_scalars_summary = summary_options[WRITE_SCALARS_SUMMARY_KEY]
                # tf_write_hists_summary = False #summary_options[WRITE_HISTS_SUMMARY_KEY]
                tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                if mode == TRANSITION_MODE:
                    if step % self.batch_repeats == 0:
                        alpha = compute_alpha(step, benchmark_steps)
                        D_model = update_wsum_alpha(D_model, alpha)
                        G_model = update_wsum_alpha(G_model, alpha)
                        if Gs_model is not None:
                            Gs_model = update_wsum_alpha(Gs_model, alpha)

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                real_images = next(images_dataset)
                if mode == TRANSITION_MODE:
                    real_images = self.smooth_crossfade_images(real_images, alpha)
                self.train_step(
                    G_model=G_model, D_model=D_model,
                    G_latents=G_latents, D_latents=D_latents, images=real_images,
                    write_scalars_summary=tf_write_scalars_summary, write_hists_summary=tf_write_hists_summary, step=tf_step
                )

                # Run metrics twice to make sure everything is fine
                if run_metrics and (step == 50 or step == 100):
                    metrics_start_time = time.time()
                    self.run_metrics(res, mode, -1, summary_writer)
                    metrics_time += time.time() - metrics_start_time

        shutil.rmtree(benchmark_dir)

        stage_total_time = time.time() - stage_start_time
        train_time = stage_total_time - metrics_time
        print(f'\nBenchmark finished in {format_time(stage_total_time)}. '
              f'Metrics run (2 iterations) in {format_time(metrics_time)}.\n'
              f'Training took {format_time(train_time)} for {images} images. In average {(images / train_time):.3f} images/sec.')

    def run_train_stage(self, res, mode):
        assert self.start_resolution_log2 <= res <= self.resolution_log2
        if mode == STABILIZATION_MODE:
            self.run_stabilization_stage(res)
        elif mode == TRANSITION_MODE:
            assert res > self.start_resolution_log2
            self.run_transition_stage(res)
        else:
            assert False, f'Train stage must be one of f[{STABILIZATION_MODE}, {TRANSITION_MODE}]'
