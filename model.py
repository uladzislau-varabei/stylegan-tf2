import os
import logging
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from losses import select_G_loss_fn, select_D_loss_fn
from utils import compute_alpha, update_wsum_alpha,\
    save_model, load_model,\
    TRANSITION_MODE, STABILIZATION_MODE, SMOOTH_POSTFIX, OPTIMIZER_POSTFIX,\
    create_images_dir_name, create_images_grid_title, remove_old_models,\
    format_time, to_int_dict, validate_data_format, trace_vars, trace_message, mult_by_zero, is_last_step,\
    maybe_scale_loss, maybe_unscale_grads, is_optimizer_ready, set_optimizer_iters, load_images_paths, set_tf_logging
from utils import TARGET_RESOLUTION, START_RESOLUTION, LATENT_SIZE, USE_MIXED_PRECISION,\
    DATA_FORMAT, MODEL_NAME, MAX_MODELS_TO_KEEP, \
    SUMMARY_EVERY, SAVE_MODEL_EVERY, SAVE_IMAGES_EVERY,\
    G_LOSS_FN, D_LOSS_FN, G_LOSS_FN_PARAMS, D_LOSS_FN_PARAMS,\
    G_LEARNING_RATE, D_LEARNING_RATE, \
    G_LEARNING_RATE_DICT, D_LEARNING_RATE_DICT,\
    ADAM_BETA1, ADAM_BETA2, RESET_OPT_STATE_FOR_NEW_LOD,\
    USE_G_SMOOTHING, G_SMOOTHING_BETA, G_SMOOTHING_BETA_KIMAGES, USE_GPU_FOR_GS,\
    GENERATOR_NAME, DISCRIMINATOR_NAME,\
    NCHW_FORMAT, NHWC_FORMAT,\
    TF_LOGS_DIR, STORAGE_PATH, DATASET_CACHE_FOLDER,\
    BATCH_SIZES, BATCH_REPEATS, DATASET_MAX_CACHE_RES,\
    TOTAL_KIMAGES, TRANSITION_KIMAGES, TRANSITION_KIMAGES_DICT, STABILIZATION_KIMAGES, STABILIZATION_KIMAGES_DICT,\
    DATASET_N_PARALLEL_CALLS, DATASET_N_PREFETCHED_BATCHES, DATASET_N_MAX_KIMAGES,\
    SHUFFLE_DATASET, MIRROR_AUGMENT,\
    VALID_GRID_NROWS, VALID_GRID_NCOLS, VALID_MIN_TARGET_SINGLE_IMAGE_SIZE, VALID_MAX_PNG_RES,\
    TRAIN_MODE, INFERENCE_MODE, DEFAULT_MODE
from utils import DEFAULT_STORAGE_PATH, DEFAULT_MAX_MODELS_TO_KEEP,\
    DEFAULT_SUMMARY_EVERY, DEFAULT_SAVE_MODEL_EVERY, DEFAULT_SAVE_IMAGES_EVERY,\
    DEFAULT_BATCH_REPEATS, DEFAULT_G_LOSS_FN, DEFAULT_D_LOSS_FN, DEFAULT_G_LOSS_FN_PARAMS, DEFAULT_D_LOSS_FN_PARAMS,\
    DEFAULT_G_LEARNING_RATE, DEFAULT_D_LEARNING_RATE,\
    DEFAULT_G_LEARNING_RATE_DICT, DEFAULT_D_LEARNING_RATE_DICT,\
    DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2, DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD,\
    DEFAULT_DATASET_MAX_CACHE_RES,\
    DEFAULT_START_RESOLUTION, DEFAULT_USE_MIXED_PRECISION,\
    DEFAULT_DATASET_N_PARALLEL_CALLS,\
    DEFAULT_DATASET_N_PREFETCHED_BATCHES, DEFAULT_DATASET_N_MAX_KIMAGES,\
    DEFAULT_SHUFFLE_DATASET, DEFAULT_MIRROR_AUGMENT,\
    DEFAULT_TOTAL_KIMAGES, DEFAULT_TRANSITION_KIMAGES, DEFAULT_STABILIZATION_KIMAGES, \
    DEFAULT_TRANSITION_KIMAGES_DICT, DEFAULT_STABILIZATION_KIMAGES_DICT,\
    DEFAULT_USE_G_SMOOTHING, DEFAULT_G_SMOOTHING_BETA, DEFAULT_G_SMOOTHING_BETA_KIMAGES, DEFAULT_USE_GPU_FOR_GS,\
    DEFAULT_DATA_FORMAT,\
    DEFAULT_VALID_GRID_NROWS, DEFAULT_VALID_GRID_NCOLS,\
    DEFAULT_VALID_MIN_TARGET_SINGLE_IMAGE_SIZE, DEFAULT_VALID_MAX_PNG_RES
from dataloader_utils import create_training_dataset, convert_outputs_to_images
from image_utils import fast_save_grid
from networks import Generator, Discriminator


set_tf_logging(debug_mode=False)


MIXED_PRECISION_MAX_INIT_OPTIMIZER_ITERS = 20


"""
How to load model:
1) Create model
2) Initialize model
3) Load weights
"""


class StyleGAN:

    def __init__(self, config, mode=DEFAULT_MODE, images_paths=None, res=None, stage=None,
                 single_process_training=False):

        self.target_resolution = config[TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.latent_size = config[LATENT_SIZE]
        if self.data_format == NCHW_FORMAT:
            self.z_dim = (self.latent_size, 1, 1)
        elif self.data_format == NHWC_FORMAT:
            self.z_dim = (1, 1, self.latent_size)

        self.model_name = config[MODEL_NAME]

        # Training images and batches
        self.batch_sizes = to_int_dict(config[BATCH_SIZES])
        self.batch_repeats = config.get(BATCH_REPEATS, DEFAULT_BATCH_REPEATS)
        self.total_kimages = config.get(TOTAL_KIMAGES, DEFAULT_TOTAL_KIMAGES)
        self.transition_kimages = config.get(TRANSITION_KIMAGES, DEFAULT_TRANSITION_KIMAGES)
        self.transition_kimages_dict = to_int_dict(config.get(TRANSITION_KIMAGES_DICT, DEFAULT_TRANSITION_KIMAGES_DICT))
        self.stabilization_kimages = config.get(STABILIZATION_KIMAGES, DEFAULT_STABILIZATION_KIMAGES)
        self.stabilization_kimages_dict = to_int_dict(config.get(STABILIZATION_KIMAGES_DICT, DEFAULT_STABILIZATION_KIMAGES_DICT))

        # Computation
        self.use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
        self.compute_dtype = 'float16' if self.use_mixed_precision else 'float32'
        self.use_Gs = config.get(USE_G_SMOOTHING, DEFAULT_USE_G_SMOOTHING)
        self.use_gpu_for_Gs = config.get(USE_GPU_FOR_GS, DEFAULT_USE_GPU_FOR_GS)
        self.Gs_beta = config.get(G_SMOOTHING_BETA, DEFAULT_G_SMOOTHING_BETA)
        self.Gs_beta_kimgs = config.get(G_SMOOTHING_BETA_KIMAGES, DEFAULT_G_SMOOTHING_BETA_KIMAGES)
        # Resolution-specific betas for Gs
        self.Gs_betas = {}

        # Dataset
        self.shuffle_dataset = config.get(SHUFFLE_DATASET, DEFAULT_SHUFFLE_DATASET)
        self.dataset_n_parallel_calls = config.get(DATASET_N_PARALLEL_CALLS, DEFAULT_DATASET_N_PARALLEL_CALLS)
        self.dataset_n_prefetched_batches = config.get(DATASET_N_PREFETCHED_BATCHES, DEFAULT_DATASET_N_PREFETCHED_BATCHES)
        self.dataset_n_max_kimages = config.get(DATASET_N_MAX_KIMAGES, DEFAULT_DATASET_N_MAX_KIMAGES)
        self.dataset_max_cache_res = config.get(DATASET_MAX_CACHE_RES, DEFAULT_DATASET_MAX_CACHE_RES)
        self.mirror_augment = config.get(MIRROR_AUGMENT, DEFAULT_MIRROR_AUGMENT)

        # Losses
        self.G_loss_fn_name = config.get(G_LOSS_FN, DEFAULT_G_LOSS_FN)
        self.D_loss_fn_name = config.get(D_LOSS_FN, DEFAULT_D_LOSS_FN)
        self.G_loss_fn = select_G_loss_fn(self.G_loss_fn_name)
        self.D_loss_fn = select_D_loss_fn(self.D_loss_fn_name)
        self.G_loss_params = config.get(G_LOSS_FN_PARAMS, DEFAULT_G_LOSS_FN_PARAMS)
        self.D_loss_params = config.get(D_LOSS_FN_PARAMS, DEFAULT_D_LOSS_FN_PARAMS)

        # Optimizers options
        self.G_learning_rate = config.get(G_LEARNING_RATE, DEFAULT_G_LEARNING_RATE)
        self.D_learning_rate = config.get(D_LEARNING_RATE, DEFAULT_D_LEARNING_RATE)
        self.G_learning_rate_dict = to_int_dict(config.get(G_LEARNING_RATE_DICT, DEFAULT_G_LEARNING_RATE_DICT))
        self.D_learning_rate_dict = to_int_dict(config.get(D_LEARNING_RATE_DICT, DEFAULT_D_LEARNING_RATE_DICT))
        self.beta1 = config.get(ADAM_BETA1, DEFAULT_ADAM_BETA1)
        self.beta2 = config.get(ADAM_BETA2, DEFAULT_ADAM_BETA2)
        self.reset_opt_state_for_new_lod = config.get(RESET_OPT_STATE_FOR_NEW_LOD, DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD)

        # Valid images options
        self.valid_grid_nrows = config.get(VALID_GRID_NROWS, DEFAULT_VALID_GRID_NROWS)
        self.valid_grid_ncols = config.get(VALID_GRID_NCOLS, DEFAULT_VALID_GRID_NCOLS)
        self.valid_grid_padding = 2
        self.min_target_single_image_size = config.get(VALID_MIN_TARGET_SINGLE_IMAGE_SIZE, DEFAULT_VALID_MIN_TARGET_SINGLE_IMAGE_SIZE)
        if self.min_target_single_image_size < 0:
           self.min_target_single_image_size = max(2 ** (self.resolution_log2 - 1), 2 ** 7)
        self.max_png_res = config.get(VALID_MAX_PNG_RES, DEFAULT_VALID_MAX_PNG_RES)

        self.valid_latents = self.generate_latents(self.valid_grid_nrows * self.valid_grid_ncols)

        # Summaries
        self.storage_path = config.get(STORAGE_PATH, DEFAULT_STORAGE_PATH)
        self.max_models_to_keep = config.get(MAX_MODELS_TO_KEEP, DEFAULT_MAX_MODELS_TO_KEEP)
        self.summary_every = config.get(SUMMARY_EVERY, DEFAULT_SUMMARY_EVERY)
        self.save_model_every = config.get(SAVE_MODEL_EVERY, DEFAULT_SAVE_MODEL_EVERY)
        self.save_images_every = config.get(SAVE_IMAGES_EVERY, DEFAULT_SAVE_IMAGES_EVERY)
        self.logs_path = os.path.join(TF_LOGS_DIR, self.model_name)
        self.writers_dirs = {
            res: os.path.join(self.logs_path, f'{2**res}x{2**res}')
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.summary_writers = {
            res: tf.summary.create_file_writer(self.writers_dirs[res])
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }

        self.validate_config()

        self.clear_session_for_new_model = True

        self.G_object = Generator(config)
        self.D_object = Discriminator(config)

        if self.use_Gs:
            Gs_config = config
            self.Gs_valid_latents = self.valid_latents
            self.Gs_device = '/GPU:0' if self.use_gpu_for_Gs else '/CPU:0'

            if not self.use_gpu_for_Gs:
                Gs_config[DATA_FORMAT] = NHWC_FORMAT
                # NCHW -> NHWC
                self.toNHWC_axis = [0, 2, 3, 1]
                # NCHW -> NHWC -> NCHW
                self.toNCHW_axis = [0, 3, 1, 2]
                self.Gs_valid_latents = tf.transpose(self.valid_latents, self.toNHWC_axis)

            self.Gs_object = Generator(Gs_config)

        if mode == INFERENCE_MODE:
            self.initialize_models()
        elif mode == TRAIN_MODE:
            if single_process_training:
                self.setup_Gs_betas()
                self.initialize_models()
                self.create_images_generators(config)
                self.initialize_optimizers(create_all_variables=True)
            else:
                self.setup_Gs_betas(res)
                self.initialize_models(res, stage)
                self.create_images_generator(res, images_paths)
                self.initialize_optimizers(create_all_variables=False)

    def initialize_models(self, model_res=None, stage=None):
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
                # Not sure about this line
                Gs_model.set_weights(G_model.get_weights())

    def setup_Gs_betas(self, res=None):
        def get_res_beta(self, res):
            if self.Gs_beta is None:
                beta = tf.constant(0.5 ** (self.batch_sizes[2 ** res] / (1000.0 * self.Gs_beta_kimgs)), dtype='float32')
            else:
                beta = tf.constant(self.Gs_beta, dtype='float32')
            logging.info(f'Gs beta for res={res}: {beta}')
            return beta

        if res is None:
            for r in range(self.start_resolution_log2, self.resolution_log2 + 1):
                self.Gs_betas[r] = get_res_beta(self, r)
        else:
            self.Gs_betas[res] = get_res_beta(self, res)

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

    def validate_config(self):
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            if 2 ** res not in self.batch_sizes.keys():
                assert False, f'Missing batch size for res={2**res}'
        self.get_n_steps_for_last_stage()

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
            to_input_shape = (batch_size,) + to_layer.input_shape[1:]
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

    def initialize_G_optimizer(self, create_all_variables: bool):
        self.G_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.G_learning_rate,
            beta_1=self.beta1, beta_2=self.beta2,
            epsilon=1e-8,
            name='G_Adam'
        )
        if self.use_mixed_precision:
            self.G_optimizer = mixed_precision.LossScaleOptimizer(self.G_optimizer, dynamic=True)
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

        self.n_G_vars = len(self.G_optimizer.weights)

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
                shape=(batch_size,) + D_input_shape, minval=-1.0, maxval=1.0, dtype=self.compute_dtype
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
            from_input_shape = (batch_size,) + from_layer.input_shape[1:]
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

    def initialize_D_optimizer(self, create_all_variables: bool):
        self.D_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.D_learning_rate,
            beta_1=self.beta1, beta_2=self.beta2,
            epsilon=1e-8,
            name='D_Adam'
        )
        if self.use_mixed_precision:
            self.D_optimizer = mixed_precision.LossScaleOptimizer(self.D_optimizer, dynamic=True)
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

        self.n_D_vars = len(self.D_optimizer.weights)

    def reset_optimizers_state(self):
        G_optimizer_new_weights = mult_by_zero(self.G_optimizer.get_weights())
        self.G_optimizer.set_weights(G_optimizer_new_weights)

        D_optimizer_new_weights = mult_by_zero(self.D_optimizer.get_weights())
        self.D_optimizer.set_weights(D_optimizer_new_weights)

        logging.info('Optimizers states reset')

    def initialize_optimizers(self, create_all_variables):
        start_time = time.time()
        logging.info('Initializing optimizers...')

        # 1: create optimizer states for all internal and final output layers
        # 2: create optimizer states for all intermediate output layers
        # 3: set slots of optimizer to zero
        self.initialize_G_optimizer(create_all_variables)
        self.initialize_D_optimizer(create_all_variables)

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
                     f'G_lr={self.G_optimizer.learning_rate.numpy():.5f}, '
                     f'D_lr={self.D_optimizer.learning_rate.numpy():.5f}')

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

        optimizer_steps = self.compute_optimizer_steps(res, stage)
        set_optimizer_iters(self.D_optimizer, optimizer_steps)
        set_optimizer_iters(self.G_optimizer, optimizer_steps)

        # TODO: for debugging, remove later
        print('D opt iters:', self.D_optimizer.iterations)
        print('G opt iters:', self.G_optimizer.iterations)
        self.n_G_vars = len(self.G_optimizer.weights)
        self.n_D_vars = len(self.D_optimizer.weights)

    def compute_optimizer_steps(self, res, stage):
        # Note: res and stage refer to the last used model not the current one
        n_steps = 0
        if self.reset_opt_state_for_new_lod:
            assert stage == TRANSITION_MODE, \
                'If optimizers states should be reset for each resolution, ' \
                'then optimizers weights should only be loaded from transition stage'
            n_steps = self.get_n_steps(res, stage)
        else:
            # Optimizer state is not reset, so sum all previous number of iters
            for r in range(self.start_resolution_log2, res + 1):
                if r > self.start_resolution_log2:
                    n_steps += self.get_n_steps(r, TRANSITION_MODE)
                if r < res or (r == res and stage == STABILIZATION_MODE):
                    n_steps += self.get_n_steps(r, STABILIZATION_MODE)
        # TODO: for debugging, remove later
        print(f'Opt steps for res={res} and stage={stage}: {n_steps}')
        return np.array(n_steps)

    def create_images_generators(self, config):
        # This method is used in function train() which was used when training all models used the same process
        start_time = time.time()
        logging.info('Initializing images generators...')

        images_paths = load_images_paths(config)

        self.images_generators = {}
        for res in tqdm(range(self.start_resolution_log2, self.resolution_log2 + 1), desc='Generator res'):
            self.images_generators[res] = self.create_images_generator(
                res, images_paths, messages=False, return_generator=True
            )

        total_time = time.time() - start_time
        logging.info(f'Images generators initialized in {total_time:.3f} seconds!')

    def create_images_generator(self, res, images_paths, messages=True, return_generator=False):
        # This method is used for training each model in a separate process
        if messages:
            start_time = time.time()
            logging.info(f'Initializing images generator for {2**res}x{2**res} model...')

        # No caching by default
        cache = False
        if self.dataset_max_cache_res is not None:
            if res <= self.dataset_max_cache_res:
                cache = os.path.join(DATASET_CACHE_FOLDER, self.model_name, str(res))
                if STORAGE_PATH is not None:
                    cache = os.path.join(STORAGE_PATH, cache)
                if not os.path.exists(cache):
                    os.makedirs(cache)

        batch_size = self.batch_sizes[2 ** res]
        images_generator = create_training_dataset(
            images_paths, res, cache, batch_size,
            mirror_augment=self.mirror_augment,
            shuffle_dataset=self.shuffle_dataset,
            dtype=self.compute_dtype, data_format=self.data_format,
            n_parallel_calls=self.dataset_n_parallel_calls,
            n_prefetched_batches=self.dataset_n_prefetched_batches
        )

        if return_generator:
            return images_generator
        else:
            self.images_generator = images_generator

        if messages:
            total_time = time.time() - start_time
            logging.info(f'Image generator initialized in {total_time:.3f} seconds!')

    def create_models(self, res, mode):
        # All models should be initialized before calling this function
        D_model = self.D_object.create_D_model(res, mode=mode)
        G_model = self.G_object.create_G_model(res, mode=mode)
        if self.use_Gs:
            Gs_model = self.Gs_object.create_G_model(res, mode=mode)
            Gs_model.set_weights(G_model.get_weights())
        else:
            Gs_model = None
        return D_model, G_model, Gs_model

    def update_models_weights(self):
        self.D_object.save_D_weights_in_class(self.D_model)
        self.G_object.save_G_weights_in_class(self.G_model)
        if self.use_Gs:
            self.Gs_object.save_G_weights_in_class(self.Gs_model)

    def load_models_from_class(self):
        D_model = self.D_object.load_D_weights_from_class(self.D_model)
        G_model = self.G_object.load_G_weights_from_class(self.G_model)
        if self.use_Gs:
            Gs_model = self.Gs_object.load_G_weights_from_class(self.Gs_model)
        else:
            Gs_model = None
        return D_model, G_model, Gs_model

    def validate_res_and_mode_combination(self, res, mode):
        assert self.start_resolution_log2 <= res <= self.resolution_log2
        assert res > self.start_resolution_log2 or mode == STABILIZATION_MODE, \
            'For start resolution only stabilization stage is run'

    def get_n_steps_for_last_stage(self):
        batch_size = self.batch_sizes[self.target_resolution]
        # Simple case when total number of training images is not provided
        if self.total_kimages is None:
            n_kimages = self.stabilization_kimages_dict.get(self.target_resolution, self.stabilization_kimages)
            return int(np.ceil(1000 * n_kimages / batch_size))
        # Advanced case when total number of training images is not provided
        n_kimages = 0
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            if res > self.start_resolution_log2:
                n_kimages += self.transition_kimages_dict.get(2 ** res, self.transition_kimages)
            if res < self.resolution_log2:
                n_kimages += self.stabilization_kimages_dict.get(2 ** res, self.stabilization_kimages)
        logging.info('n_kimages for the last stage:', n_kimages)
        assert n_kimages > 0, 'Total number of images is greater than total number of images for all stages'
        return int(np.ceil(1000 * n_kimages / batch_size))

    def get_n_steps(self, res, mode):
        """
        Returns number of training steps fir provided res and mode
        """
        self.validate_res_and_mode_combination(res, mode)
        images_res = 2 ** res
        batch_size = self.batch_sizes[images_res]
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

    def get_n_finished_steps(self, res, mode):
        """
        Returns number of finished training steps when training for provided res and mode
        """
        self.validate_res_and_mode_combination(res, mode)
        n_steps = 0
        # Iterate for all previous resolutions
        for r in range(self.start_resolution_log2, res + 1):
            if r == res:
                if mode == STABILIZATION_MODE and r > self.start_resolution_log2:
                    n_steps += self.get_n_steps(res, TRANSITION_MODE)
            else:
                # The first resolution doesn't have transition stage
                if r > self.start_resolution_log2:
                    n_steps += self.get_n_steps(r, TRANSITION_MODE)
                n_steps += self.get_n_steps(r, STABILIZATION_MODE)
        return n_steps

    def load_trained_models(self, res, mode, models=None):
        # Note: when calling this function for inference models arg should not be passed
        if models is None:
            self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=mode)
        else:
            self.D_model, self.G_model, self.Gs_model = models

        step = self.get_n_steps(res, mode) - 1

        self.D_model = load_model(
            self.D_model, self.model_name, DISCRIMINATOR_NAME,
            res=res, stage=mode, step=step, storage_path=self.storage_path
        )
        self.G_model = load_model(
            self.G_model, self.model_name, GENERATOR_NAME,
            res=res, stage=mode, step=step, storage_path=self.storage_path
        )
        if self.use_Gs:
            self.Gs_model = load_model(
                self.Gs_model, self.model_name, GENERATOR_NAME + SMOOTH_POSTFIX,
                res=res, stage=mode, step=step, storage_path=self.storage_path
            )

        logging.info(f'Model weights loaded for res={res}, mode={mode}')
        return self.D_model, self.G_model, self.Gs_model

    def save_model_wrapper(self, model_type, res, stage, step):
        Gs_name = 'G' + SMOOTH_POSTFIX
        assert model_type in ['G', 'D', Gs_name]
        MODEL_ARG = 'model'
        MODEL_TYPE_ARG = 'model_type'

        if model_type == 'D':
            kwargs = {MODEL_ARG: self.D_model, MODEL_TYPE_ARG: DISCRIMINATOR_NAME}
        elif model_type == 'G':
            kwargs = {MODEL_ARG: self.G_model, MODEL_TYPE_ARG: GENERATOR_NAME}
        elif model_type == Gs_name:
            kwargs = {MODEL_ARG: self.Gs_model, MODEL_TYPE_ARG: GENERATOR_NAME + SMOOTH_POSTFIX}
        else:
            assert False, f"Unsupported model type '{model_type}'"

        shared_kwargs = {
            'model_name': self.model_name,
            'res': res,
            'stage': stage,
            'step': step,
            'storage_path': self.storage_path
        }

        kwargs = {**kwargs, **shared_kwargs}
        save_model(**kwargs)

    def save_models(self, res, mode, step):
        self.save_model_wrapper(model_type='D', res=res, stage=mode, step=step)
        self.save_model_wrapper(model_type='G', res=res, stage=mode, step=step)
        if self.use_Gs:
            self.save_model_wrapper(model_type='G' + SMOOTH_POSTFIX, res=res, stage=mode, step=step)

    def save_optimizers_weights(self, res, stage):
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

        save_model(**D_optimizer_kwargs)
        save_model(**G_optimizer_kwargs)

    def save_valid_images(self, res, step, stage, smoothed=False):
        dir_stage = stage
        if smoothed:
            dir_stage += SMOOTH_POSTFIX

        digits_in_number = 6
        fname = ('%0' + str(digits_in_number) + 'd') % step

        valid_images_dir = create_images_dir_name(self.model_name, res, dir_stage)
        use_grid_title = False
        if use_grid_title:
            valid_images_grid_title = create_images_grid_title(res, dir_stage, step)
        else:
            valid_images_grid_title = None

        save_in_jpg = res > self.max_png_res

        if smoothed:
            valid_images = self.Gs_model(self.Gs_valid_latents)
            if not self.use_gpu_for_Gs:
                valid_images = tf.transpose(valid_images, self.toNCHW_axis)
        else:
            valid_images = self.G_model(self.valid_latents)

        valid_images = convert_outputs_to_images(
            valid_images,
            max(2 ** res, self.min_target_single_image_size),
            data_format=self.data_format
        ).numpy()

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

    @tf.function
    def generate_latents(self, batch_size):
        latents = tf.random.normal(
            shape=(batch_size,) + self.z_dim, mean=0., stddev=1., dtype=self.compute_dtype
        )
        return latents

    @tf.function
    def smooth_net_weights(self, Gs_model, G_model, beta):
        trace_message('...Tracing smoothing weights...')
        smoothed_net_vars = Gs_model.trainable_variables
        source_net_vars = G_model.trainable_variables
        trace_vars(smoothed_net_vars, 'Smoothed vars:')
        trace_vars(source_net_vars, 'Source vars:')

        with tf.device(self.Gs_device):
            for a, b in zip(smoothed_net_vars, source_net_vars):
                a.assign(b + (a - b) * beta)

    @tf.function
    def G_train_step(self, G_model, D_model, latents, write_summary, step):
        G_vars = G_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(G_vars, 'Generator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as G_tape:
            G_tape.watch(G_vars)
            G_loss = self.G_loss_fn(G_model, D_model, self.G_optimizer,
                                    latents=latents,
                                    write_summary=write_summary,
                                    step=step,
                                    **self.G_loss_params)
            G_loss = maybe_scale_loss(G_loss, self.G_optimizer)

        G_grads = G_tape.gradient(G_loss, G_vars)
        G_grads = maybe_unscale_grads(G_grads, self.G_optimizer)
        self.G_optimizer.apply_gradients(zip(G_grads, G_vars))

        if write_summary:
            # Write gradients
            with tf.name_scope('G-grads'):
                for grad, var in zip(G_grads, G_vars):
                    tf.summary.histogram(var.name, grad, step=step)
            # Write weights
            with tf.name_scope('G-weights'):
                for var in G_vars:
                    tf.summary.histogram(var.name, var, step=step)

    @tf.function
    def D_train_step(self, G_model, D_model, latents, images, write_summary, step):
        D_vars = D_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(D_vars, 'Discriminator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as D_tape:
            D_tape.watch(D_vars)
            D_loss = self.D_loss_fn(G_model, D_model, self.D_optimizer,
                                    latents=latents,
                                    real_images=images,
                                    write_summary=write_summary,
                                    step=step,
                                    **self.D_loss_params)
            D_loss = maybe_scale_loss(D_loss, self.D_optimizer)

        D_grads = D_tape.gradient(D_loss, D_vars)
        D_grads = maybe_unscale_grads(D_grads, self.D_optimizer)
        self.D_optimizer.apply_gradients(zip(D_grads, D_vars))

        if write_summary:
            # Write gradients
            with tf.name_scope('D-grads'):
                for grad, var in zip(D_grads, D_vars):
                    tf.summary.histogram(var.name, grad, step=step)
            # Write weights
            with tf.name_scope('D-weights'):
                for var in D_vars:
                    tf.summary.histogram(var.name, var, step=step)

    @tf.function
    def train_step(self, G_model, D_model, G_latents, D_latents,
                   images, write_summary, step):
        trace_message(' ...Modified train step tracing... ')
        # Note: explicit use of G and D models allows one to make sure that
        # tf.function doesn't compile models. Additionally tracing is used
        # (previously for res=3 and mode=transition G model used variables only from res=2)
        self.D_train_step(G_model, D_model, D_latents, images, write_summary, step)
        self.G_train_step(G_model, D_model, G_latents, write_summary, step)

    def post_train_step_actions(self, res, mode, step, last_step_cond, write_summary, summary_writer):
        if step > 0:
            # For the first step optimizers learning rates are zeros
            if self.use_Gs:
                self.smooth_net_weights(
                    Gs_model=self.Gs_model, G_model=self.G_model, beta=self.Gs_betas[res]
                )

        if write_summary:
            summary_writer.flush()

        if step % self.save_model_every == 0 or last_step_cond:
            self.save_models(res=res, mode=mode, step=step)

        if step % self.save_images_every == 0 or last_step_cond:
            self.save_valid_images(res, step, stage=mode)
            if self.use_Gs:
                self.save_valid_images(res, step, stage=mode, smoothed=True)

    def train(self):
        # TODO: refactor this function, and make it consistent with training for each separate stage
        train_start_time = time.time()
        tStep = tf.Variable(0, trainable=False, dtype=tf.int64)
        tWriteSummary = tf.Variable(True, trainable=False, dtype=tf.bool)

        for res in tqdm(range(self.start_resolution_log2, self.resolution_log2 + 1), desc='Training res'):
            logging.info(f'Training {2**res}x{2**res} model...')
            res_start_time = time.time()

            if self.reset_opt_state_for_new_lod:
                self.reset_optimizers_state()
            self.adjust_optimizers_learning_rate(res)

            images_generator = iter(self.images_generators[res])
            batch_size = self.batch_sizes[2 ** res]
            summary_writer = self.summary_writers[res]

            n_finished_steps = self.get_n_finished_steps(res, TRANSITION_MODE) if res > self.start_resolution else 0
            transition_steps = self.get_n_steps(res, TRANSITION_MODE)
            stabilization_steps = self.get_n_steps(res, STABILIZATION_MODE)

            with summary_writer.as_default():
                # The first resolution doesn't use alpha parameter,
                # but has usual number of steps for stabilization phase
                if res > self.start_resolution_log2:
                    # Fading in stage
                    transition_stage_start_time = time.time()

                    if self.clear_session_for_new_model:
                        logging.info('Clearing session...')
                        tf.keras.backend.clear_session()

                    self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=TRANSITION_MODE)
                    self.D_model, self.G_model, self.Gs_model = self.load_models_from_class()

                    tStep.assign(n_finished_steps)
                    tWriteSummary.assign(True)

                    desc = f'{2**res}x{2**res} model, transition steps'
                    for step in tqdm(range(transition_steps), desc=desc):
                        last_step_cond = is_last_step(step, transition_steps)

                        if step % self.batch_repeats == 0:
                            alpha = compute_alpha(step, transition_steps)
                            self.G_model = update_wsum_alpha(self.G_model, alpha)
                            self.D_model = update_wsum_alpha(self.D_model, alpha)
                            if self.use_Gs:
                                self.Gs_model = update_wsum_alpha(self.Gs_model, alpha)

                        write_summary = step % self.summary_every == 0 or last_step_cond
                        tWriteSummary.assign(write_summary)
                        if write_summary:
                            tStep.assign(step + n_finished_steps)
                            tf.summary.scalar('alpha', alpha, step=step)

                        G_latents = self.generate_latents(batch_size)
                        D_latents = self.generate_latents(batch_size)
                        batch_images = next(images_generator)

                        self.train_step(
                            G_model=self.G_model, D_model=self.D_model,
                            G_latents=G_latents, D_latents=D_latents, images=batch_images,
                            write_summary=tWriteSummary, step=tStep
                        )
                        self.post_train_step_actions(
                            res=res, mode=TRANSITION_MODE, step=step, last_step_cond=last_step_cond,
                            write_summary=write_summary, summary_writer=summary_writer
                        )

                    self.update_models_weights()
                    remove_old_models(
                        self.model_name, res, stage=TRANSITION_MODE,
                        max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
                    )

                    transition_stage_total_time = time.time() - transition_stage_start_time
                    logging.info(f'Transition stage took {format_time(transition_stage_total_time)}')

                # Stabilization stage
                stabilization_stage_start_time = time.time()

                if self.clear_session_for_new_model:
                    logging.info('Clearing session...')
                    tf.keras.backend.clear_session()

                self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=STABILIZATION_MODE)
                self.D_model, self.G_model, self.Gs_model = self.load_models_from_class()

                tStep.assign(transition_steps + n_finished_steps)
                tWriteSummary.assign(True)

                desc = f'{2**res}x{2**res} model, stabilization steps'
                for step in tqdm(range(stabilization_steps), desc=desc):
                    last_step_cond = is_last_step(step, stabilization_steps)

                    write_summary = step % self.summary_every == 0 or last_step_cond
                    tWriteSummary.assign(write_summary)
                    if write_summary:
                        tStep.assign(step + transition_steps + n_finished_steps)

                    G_latents = self.generate_latents(batch_size)
                    D_latents = self.generate_latents(batch_size)
                    batch_images = next(images_generator)

                    self.train_step(
                        G_model=self.G_model, D_model=self.D_model,
                        G_latents=G_latents, D_latents=D_latents, images=batch_images,
                        write_summary=tWriteSummary, step=tStep
                    )
                    self.post_train_step_actions(
                        res=res, mode=STABILIZATION_MODE, step=step, last_step_cond=last_step_cond,
                        write_summary=write_summary, summary_writer=summary_writer
                    )

                self.update_models_weights()
                remove_old_models(
                    self.model_name, res, stage=STABILIZATION_MODE,
                    max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
                )

                stabilization_stage_total_time = time.time() - stabilization_stage_start_time
                logging.info(f'Stabilization stage took {format_time(stabilization_stage_total_time)}')

                res_total_time = time.time() - res_start_time
                logging.info(f'Training of {2**res}x{2**res} model took {format_time(res_total_time)}')
                logging.info(f'----------------------------------------------------------------------')
                logging.info('')

        train_total_time = time.time() - train_start_time
        logging.info(f'Training finished in {format_time(train_total_time)}!')

    def run_transition_stage(self, res):
        transition_stage_start_time = time.time()

        self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=TRANSITION_MODE)
        # Load weights from previous stage: res - 1 and stabilization mode
        logging.info(f'Loading models for res={res} and mode={TRANSITION_MODE}...')
        self.D_model, self.G_model, self.Gs_model = self.load_trained_models(
            res - 1, STABILIZATION_MODE, models=[self.D_model, self.G_model, self.Gs_model]
        )
        self.zero_optimizers_learning_rate()

        images_generator = iter(self.images_generator)
        batch_size = self.batch_sizes[2 ** res]
        summary_writer = self.summary_writers[res]

        n_finished_steps = self.get_n_finished_steps(res, TRANSITION_MODE)
        transition_steps = self.get_n_steps(res, TRANSITION_MODE)

        tStep = tf.Variable(n_finished_steps, trainable=False, dtype=tf.int64)
        tWriteSummary = tf.Variable(True, trainable=False, dtype=tf.bool)

        with summary_writer.as_default():
            desc = f'{2**res}x{2**res} model, transition steps'
            for step in tqdm(range(transition_steps), desc=desc):
                last_step_cond = is_last_step(step, transition_steps)

                if step % self.batch_repeats == 0:
                    alpha = compute_alpha(step, transition_steps)
                    self.D_model = update_wsum_alpha(self.D_model, alpha)
                    self.G_model = update_wsum_alpha(self.G_model, alpha)
                    if self.use_Gs:
                        self.Gs_model = update_wsum_alpha(self.Gs_model, alpha)

                write_summary = step % self.summary_every == 0 or last_step_cond
                tWriteSummary.assign(write_summary)
                if write_summary:
                    tStep.assign(step + n_finished_steps)
                    tf.summary.scalar('alpha', alpha, step=step)

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                batch_images = next(images_generator)

                self.train_step(
                    G_model=self.G_model, D_model=self.D_model,
                    G_latents=G_latents, D_latents=D_latents, images=batch_images,
                    write_summary=tWriteSummary, step=tStep
                )
                # TODO: for debugging, remove later
                if step > 0:
                    if len(self.G_optimizer.weights) > self.n_G_vars:
                        print(f'Increased G_vars, step = {step}')
                    if len(self.D_optimizer.weights) > self.n_D_vars:
                        print(f'Increased D_vars, step = {step}')
                if step == 0:
                    if not self.reset_opt_state_for_new_lod:
                        self.restore_optimizers_state(res - 1, STABILIZATION_MODE)
                    # Always adjust learning rates
                    self.adjust_optimizers_learning_rate(res)
                    # TODO: for debugging, remove later
                    self.n_G_vars = len(self.G_optimizer.weights)
                    self.n_D_vars = len(self.D_optimizer.weights)
                self.post_train_step_actions(
                    res=res, mode=TRANSITION_MODE, step=step, last_step_cond=last_step_cond,
                    write_summary=write_summary, summary_writer=summary_writer
                )

        remove_old_models(
            self.model_name, res, stage=TRANSITION_MODE,
            max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        # Save states after extra weights are removed
        self.save_optimizers_weights(res, stage=TRANSITION_MODE)

        transition_stage_total_time = time.time() - transition_stage_start_time
        logging.info(f'Transition stage took {format_time(transition_stage_total_time)}')

    def run_stabilization_stage(self, res):
        stabilization_stage_start_time = time.time()

        self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=STABILIZATION_MODE)
        if res > self.start_resolution_log2:
            logging.info(f'Loading models for res={res} and mode={STABILIZATION_MODE}...')
            # Load weights from previous stage: res and transition mode
            self.D_model, self.G_model, self.Gs_model = self.load_trained_models(
                res, TRANSITION_MODE, models=[self.D_model, self.G_model, self.Gs_model]
            )
        self.zero_optimizers_learning_rate()

        images_generator = iter(self.images_generator)
        batch_size = self.batch_sizes[2 ** res]
        summary_writer = self.summary_writers[res]

        n_finished_steps = self.get_n_finished_steps(res, STABILIZATION_MODE)
        stabilization_steps = self.get_n_steps(res, STABILIZATION_MODE)

        tStep = tf.Variable(n_finished_steps, trainable=False, dtype=tf.int64)
        tWriteSummary = tf.Variable(True, trainable=False, dtype=tf.bool)

        with summary_writer.as_default():
            desc = f'{2**res}x{2**res} model, stabilization steps'
            for step in tqdm(range(stabilization_steps), desc=desc):
                last_step_cond = is_last_step(step, stabilization_steps)

                write_summary = step % self.summary_every == 0 or last_step_cond
                tWriteSummary.assign(write_summary)
                if write_summary:
                    tStep.assign(step + n_finished_steps)

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                batch_images = next(images_generator)

                self.train_step(
                    G_model=self.G_model, D_model=self.D_model,
                    G_latents=G_latents, D_latents=D_latents, images=batch_images,
                    write_summary=tWriteSummary, step=tStep
                )
                # TODO: for debugging, remove later
                if step > 0:
                    if len(self.G_optimizer.weights) > self.n_G_vars:
                        print(f'Increased G_vars, step = {step}')
                    if len(self.D_optimizer.weights) > self.n_D_vars:
                        print(f'Increased D_vars, step = {step}')
                if step == 0:
                    if res > self.start_resolution_log2:
                        self.restore_optimizers_state(res, stage=TRANSITION_MODE)
                    # Always adjust learning rates
                    self.adjust_optimizers_learning_rate(res)
                    # TODO: for debugging, remove later
                    self.n_G_vars = len(self.G_optimizer.weights)
                    self.n_D_vars = len(self.D_optimizer.weights)
                self.post_train_step_actions(
                    res=res, mode=STABILIZATION_MODE, step=step, last_step_cond=last_step_cond,
                    write_summary=write_summary, summary_writer=summary_writer
                )

        remove_old_models(
            self.model_name, res, stage=STABILIZATION_MODE,
            max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        # Save states after extra weights are removed
        self.save_optimizers_weights(res, stage=STABILIZATION_MODE)

        stabilization_stage_total_time = time.time() - stabilization_stage_start_time
        logging.info(f'Stabilization stage took {format_time(stabilization_stage_total_time)}')

    def run_train_stage(self, res, mode):
        assert self.start_resolution_log2 <= res <= self.resolution_log2
        if mode == STABILIZATION_MODE:
            self.run_stabilization_stage(res)
        elif mode == TRANSITION_MODE:
            assert res > self.start_resolution_log2
            self.run_transition_stage(res)
        else:
            assert False, f'Train stage must be one of f[{STABILIZATION_MODE}, {TRANSITION_MODE}]'
