class Config:

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
    # Batch size for stabilization stage for target resolution.
    # It's the longest training stage and sometimes it's possible to increase batch comparing with transition stage
    FINAL_BATCH_SIZE = 'final_batch_size'
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

    # Height / width ratio of dataset images. Only set it if a wide dataset is used, e.g. LSUN Car.
    # The value is target height / image
    DATASET_HW_RATIO = 'dataset_hw_ratio'
    # Number of parallel calls to dataset
    # Note: a good choice is to use a number of cpu cores
    DATASET_N_PARALLEL_CALLS = 'dataset_n_parallel_calls'
    # Number of prefetched batches for dataset
    DATASET_N_PREFETCHED_BATCHES = 'dataset_n_prefetched_batches'
    # Maximum number of images to be used for training
    DATASET_N_MAX_KIMAGES = 'dataset_max_kimages'
    # Shuffle dataset every time it is finished?
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


    ### ---------- Default options ---------- ###
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
    DEFAULT_LATENT_SIZE = 512
    DEFAULT_DLATENT_SIZE = 512 # Maybe set default equal to latent size?
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
    DEFAULT_FINAL_BATCH_SIZE = None
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
    DEFAULT_DATASET_HW_RATIO = 1
    DEFAULT_DATASET_N_PARALLEL_CALLS = 'auto'
    DEFAULT_DATASET_N_PREFETCHED_BATCHES = 'auto'
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
