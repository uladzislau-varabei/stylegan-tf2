import os
import argparse
import time

import numpy as np
import tensorflow as tf

from dataloader_utils import convert_outputs_to_images
from image_utils import fast_save_grid
from utils import INFERENCE_MODE, \
    LATENT_SIZE, DEFAULT_VALID_GRID_NROWS, DEFAULT_VALID_GRID_NCOLS,\
    DATA_FORMAT, DEFAULT_DATA_FORMAT, NCHW_FORMAT, WEIGHTS_FOLDER
from utils import load_config, prepare_gpu, load_weights, generate_latents
from model import StyleGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train Progressive GAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        default=os.path.join('configs', 'lsun_living_room.json'),
        required=True
    )
    parser.add_argument(
        '--weights_path',
        help='Path to a model weights',
        required=True
    )
    # Image options
    parser.add_argument(
        '--image_fname',
        help='Filename for generated image',
        required=True
    )
    parser.add_argument(
        '--grid_cols',
        help='Number of columns in image grid',
        type=int,
        default=DEFAULT_VALID_GRID_NCOLS
    )
    parser.add_argument(
        '--grid_rows',
        help='Number of rows in image grid',
        type=int,
        default=DEFAULT_VALID_GRID_NROWS
    )
    parser.add_argument(
        '--save_in_jpg',
        help='Save generated image in jpg? If not, png will be used',
        default=True
    )
    args = parser.parse_args()
    return args


def generate_images(model: tf.keras.Model, config: dict):
    start_time = time.time()

    latent_size = config[LATENT_SIZE]
    data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
    if data_format == NCHW_FORMAT:
        z_dim = (latent_size, 1, 1)
    else: # dat_format == NHWC
        z_dim = (1, 1, latent_size)

    # Try dealing with a case when lots of images are to be generated
    if grid_cols * grid_rows < 32:
        iters, batch_size = 1, grid_cols * grid_rows
    else:
        iters, batch_size = max(grid_cols, grid_rows), min(grid_cols, grid_rows)

    images = []
    for _ in range(iters):
        batch_latents = generate_latents(batch_size, z_dim)
        images.append(
            model(batch_latents, training=False)
        )
    images = tf.concat(images, axis=0)
    images = convert_outputs_to_images(images, 2 ** res, data_format=data_format).numpy()

    total_time = time.time() - start_time
    print(f'Generated images in {total_time:.3f}s')

    return images


def extract_res_and_stage(p):
    s1 = p.split(WEIGHTS_FOLDER)[1]
    splits = s1.split(os.path.sep)
    res = int(np.log2(int(splits[2].split('x')[0])))
    stage = splits[3]
    return res, stage


if __name__ == '__main__':
    # Example call:
    # python .\inference.py --config_path .\configs\lsun_living_room.json  --weights_path .\weights\lsun_living_room\256x256\stabilization\step3000000\G_model_smoothed.h5 --image_fname images --grid_cols 12 --grid_rows 9
    args = parse_args()

    config = load_config(args.config_path)
    res, stage = extract_res_and_stage(args.weights_path)
    weights_path = args.weights_path

    # Grid image options
    image_fname = args.image_fname
    grid_cols = args.grid_cols
    grid_rows = args.grid_rows
    save_in_jpg = args.save_in_jpg

    prepare_gpu(mode='growth')
    StyleGAN_model = StyleGAN(config, mode=INFERENCE_MODE, res=res, stage=stage)
    Gs_model = StyleGAN_model.Gs_object.create_G_model(model_res=res, mode=stage)
    Gs_model = load_weights(Gs_model, weights_path, optimizer_call=False)
    images = generate_images(Gs_model, config)

    out_dir = 'results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fast_save_grid(
        out_dir=out_dir,
        fname=image_fname,
        images=images,
        title=None,
        nrows=grid_rows,
        ncols=grid_cols,
        padding=2,
        save_in_jpg=save_in_jpg
    )
