import os
import argparse
import logging

import numpy as np

from config import Config as cfg
from utils import TRAIN_MODE, TRANSITION_MODE, STABILIZATION_MODE
from utils import load_config, load_images_paths, prepare_logger
from tf_utils import prepare_gpu
from model import StyleGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Script to benchmark StyleGAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        required=True
    )
    parser.add_argument(
        '--res',
        help='Images resolution to benchmark, by default target resolution is used',
        type=int,
    )
    parser.add_argument(
        '--transition_stage',
        help='Use transition stage of model? If not provided, stabilization mode is used',
        action='store_true'
    )
    args = parser.parse_args()
    return args


def run_train_stage(config, res, mode):
    prepare_logger(config[cfg.MODEL_NAME], res, mode)
    logging.info('Running stage with the following config:')
    logging.info(config)
    images_paths = load_images_paths(config)
    pid = os.getpid()
    logging.info(f'Training for {2 ** res}x{2 ** res} resolution and {mode} mode uses PID={pid}')
    prepare_gpu()
    StyleGAN_model = StyleGAN(config, mode=TRAIN_MODE, images_paths=images_paths, res=res, stage=mode)
    StyleGAN_model.run_train_stage(res=res, mode=mode)
    logging.info('----------------------------------------------------------------------')
    logging.info('')


if __name__ == '__main__':
    # Note: this script can only be called if there exists a folder with weights from previous stage
    # (if it's needed according to a config start resolution)
    # python .\run_train_stage.py --config_path .\configs\lsun_car_512x384.json --res 256 --transition_stage
    # python .\run_train_stage.py --config_path .\configs\lsun_car_512x384.json --res 512

    args = parse_args()

    res = int(np.log2(args.res))
    transition_stage = args.transition_stage
    stage = TRANSITION_MODE if transition_stage else STABILIZATION_MODE
    config = load_config(args.config_path)

    run_train_stage(config=config, res=res, mode=stage)
