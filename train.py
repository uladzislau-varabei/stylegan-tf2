import os
import argparse
import logging
import time
from multiprocessing import Process

import numpy as np
# Note: do not import tensorflow here or you won't be able to train each stage
# in a new process

from config import Config as cfg
from utils import TRAIN_MODE, INFERENCE_MODE, TRANSITION_MODE, STABILIZATION_MODE, DEBUG_MODE
from utils import load_config, load_images_paths, format_time, prepare_logger, sleep
from tf_utils import prepare_gpu
from networks import ModelConfig
from model import StyleGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train StyleGAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        #default=os.path.join('configs', 'demo_config.json'),
        default=os.path.join('configs', 'debug_config.json'),
        #default=os.path.join('configs', 'lsun_living_room.json'),
        #default=os.path.join('configs', 'lsun_car_512x384.json'),
        #required=True
    )
    args = parser.parse_args()
    return args


def run_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()


def trace_graphs(config):
    prepare_logger(config[cfg.MODEL_NAME])
    pid = os.getpid()
    logging.info(f'Tracing graphs uses PID={pid}')
    prepare_gpu()
    StyleGAN_model = StyleGAN(config, mode=INFERENCE_MODE)
    StyleGAN_model.trace_graphs()


def run_train_stage(config, images_paths, res, mode):
    prepare_logger(config[cfg.MODEL_NAME])
    pid = os.getpid()
    logging.info(f'Training for {2**res}x{2**res} resolution and {mode} mode uses PID={pid}')
    prepare_gpu('growth')
    StyleGAN_model = StyleGAN(config, mode=TRAIN_MODE, images_paths=images_paths, res=res, stage=mode)
    StyleGAN_model.run_train_stage(res=res, mode=mode)


def train_model(config):
    model_cfg = ModelConfig(config)

    # Load images paths before training starts to avoid breaking training process in case of
    # a sudden remove of a file with images paths
    images_paths = load_images_paths(config)

    # run_process(target=trace_graphs, args=(config, ))
    sleep(3)

    train_start_time = time.time()

    for res in range(model_cfg.start_resolution_log2, model_cfg.resolution_log2 + 1):
        logging.info(f'Training {2**res}x{2**res} model...')
        res_start_time = time.time()

        if res > model_cfg.start_resolution_log2:
            # The first resolution doesn't use alpha parameter,
            # but has usual number of steps for stabilization phase

            # Transition stage
            run_process(
                target=run_train_stage,
                args=(config, images_paths, res, TRANSITION_MODE)
            )
            sleep(3)

        # Stabilization stage
        run_process(
            target=run_train_stage,
            args=(config, images_paths, res, STABILIZATION_MODE)
        )
        sleep(3)

        res_total_time = time.time() - res_start_time
        logging.info(f'Training of {2**res}x{2**res} model took {format_time(res_total_time)}')
        logging.info(f'----------------------------------------------------------------------')
        logging.info('')

    train_total_time = time.time() - train_start_time
    logging.info(f'Training finished in {format_time(train_total_time)}!')


if __name__ == '__main__':
    args = parse_args()

    config = load_config(args.config_path)
    prepare_logger(config[cfg.MODEL_NAME])
    logging.info('Training with the following config:')
    logging.info(config)

    # Training model of each stage in a separate process can be much faster
    # as all GPU resources are released after process is finished. This mode is strongly recommended
    single_process_training = False

    # Should log debug information?
    debug_mode = False
    os.environ[DEBUG_MODE] = '1' if debug_mode else '0'

    if single_process_training:
        prepare_gpu()
        StyleGAN_model = StyleGAN(config, mode=TRAIN_MODE, single_process_training=True)
        StyleGAN_model.trace_graphs()
        StyleGAN_model.train()
    else:
        train_model(config)
