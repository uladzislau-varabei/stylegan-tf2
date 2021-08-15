import os
import sys
import argparse
import logging
import time
from multiprocessing import Process

import numpy as np
# Note: do not import tensorflow here or you won't be able to train each stage
# in a new process

from config import Config as cfg
from utils import LOGS_DIR, TRAIN_MODE, INFERENCE_MODE, TRANSITION_MODE, STABILIZATION_MODE
from utils import load_config, prepare_gpu, load_images_paths, format_time, DEBUG_MODE
from model import StyleGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train StyleGAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        #default=os.path.join('configs', 'demo_config.json'),
        default=os.path.join('configs', 'debug_config.json'),
        #default=os.path.join('configs', 'lsun_living_room.json'),
        #required=True
    )
    args = parser.parse_args()
    return args


def prepare_logger(config_path):
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    filename = os.path.join(
        LOGS_DIR, 'logs_' + os.path.split(config_path)[1].split('.')[0] + '.txt'
    )
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)

    print('Logging initialized!')


def run_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()


def trace_graphs(config, config_path=None):
    if config_path is not None:
        prepare_logger(config_path)
    pid = os.getpid()
    logging.info(f'Tracing graphs uses PID={pid}')
    prepare_gpu()
    StyleGAN_model = StyleGAN(config, mode=INFERENCE_MODE)
    StyleGAN_model.trace_graphs()


def run_train_stage(config, images_paths, res, mode, config_path=None):
    if config_path is not None:
        prepare_logger(config_path)
    pid = os.getpid()
    logging.info(f'Training for {2**res}x{2**res} resolution and {mode} mode uses PID={pid}')
    prepare_gpu()
    StyleGAN_model = StyleGAN(
        config, mode=TRAIN_MODE, images_paths=images_paths, res=res, stage=mode
    )
    StyleGAN_model.run_train_stage(res=res, mode=mode)


def train_model(config, config_path=None):
    target_resolution = config[cfg.TARGET_RESOLUTION]
    resolution_log2 = int(np.log2(target_resolution))
    assert target_resolution == 2 ** resolution_log2 and target_resolution >= 4

    start_resolution = config.get(cfg.START_RESOLUTION, cfg.DEFAULT_START_RESOLUTION)
    start_resolution_log2 = int(np.log2(start_resolution))
    assert start_resolution == 2 ** start_resolution_log2 and start_resolution >= 4

    # Load images paths before training starts to avoid breaking training process in case of
    # a sudden remove of a file with images paths
    images_paths = load_images_paths(config)

    # run_process(target=trace_graphs, args=(config, config_path))
    sleep(1)

    train_start_time = time.time()

    for res in range(start_resolution_log2, resolution_log2 + 1):
        logging.info(f'Training {2**res}x{2**res} model...')
        res_start_time = time.time()

        if res > start_resolution_log2:
            # The first resolution doesn't use alpha parameter,
            # but has usual number of steps for stabilization phase

            # Transition stage
            run_process(
                target=run_train_stage,
                args=(config, images_paths, res, TRANSITION_MODE, config_path)
            )
            sleep(1)

        # Stabilization stage
        run_process(
            target=run_train_stage,
            args=(config, images_paths, res, STABILIZATION_MODE, config_path)
        )
        sleep(1)

        res_total_time = time.time() - res_start_time
        logging.info(f'Training of {2**res}x{2**res} model took {format_time(res_total_time)}')
        logging.info(f'----------------------------------------------------------------------')
        logging.info('')

    train_total_time = time.time() - train_start_time
    logging.info(f'Training finished in {format_time(train_total_time)}!')


def sleep(s):
    print(f"Sleeping {s}s...")
    time.sleep(s)
    print("Sleeping finished")


if __name__ == '__main__':
    args = parse_args()

    prepare_logger(args.config_path)
    config = load_config(args.config_path)
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
        train_model(config, args.config_path)
