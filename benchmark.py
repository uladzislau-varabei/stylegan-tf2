import os
import argparse

import numpy as np

from config import Config as cfg
from utils import load_config, prepare_gpu, STABILIZATION_MODE, TRANSITION_MODE, BENCHMARK_MODE
from model import StyleGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Script to benchmark StyleGAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to benchmark (json format)',
        default=os.path.join('configs', 'new_lsun_living_room.json'),
        # required=True
    )
    parser.add_argument(
        '--images',
        help='Number of images to run through models',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--res',
        help='Images resolution to benchmark, by default target resolution is used',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--transition_stage',
        help='Use transition stage of model? If not provided, stabilization mode is used',
        action='store_true'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Example call:
    # python .\benchmark.py --config .\configs\new_lsun_living_room.json --images 1000 --res 256
    args = parse_args()

    images = args.images
    res = args.res
    transition_stage = args.transition_stage
    config = load_config(args.config_path)

    # Determine res and stage of model
    if res == -1:
        res = int(np.log2(config[cfg.TARGET_RESOLUTION]))
        stage = STABILIZATION_MODE
    else:
        res = int(np.log2(res))
        stage = TRANSITION_MODE if transition_stage else STABILIZATION_MODE

    # prepare_gpu(mode='growth')
    prepare_gpu()
    StyleGAN_model = StyleGAN(config, mode=BENCHMARK_MODE, res=res, stage=stage)
    # Note: script benchmarks only model training time, metrics and other post train step actions are not run
    StyleGAN_model.run_benchmark_stage(res, stage, images)


### ----- Results -----
"""
1. LSUN Living Room (old)
- Benchmark took 1m 37s for 1000 images. In average 10.358 images/sec.
- GPU usage around 80%
2. LSUN Living Room: XLA
- Benchmark took 1m 23s for 1000 images. In average 12.048 images/sec.
- GPU usage around 84%
- Performance boost around 16%. Mostly due to GPU usage (which is probably higher due to XLA)
3. LSUN Living Room: XLA, batch size x2 - 16 for target resolution
- Benchmark took 1m 07s for 1000 images. In average 14.971 images/sec.
- GPU usage around 88-90%
- Performance boost around 24%
- Warnings about OOM error
- No metrics were ran
4. LSUN Living Room: XLA, batch size x2 - 16 for target resolution, target resolution in transition stage
- Benchmark took 12m 14s for 10000 images. In average 13.621 images/sec.
- GPU usage around 87-90%
- No metrics were ran
5. LSUN Living Room: XLA, batch size x2 - 16 for target resolution, target resolution in transition stage, 
all swish activations
- OOM
6. LSUN Living Room: XLA, batch size x2 - 16 for target resolution, target resolution in transition stage, 
all selu activations
- OOM
7. LSUN Living Room: XLA, batch size x2 - 16 for target resolution, target resolution in transition stage, 
selu activation for mapping
- Benchmark took 12m 07s for 10000 images. In average 13.750 images/sec.
8. LSUN Living Room: XLA, batch size x2 - 16 for target resolution, target resolution in transition stage, 
selu activation for mapping, h-swish for G and D
-  OOM
9. LSUN Living Room: XLA, batch size x1.5 - 12 for target resolution, target resolution in transition stage, 
selu activation for mapping, h-swish for G and D
-  OOM
10. LSUN Living Room: XLA, batch size x1 - 8 for target resolution, target resolution in transition stage, 
selu activation for mapping, h-swish for G and D
- Benchmark took 7m 36s for 5000 images. In average 10.961 images/sec.
- GPU usage around 84%
11. LSUN Living Room: XLA, batch size x1 - 8 for target resolution, target resolution in transition stage, 
all mish activations
- Benchmark took 7m 52s for 5000 images. In average 10.590 images/sec.
- GPU usage around 83%
12. LSUN Living Room: XLA, batch size x1 - 8 for target resolution, target resolution in transition stage, 
all mish activations with XLA for function and its grad 
(does it have effect if XLA is enabled in config and fused_bias_act layer is used? maybe effect for backprop only?)
- Benchmark took 7m 33s for 5000 images. In average 11.041 images/sec.
- GPU usage around 84%
13. LSUN Living Room: XLA, batch size x2 - 16 for target resolution, target resolution in transition stage, 
all mish activations with XLA for function and its grad 
(does it have effect if XLA is enabled in config and fused_bias_act layer is used? maybe effect for backprop only?)
- OOM
14. 
2000 imgs, stabilization
bs = 6: 5.413 images/sec
bs = 8 (mbstd f2): 
    1 - 7.618 images/sec (around 10 OOM warnings for 1.41G), 
    2 - error (existing window), 
    3 - error (existing window), 
    4 - 7.803 images/sec, no warnings (new window)
    5 - 8.170 images/sec  lots of OOM warnings for 1.43G - with browser opened later (existing window)
2000 imgs, transition
bs = 6: 5.182 images/sec images/sec
bs = 8: error (lots of OOM warnings for 1.41G)
"""

"""

...project/dnnlib/ops/fused_bias_act.cu(204): error: expected an expression

...project/dnnlib/ops/fused_bias_act.cu(204): error: no instance of constructor "tensorflow::register_op::OpDefBuilderWrapper::OpDefBuilderWrapper" matches the argument list
            argument types are: (const char [13], __nv_bool)

...project/dnnlib/ops/fused_bias_act.cu(217): error: expected an expression

...project/dnnlib/ops/fused_bias_act.cu(217): error: expected an expression

...project/dnnlib/ops/fused_bias_act.cu(217): error: expected a type specifier

...project/dnnlib/ops/fused_bias_act.cu(217): error: expected an expression

...project/dnnlib/ops/fused_bias_act.cu(218): error: expected an expression

...project/dnnlib/ops/fused_bias_act.cu(218): error: expected an expression

...project/dnnlib/ops/fused_bias_act.cu(218): error: expected a type specifier

...project/dnnlib/ops/fused_bias_act.cu(218): error: expected an expression

10 errors detected in the compilation of "...project/dnnlib/ops/fused_bias_act.cu".
_pywrap_tensorflow_internal.lib
fused_bias_act.cu

"""