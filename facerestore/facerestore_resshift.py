#!/usr/bin/env python3
# Standard library imports
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Tuple

# Third party imports
import fastai
from loguru import logger
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# IO defaults
BENCHMARK = "test_dataset"
PRED_STEP = None  # Predecessor step to use
STEP = "facerestore_resshift"  # HuggingFace colorization and super resolution used in this script
ROOT_DIR = Path("/data")
SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SUPPORTED_BENCHMARKS = (
    "fer2013",
    "test_dataset",
)  # test_dataset is just a sampling of fer2013
# Colorizer defaults
RENDER_FACTOR = 15


def argparser():
    prog = "FaceRestoration"
    parser = argparse.ArgumentParser(
        prog="FaceRestoration",
        description="Restore images of grayscale faces using colorization and super resolution.",
    )
    # IO arguments
    # TODO: Add specifics on root directory's construction
    bm_help = "Name of benchmark dataset to process."
    io_help = "Path to root data directory; refer to README for specifics on that directory's construction."
    ps_help = "Name of predecessor step corresponding to sub-directory in data/facerestore/ dir."
    st_help = "Name of destination directory (e.g., HF for HuggingFace routine in this script)."
    parser.add_argument("-bm", "--benchmark", default=BENCHMARK, type=str, help=bm_help)
    parser.add_argument("-io", "--root_dir", default=ROOT_DIR, type=Path, help=io_help)
    parser.add_argument("-ps", "--pred_step", default=PRED_STEP, help=ps_help)
    parser.add_argument("-st", "--step", default=STEP, type=str, help=st_help)
    # Colorizer arguments
    rf_help = "Lower number is more colorful but more likely to hallucinate."
    parser.add_argument("-rf", "--render_factor", type=int, default=RENDER_FACTOR, help=rf_help)
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    for k, v in vars(args).items():
        logger.debug(f"{k:>20s}:{v}")
    if args.benchmark in SUPPORTED_BENCHMARKS:
        benchmark_dir = args.root_dir / args.benchmark
        if args.pred_step is None:
            pred_dir = benchmark_dir / "raw" / "test"
        else:
            pred_dir = benchmark_dir / args.pred_step
        emo_pred_dirs = [x for x in pred_dir.iterdir() if x.is_dir()]
        for emo_pred_dir in emo_pred_dirs:
            logger.info(f"Process {emo_pred_dir}.")
            emo_dst_dir = benchmark_dir / args.step / emo_pred_dir.name
            emo_dst_dir.mkdir(exist_ok=True, parents=True)
            cmd = [
                "python", "inference_resshift.py",
                "-i", emo_pred_dir,
                "-o", emo_dst_dir,
                "--task", "realsr",
                "--scale", "4",
                "--version", "v3",
            ]
            subprocess.check_output(cmd)
    else:
        raise NotImplementedError(f"Benchmark {args.benchmark} is not supported.")

    loguru.info(f"Finish ResShift super resolution.")
