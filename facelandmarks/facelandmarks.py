#/usr/bin/env python3
# Standard library imports
import argparse
from pathlib import Path
import sys
# Third party imports
from loguru import logger
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm
# Local imports
from src import detect_faces


# IO defaults
ROOT_DIR = Path("/data/")
BENCHMARK = "test_dataset"
SUPPORTED_BENCHMARKS = ("fer2013", "test_dataset",)  # test_datset is just a sampling of fer2013
SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff",)


def argparser():
    parser_ = argparse.ArgumentParser()
    # IO arguments
    # TODO: Add specifics on root directory's construction
    io_help = "Path to root data directory; refer to repository-level README for specifics."
    bm_help = "Name of benchmark dataset to process."
    parser_.add_argument("-io", "--root_dir", type=Path, default=ROOT_DIR, help=io_help)
    parser_.add_argument("-bm", "--benchmark", type=str, default=BENCHMARK, help=bm_help)
    return parser_.parse_args()

  
if __name__ == "__main__":
    args = argparser()
    logger.add("facelandmarks.log", rotation="1 week")
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.debug(f"{k}: {v}")
    if args.benchmark in SUPPORTED_BENCHMARKS:
        benchmark_dir = args.root_dir / args.benchmark
        src_dir = benchmark_dir / "facerestore"
        im_folder = ImageFolder(src_dir)
        srcs = [Path(x[0]) for x in im_folder.imgs]
        im_folder = ImageFolder(src_dir)
        dst_dir = benchmark_dir / "facelandmarks"
    else:
        logger.error(f"Benchmark {args.benchmark} is not supported")
        sys.exit(1)
    srcs = [x for x in srcs if x.suffix in SUFFIXES]
    if len(srcs) == 0:
        logger.error(f"No files to process in {args.src_dir}.")
        sys.exit(1)
    dst_dir.mkdir(exist_ok=True, parents=True)
    for src in tqdm(srcs):
        image = Image.open(src)
        with torch.no_grad():
            bb, lm = detect_faces(image)
        # TODO: Track these failures quantitatively
        # TODO: Look into how these errors can be avoided entirely
        try:
            lm = lm[0].reshape((2, 5)).T
            emo_subdir = dst_dir / src.parents[0].name
            emo_subdir.mkdir(exist_ok=True, parents=True)
            dst = emo_subdir / f"{src.stem}.txt"
            np.savetxt(dst, lm, fmt="%.2f")
        except Exception as e:
            logger.error(f"{src} failed.")

