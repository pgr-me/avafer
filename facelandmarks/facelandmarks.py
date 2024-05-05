# /usr/bin/env python3
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
BENCHMARK = "test_dataset"
STEP = "facelandmarks"  # Sub-directory to write to in facelandmarks directory
PRED_STEP = "facerestore_deoldify"  # Predecessor step to use
ROOT_DIR = Path("/data")
SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SUPPORTED_BENCHMARKS = (
    "fer2013",
    "test_dataset",
)  # test_datset is just a sampling of fer2013


def argparser():
    parser = argparse.ArgumentParser()
    # IO arguments
    # TODO: Add specifics on root directory's construction
    bm_help = "Name of benchmark dataset to process."
    io_help = (
        "Path to root data directory; refer to repository-level README for specifics."
    )
    ps_help = "Name of predecessor step corresponding to sub-directory in data/facerestore/ dir."
    st_help = "Name of destination sub-directory in data/facelandmarks/ dir."
    parser.add_argument("-bm", "--benchmark", type=str, default=BENCHMARK, help=bm_help)
    parser.add_argument("-io", "--root_dir", type=Path, default=ROOT_DIR, help=io_help)
    parser.add_argument("-ps", "--pred_step", default=PRED_STEP, type=str, help=ps_help)
    parser.add_argument("-st", "--step", default=STEP, type=str, help=st_help)
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    logger.add("facelandmarks.log", rotation="1 week")
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.debug(f"{k}: {v}")
    if args.benchmark in SUPPORTED_BENCHMARKS:
        benchmark_dir = args.root_dir / args.benchmark
        if args.pred_step is None:
            pred_dir = benchmark_dir / "raw" / "test"
        else:
            pred_dir = benchmark_dir / args.pred_step
        im_folder = ImageFolder(pred_dir)
        srcs = [Path(x[0]) for x in im_folder.imgs]
        dst_dir = benchmark_dir / args.step
    else:
        logger.error(f"Benchmark {args.benchmark} is not supported")
        sys.exit(1)
    srcs = [x for x in srcs if x.suffix in SUFFIXES]
    if len(srcs) == 0:
        logger.error(f"No files to process in {args.src_dir}.")
        sys.exit(1)
    dst_dir.mkdir(exist_ok=True, parents=True)
    for src in tqdm(srcs):
        emo = src.parents[0].name
        emo_subdir = dst_dir / emo
        emo_subdir.mkdir(exist_ok=True, parents=True)
        image = Image.open(src)
        with torch.no_grad():
            bb, lm = detect_faces(image)
        # TODO: Track these failures quantitatively
        # TODO: Look into how these errors can be avoided entirely
        try:
            lm = lm[0].reshape((2, 5)).T
            dst = emo_subdir / f"{src.stem}.txt"
            np.savetxt(dst, lm, fmt="%.2f")
        except Exception as e:
            logger.error(f"{src} failed.")
