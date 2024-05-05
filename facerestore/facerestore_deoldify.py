#!/usr/bin/env python3
# Standard library imports
import argparse
from pathlib import Path
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

# Local imports
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.generators import gen_inference_wide
from deoldify.filters import ColorizerFilter 

torch.backends.cudnn.benchmark = True


# IO defaults
BENCHMARK = "test_dataset"
PRED_STEP = "facerestore_resshift"  # Predecessor step to use; if None then this is first step
STEP = "facerestore_deoldify"  # HuggingFace colorization and super resolution used in this script
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
    ps_help = "Name of predecessor step corresponding to sub-directory in data/benchmark dir."
    st_help = "Name of destination directory (e.g., HF for HuggingFace routine in this script)."
    parser.add_argument("-bm", "--benchmark", default=BENCHMARK, type=str, help=bm_help)
    parser.add_argument("-io", "--root_dir", default=ROOT_DIR, type=Path, help=io_help)
    parser.add_argument("-ps", "--pred_step", default=PRED_STEP, help=ps_help)
    parser.add_argument("-st", "--step", default=STEP, type=str, help=st_help)
    # Colorizer arguments
    rf_help = "Lower number is more colorful but more likely to hallucinate."
    parser.add_argument("-rf", "--render_factor", type=int, default=RENDER_FACTOR, help=rf_help)
    return parser.parse_args()


def colorize(img: Image, device: str, model) -> Image:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.tile(arr[:, :, None], 3)
    tens_l_orig, tens_l_rs = preprocess_img(arr)
    tens_l_orig, tens_l_rs = tens_l_orig.to(device), tens_l_rs.to(device)
    output_arr = postprocess_tens(tens_l_orig, model(tens_l_rs))
    return Image.fromarray((output_arr * 255).round().astype(np.uint8))


if __name__ == "__main__":
    args = argparser()
    logger.add("facerestore_deoldify.log", rotation="1 week")
    logger.info(f"Arguments")
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
        raise NotImplementedError(f"Benchmark {args.benchmark} is not supported.")
    srcs = [x for x in srcs if x.suffix in SUFFIXES]
    if len(srcs) == 0:
        logger.error("There are no images to process.")
        sys.exit(1)
    dst_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Load colorization model and send to device.")
    learn = gen_inference_wide(root_folder=Path("./"), weights_name="ColorizeStable_gen")
    colorizer_filter = ColorizerFilter(learn)

    logger.info(f"Colorize images.")
    for src in tqdm(srcs):
        emo = src.parents[0].name
        emo_subdir = dst_dir / emo
        emo_subdir.mkdir(exist_ok=True, parents=True)
        dst = emo_subdir / f"{src.stem}.png"
        src_im = Image.open(src).convert("RGB")
        colorized_im = colorizer_filter.filter(src_im, src_im, args.render_factor, post_process=True)
        colorized_im.save(dst)
    logger.info(f"Finish DeOldify colorization.")
