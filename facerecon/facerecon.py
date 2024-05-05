"""This script is a modified version of the test script for Deep3DFaceRecon_pytorch
"""
# Standard library imports
import argparse
import os
from pathlib import Path
import sys
from typing import Tuple

# Third party imports
from loguru import logger
from loguru._logger import Logger
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, savemat
import torch
from torchvision.datasets import DatasetFolder, ImageFolder
from tqdm import tqdm

# Local imports
from data import create_dataset
from data.flist_dataset import default_flist_reader
from options.test_options import TestOptions
from models import create_model
from util.load_mats import load_lm3d
from util.preprocess import align_img
from util.util import tensor2im
from util.visualizer import MyVisualizer

# IO defaults
# TODO: Fix the CLI so that arguments actually work...remove the original args class repo used and replace with argparser
BENCHMARK = "fer2013"  # "test_dataset"
NAME = "facerecon_20230425"
PRED_STEP = "facelandmarks_mtcnn"
ROOT_DIR = Path("/data")
STEP = "facerecon_deep3dfacerecon"
SUPPORTED_BENCHMARKS = ("fer2013", "test_dataset")
# Inference defaults
DEVICE = 0
EPOCH = 20


def argparser():
    parser = argparse.ArgumentParser()
    # IO arguments
    bm_help = "Benchmark dataset to process."
    io_help = "Path to root data directory."
    na_help = "Face reconstruction model subdirectory in checkpoints/ directory."
    ps_help = (
        "Name of predecessor step corresponding to sub-directory in data/facelandmarks."
    )
    st_help = "Name of sub-directory, located in data/facerecon/, for this step."
    parser.add_argument("-bm", "--benchmark", default=BENCHMARK, type=str, help=bm_help)
    parser.add_argument("-io", "--root_dir", default=ROOT_DIR, type=Path, help=io_help)
    parser.add_argument("-na", "--name", default=NAME, type=str, help=na_help)
    parser.add_argument("-ps", "--pred_step", default=PRED_STEP, type=str, help=ps_help)
    parser.add_argument("-st", "--step", default=STEP, type=str, help=st_help)
    # Inference arguments
    dv_help = "Device to use; choose -1 to use CPU."
    ep_help = "Pretrained face reconstruction model epoch."
    parser.add_argument("-dv", "--device", default=DEVICE, type=int, help=dv_help)
    parser.add_argument("-ep", "--epoch", default=EPOCH, type=int, help=ep_help)
    # Tack on library test options
    test_opts = vars(TestOptions().parse())
    for k, v in test_opts.items():
        if k not in ("name", "epoch"):
            parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser.parse_args()


def _loader(src: Path) -> str:
    """
    Helper function to read landmark text files.
    Arguments:
        src: Path to landmarks file.
    Returns: Facial landmarks.
    Used by run function.
    """
    with open(src) as f:
        return f.read()


def read_data(
    im_path: str, lm_path: str, lm3d_std: NDArray, to_tensor: bool = True
) -> Tuple:
    """
    Read image and landmark files into memory.
    Arguments:
        im_path: Path to image file.
        lm_path: Path to landmarks file.
        lm3d_std: Used to align image.
        to_tensor: True to convert to tensor.
    Returns: Image and landmarks.
    Used by run function.
    """
    # to RGB
    im = Image.open(im_path).convert("RGB")
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = (
            torch.tensor(np.array(im) / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def run(args: argparse.Namespace, logger: Logger):
    """
    Perform 3D facial reconstruction.
    Arguments:
        args: CLI arguments.
        logger: Loguru logger.
    Uses _loader and read_data functions.
    """
    if torch.cuda.is_available() and (args.device >= 0):
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    model = create_model(args)
    model.setup(args)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(args)
    if args.benchmark in SUPPORTED_BENCHMARKS:
        benchmark_dir = args.root_dir / args.benchmark
        im_src_dir = benchmark_dir / "facerestore" / args.pred_step
        lm_src_dir = benchmark_dir / "facelandmarks" / args.pred_step
        try:
            im_folder = ImageFolder(im_src_dir)
        except Exception as e:
            logger.error(
                f"{im_src_dir} does not exist. Run the facerestore step to generate."
            )
            logger.error(e)
            sys.exit(1)
        try:
            lm_folder = DatasetFolder(lm_src_dir, _loader, extensions=("txt", ".txt"))
        except Exception as e:
            logger.error(
                f"{lm_src_dir} does not exist. Run the facelandmarks step to generate."
            )
            logger.error(e)
            sys.exit(1)
        im_srcs = sorted([Path(x[0]) for x in im_folder.imgs])
        lm_srcs = sorted([Path(x[0]) for x in lm_folder.samples])
        im_srcs_di = dict(zip([(x.parents[0].name, x.stem) for x in im_srcs], im_srcs))
        lm_srcs_di = dict(zip([(x.parents[0].name, x.stem) for x in lm_srcs], lm_srcs))
        stems = sorted(set(im_srcs_di.keys()).intersection(set(lm_srcs_di.keys())))
        src_di = {stem: (im_srcs_di[stem], lm_srcs_di[stem]) for stem in stems}
    else:
        logger.error(f"Benchmark {args.benchark} is not supported.")
        sys.exit(1)
    lm3d_std = load_lm3d(args.bfm_folder)
    dst_dir = benchmark_dir / "facerecon" / args.step
    dst_dir.mkdir(exist_ok=True, parents=True)
    # TODO: Use datasets and dataloaders so this can be done more efficiently
    for k_tuple, v_tuple in tqdm(src_di.items()):
        emo_class, obs_id = k_tuple
        im_src, lm_src = v_tuple
        im_tensor, lm_tensor = read_data(str(im_src), str(lm_src), lm3d_std)
        data = {"imgs": im_tensor, "lms": lm_tensor}
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        triple_image = model.get_current_visuals()["output_vis"]  # get image results
        triple_arr = tensor2im(triple_image[0])
        height, triple_width, channels = triple_arr.shape
        width = triple_width // 3
        arr = triple_arr[:, width : 2 * width, :]  # Extract center image
        im = Image.fromarray(arr)
        # Save outputs
        emo_subdir = dst_dir / emo_class
        emo_subdir.mkdir(exist_ok=True, parents=True)
        im_dst = emo_subdir / f"{obs_id}.png"
        mesh_dst = emo_subdir / f"{obs_id}.obj"
        coeff_dst = emo_subdir / f"{obs_id}.mat"
        im.save(im_dst)
        model.save_mesh(mesh_dst)  # save reconstruction meshes
        model.save_coeff(coeff_dst)  # save predicted coefficients


if __name__ == "__main__":
    logger.add("facerecon.log", rotation="1 week")
    args = argparser()
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.debug(f"{k}: {v}")
    run(args, logger)
    logger.info("Finished facerecon step.")
