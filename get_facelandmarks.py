#/usr/bin/env python3
# Standard library imports
import argparse
from pathlib import Path
import sys
# Third party imports
import numpy as np
from PIL import Image
import torch
# Local imports
from src import detect_faces


SRC_DIR = Path("/workspace/data/raw")
DST_DIR = Path("/workspace/data/processed")
IM_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff",)

def argparser():
    parser_ = argparse.ArgumentParser()
    i_help = "Path to directory of images to compute five facial landmarks for."
    o_help = "Path to output directory that provides a text file for each input image."
    parser_.add_argument("-i", "--src_dir", type=Path, default=SRC_DIR, help=i_help)
    parser_.add_argument("-o", "--dst_dir", type=Path, default=DST_DIR, help=o_help)
    return parser_.parse_args()

  
if __name__ == "__main__":
    args = argparser()
    assert args.src_dir.exists(), f"{args.src_dir} doesn't exist."
    srcs = sorted([x for x in args.src_dir.iterdir() if x.suffix in IM_SUFFIXES])
    if len(srcs) == 0:
        print(f"No files to process in {args.src_dir}.")
        sys.exit(1)
    args.dst_dir.mkdir(exist_ok=True, parents=True)
    for src in srcs:
        image = Image.open(src)
        with torch.no_grad():
            bb, lm = detect_faces(image)
        lm = lm[0].reshape((2, 5)).T
        dst = args.dst_dir / f"{src.stem}.txt"
        np.savetxt(dst, lm, fmt="%.2f")


    pass
