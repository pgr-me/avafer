#!/usr/bin/env python3
# Standard library imports
import argparse
from pathlib import Path
import sys
# Third party imports
import cv2
from loguru import logger
import pandas as pd
from rmn import RMN
from tqdm import tqdm

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff",)
DUMMY_DI = dict(
    xmin=-1, ymin=-1, xmax=-1, ymax=-1, emo_label="", emo_proba=None, angry=None,
    disgust=None, fear=None, happy=None, sad=None, surprise=None, neutral=None
)


def argparser():
    parser = argparse.ArgumentParser(prog="rmn_inference")
    i_help = "Path to images to perform facial expression recognition on."
    o_help = "Path to results CSV, including prediction."
    parser.add_argument("-i", "--src_dir", type=Path, required=True, help=i_help)
    parser.add_argument("-o", "--results_dst", type=Path, required=True, help=o_help)
    return parser.parse_args()


def run(args_: argparse.Namespace) -> pd.DataFrame:
    m = RMN()
    successes = 0
    li = []
    srcs = [x for x in args_.src_dir.iterdir() if x.suffix in IMAGE_SUFFIXES]
    if len(srcs) == 0:
        logger.error(f"No files to process in {args.src_dir}.")
    for src in tqdm(srcs, desc="Inference progress"):
        di = dict(fn=src.name)
        try:
            image = cv2.imread(str(src))
            results_ = m.detect_emotion_for_single_frame(image)
            di.update({k:v for k, v in results_[0].items() if k != "proba_list"})
            for emo in results_[0]["proba_list"]:
                di.update(emo)
            successes += 1
        except Exception as e:
            logger.error(f"FER inference failed for {src.name}.")
            logger.error(e)
            di.update(DUMMY_DI)
        li.append(di)
    logger.info(f"Inference successful for {successes} of {len(srcs)} images.")
    return pd.DataFrame(li)


if __name__ == "__main__":
    logger.add("fer_rmn.log", rotation="1 week")
    args = argparser()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}:\t{v}")
    args.results_dst.parents[0].mkdir(exist_ok=True, parents=True)
    results = run(args) 
    results.to_csv(args.results_dst)
    logger.info("Finished RMN FER inference.")
