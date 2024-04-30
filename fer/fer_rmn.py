#!/usr/bin/env python3
# Standard library imports
import argparse
import json
from pathlib import Path
import sys
# Third party imports
import cv2
from loguru import logger
import pandas as pd
from rmn import RMN
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# IO defaults
BENCHMARK = "test_dataset"
PRED_STEP = "HF"
ROOT_DIR = Path("/data")
STEP = "HF-RMN"
SUPPORTED_BENCHMARKS = ("fer2013", "test_dataset")
SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff",)
# Inference defaults
DUMMY_DI = dict(
    xmin=-1, ymin=-1, xmax=-1, ymax=-1, emo_label="", emo_proba=None, angry=None,
    disgust=None, fear=None, happy=None, sad=None, surprise=None, neutral=None,
    processed_ok=False,
)


def argparser():
    parser = argparse.ArgumentParser(prog="rmn_inference")
    # IO arguments
    bm_help = "Name of benchmark dataset to process."
    io_help = "Path to root data directory; refer to repository-level README for specifics."
    ps_help = "Name of predecessor step corresponding to sub-directory in data/facerecon/ dir."
    st_help = "Name of destination sub-directory in data/fer/ dir."
    parser.add_argument("-bm", "--benchmark", type=str, default=BENCHMARK, help=bm_help)
    parser.add_argument("-io", "--root_dir", type=Path, default=ROOT_DIR, help=io_help)
    parser.add_argument("-ps", "--pred_step", default=PRED_STEP, type=str, help=ps_help)
    parser.add_argument("-st", "--step", default=STEP, type=str, help=st_help)
    # Inference arguments
    return parser.parse_args()


def run(args_: argparse.Namespace, logger_) -> pd.DataFrame:
    m = RMN()
    li = []
    if args.benchmark in SUPPORTED_BENCHMARKS:
        benchmark_dir = args.root_dir / args.benchmark
        src_dir = benchmark_dir / "facerecon" / args.pred_step
        raw_dir = benchmark_dir / "raw" / "test"  # need to include faces that didn't process
        try:
            im_folder = ImageFolder(src_dir)
            raw_im_folder = ImageFolder(raw_dir)
        except Exception as e:
            logger.error(f"{src_dir} does not exist. Run the facerestore step to generate.")
            logger.error(e)
            sys.exit(1)
        im_srcs = sorted([Path(x[0]) for x in im_folder.imgs])
        raw_im_srcs = sorted([Path(x[0]) for x in raw_im_folder.imgs])
        im_srcs_di = dict(zip([(x.parents[0].name, x.stem) for x in im_srcs], im_srcs))
        raw_im_srcs_di = dict(zip([(x.parents[0].name, x.stem) for x in raw_im_srcs], raw_im_srcs))
        im_folder = ImageFolder(src_dir)
        dst_dir = benchmark_dir / "fer" / args.step
    if len(im_srcs_di) == 0:
        logger_.error(f"No files to process in {args.src_dir}.")
        sys.exit(1)
    dst_dir.mkdir(exist_ok=True, parents=True)
    for k_tuple, src in tqdm(raw_im_srcs_di.items(), desc="FER inference progress"):
        y, obs_id = k_tuple
        di = dict(obs_id=src.stem, y=y)
        # Predict / infer when there is a reconstructed image
        if k_tuple in im_srcs_di:
            try:
                image = cv2.imread(str(src))
                results_ = m.detect_emotion_for_single_frame(image)
                di.update({k:v for k, v in results_[0].items() if k != "proba_list"})
                for emo in results_[0]["proba_list"]:
                    di.update(emo)
                di.update({"processed_ok": True})
            except Exception as e:
                logger.error(f"{k_tuple} failed.")
                logger.error(e)
                di.update(DUMMY_DI)
        # Case when raw image wasn't processed
        else:
            di.update(DUMMY_DI)
        li.append(di)
    logger.info("Finished inference: Aggregate results.")
    results = pd.DataFrame(li).rename(columns=dict(emo_label="yhat"))
    results["correct"] = results["y"] == results["yhat"]
    class_totals = results.groupby("y")["correct"].count().to_dict()
    class_totals = {k: int(v) for k, v in class_totals.items()}
    class_correct_totals = results.groupby("y")["correct"].sum().to_dict()
    class_correct_totals = {k: int(v) for k, v in class_correct_totals.items()}
    class_correct_acc = {k:float((class_correct_totals[k] / class_totals[k])) for k in class_totals}
    total_acc = results["correct"].sum() / len(results)
    mask = results["processed_ok"]
    proc_acc = (results["correct"][mask]).sum() / mask.sum()
    summary = dict(
        n=len(results),
        n_proc=int(results["processed_ok"].sum()),
        acc=float(total_acc),
        proc_acc=float(proc_acc),
        class_totals=class_totals,
        class_correct_totals=class_correct_totals,
        class_correct_acc=class_correct_acc,
    )
    logger.debug(f"n={len(results)}, acc={total_acc:.4f}.")
    logger.info(f"Save outputs to {dst_dir}.")
    results_dst = dst_dir / "results.csv"
    summary_dst = dst_dir / "summary.json"
    results.to_csv(results_dst, index=False)
    with open(summary_dst, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    logger.add("fer_rmn.log", rotation="1 week")
    args = argparser()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}:\t{v}")
    run(args, logger)
    logger.info("Finished RMN FER inference.")
