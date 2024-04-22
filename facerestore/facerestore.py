#!/usr/bin/env python3
# Standard library imports
import argparse
from pathlib import Path
import sys
from typing import Tuple
# Third party imports
import numpy as np
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
from diffusers.utils import load_image, make_image_grid
import torch
from tqdm import tqdm
# Local imports
import colorizers as c
from colorizers.util import postprocess_tens, preprocess_img


# IO defaults
SRC_DIR = Path("/workspace/data/raw")
DST_DIR = Path("/workspace/data/interim/facerestore")
SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# Colorizer defaults
COLORIZER_MODELS = ("eccv16", "siggraph17")
COLORIZER_MODEL = "eccv16"
# SR defaults
SR_MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"
IN_SIZE = OUT_SIZE = 128
N_INF_STEPS = 100
ETA = 1


def argparser():
    prog = "FaceRestoration"
    parser = argparse.ArgumentParser(
        prog="FaceRestoration",
        description="Restore images of grayscale faces using colorization and super resolution."
    )
    # IO arguments
    i_help = "Path to input directory of grayscale, low-resolution images."
    o_help = "Path to output directory of in-color, higher-resolution images."
    parser.add_argument("-i", "--src_dir", default=SRC_DIR, type=Path, help=i_help)
    parser.add_argument("-o", "--dst_dir", default=DST_DIR, type=Path, help=o_help)
    # Colorizer arguments
    cm_help = "Specify `eccv16` or `siggraph17` as your colorizer model."
    parser.add_argument("-cm", "--colorizer_model", choices=COLORIZER_MODELS, default=COLORIZER_MODEL, type=str, help=cm_help)
    # SR arguments
    et_help = "Random amount of scaled noise to mix into each timestep."
    id_help = "HuggingFace diffusion model ID."
    is_help = "Resize for input to super resolution pipeline."
    ns_help = "Number of inference steps to run using super resolution diffusion model."
    os_help = "Resize after super resolution."
    parser.add_argument("-et", "--eta", default=ETA, type=float, help=et_help)
    parser.add_argument("-id", "--sr_model_id", default=SR_MODEL_ID, type=str, help=id_help)
    parser.add_argument("-is", "--in_size", default=IN_SIZE, type=int, help=is_help)
    parser.add_argument("-ns", "--n_inf_steps", default=N_INF_STEPS, type=int, help=ns_help)
    parser.add_argument("-os", "--out_size", default=OUT_SIZE, type=int, help=os_help)
    return parser.parse_args()


def colorize(img: Image, device: str, model) -> Image:
    arr = np.asarray(img)
    if(arr.ndim == 2):
        arr = np.tile(arr[:,:,None], 3)
    tens_l_orig, tens_l_rs = preprocess_img(arr)
    tens_l_orig, tens_l_rs = tens_l_orig.to(device), tens_l_rs.to(device)
    output_arr = postprocess_tens(tens_l_orig, model(tens_l_rs))
    return Image.fromarray((output_arr * 255).round().astype(np.uint8))


def sr(
        img: Image,
        pipeline: LDMSuperResolutionPipeline,
        in_resize_: Tuple=(128, 128),
        out_resize_: Tuple=(128, 128),
        n_inf_steps: int=100,
        eta: float=1
    ) -> Image:
    img = img.convert("RGB").resize(in_resize_)
    # run pipeline in inference (sample random noise and denoise)
    return pipeline(img, num_inference_steps=n_inf_steps, eta=eta).images[0].resize(out_resize_)


if __name__ == "__main__":
    args = argparser()
    srcs = [x for x in args.src_dir.iterdir() if x.suffix in SUFFIXES]
    args.dst_dir.mkdir(exist_ok=True, parents=True)
    if len(srcs) == 0:
        print("There are no images to process.")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load colorization model and send to device
    if args.colorizer_model == "eccv16":
        colorizer_model = c.eccv16(pretrained=True).eval()
    else:
        colorizer_model = c.siggraph17(pretrained=True).eval()
    colorizer_model = colorizer_model.to(device)

    # Load super resolution pipeline and send to device 
    in_resize = (args.in_size, args.in_size)
    out_resize = (args.out_size, args.out_size)
    sr_pipeline = LDMSuperResolutionPipeline.from_pretrained(args.sr_model_id)
    sr_pipeline = sr_pipeline.to(device)
    
    print(f"Transform images.")
    for src in tqdm(srcs):
        src_im = Image.open(src).convert("RGB")
        rgb_im = colorize(src_im, device, colorizer_model)
        sr_im = sr(rgb_im, sr_pipeline, in_resize, out_resize, args.n_inf_steps, args.eta)
        dst = args.dst_dir / f"{src.stem}.png"
        sr_im.save(dst)

