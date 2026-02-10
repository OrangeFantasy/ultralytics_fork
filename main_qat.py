import ast
import os
import sys
from argparse import ArgumentParser

import main

from tools.qat.pipeline import run_qat
from tools.qat.config import *

def add_qat_arguments(parser: ArgumentParser):
    parser.add_argument("--platform", type=str, default=None, choices=["ascend", "sophgo", "nvidia", "torchao", "rknn", "debug"])
    parser.add_argument("--num_calibration", type=int, default=10, help="For sophgo and nvidia.")
    parser.add_argument("--sophgo_chip", type=str, default="BM1688", help="For sophgo.")
    parser.add_argument("--amct_init_config", action="store_true", help="For amct.")

def run(args):
    os.environ["__global_args__val_period"] = "10"
    os.environ["__global_args__val_last_epochs"] = "10"
 
    config = get_sports_qrcode_qat_config(args)
    run_qat(config)

def override_debug_params(args):
    print("==> Debug mode.")
    ball_sports_args = {
        "model": "cfg/ball_sports/models/yolov8s_25kpts_2H.yaml",
        "data": "cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml",
        "nkpts": 25,
        "act": "relu6",
        "epochs": 10,
        "batch": 128,
        "device": "2",
        "override_hyp": ast.literal_eval(r"{ 'plots': False, 'scale': 0.6, 'albumentations': 1.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }"),
    }
    student_qrcode_args = {
        "model": "../cfg/qrcode/models/yolov8s_19kpts_MergeRaiseHandAndStandUp.yaml",
        "data": "../cfg/qrcode/datasets/19kpts_MergeRaiseHandAndStandUp.yaml",
        "nkpts": 19,
        "act": "relu6",
    }
    args.__dict__.update(student_qrcode_args)
    args.__dict__.update({
        "ema": False,
        "project": ".experiments"
    })

if __name__ == "__main__":
    main.on_add_extra_arguments = add_qat_arguments

    args = main.parse_args()
    if sys.gettrace() is not None or False:
        override_debug_params(args)

    main.initlize_global_args(args)
    run(args)
