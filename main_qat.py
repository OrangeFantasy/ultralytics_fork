import ast
import os
import sys
from argparse import ArgumentParser

import main

from tools.qat.pipeline import run_qat
from tools.qat.config import *

def add_qat_arguments(parser: ArgumentParser):
    parser.add_argument("--platform", type=str, default=None, choices=["ascend", "sophgo", "nvidia", "torchao", "rknn", "debug"])
    parser.add_argument("--qat_pretrained", type=str, default=None, help="Set qat_pretrained will skip training.")
    parser.add_argument("--sophgo_chip", type=str, default="BM1688", help="For sophgo.")
    parser.add_argument("--amct_init_config", action="store_true", help="For ascend.")

def run(args):
    os.environ["__global_args__val_period"] = "10"
    os.environ["__global_args__val_last_epochs"] = "10"
 
    config = get_qat_config__Teacher(args)
    run_qat(config)

def override_debug_params(args):
    print("==> Debug mode.")
    ball_sports_args = {
        "model": "cfg/ball_sports/models/yolov8s_25kpts_2H.yaml",
        "data": "cfg/ball_sports/datasets/25kpts_2H_Solid.yaml",
        "nkpts": 25,
        "act": "relu6",
        "epochs": 10,
        "batch": 128,
        "device": "2",
        "platform": "nvidia",
        "override_hyp": ast.literal_eval(r"{ 'plots': False, 'scale': 0.6, 'albumentations': 1.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }"),
    }
    student_qrcode_args = {
        "model": "cfg/qrcode/models/yolov8s_19kpts_MergeRaiseHandAndStandUp.yaml",
        "pretrained": "runs/multi-head/train/20260207_231833_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt",
        "data": "cfg/qrcode/datasets/19kpts_MergeRaiseHandAndStandUp.yaml",
        "nkpts": 19,
        "class_ranges": "[[0, 0], [1, 5], [2, 5], [6, 6], [7, 10], [11, 11]]",
        "act": "relu6",
        "epochs": 1,
        "batch": 128,
        "override_hyp": ast.literal_eval(r"{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }"),
        "platform": "rknn",
    }
    student_args = {
        "model": "cfg/classroom/models/yolov8s-19kpts_Student_MergeActions.yaml",
        "pretrained": "runs/multi-head/qrcode/20260312_142126_yolov8s-19kpts_Student_MergeActions/weights/last.pt",
        "data": "cfg/classroom/datasets/19kpts_Student_MergeActions.yaml",
        "nkpts": 19,
        "class_ranges": "[[0, 0], [1, 5], [2, 5], [6, 6]]",
        "act": "relu6",
        "device": "2",
        "epochs": 1,
        "batch": 128,
        "override_hyp": ast.literal_eval(r"{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }"),
        "platform": "rknn",
    }
    teacher_args = {
        "model": "cfg/classroom/models/yolov8s-17kpts_Teacher.yaml",
        "pretrained": "runs/multi-head/classroom/20260313_133533_yolov8s-17kpts_Teacher/weights/best.pt",
        "data": "cfg/classroom/datasets/17kpts_Teacher.yaml",
        "nkpts": 17,
        "act": "relu6",
        "platform": "rknn",
        "device": "2",
        "epochs": 1,
        "batch": 128,
        "override_hyp": ast.literal_eval(r"{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }"),
    }
    args.__dict__.update(teacher_args)
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
