from typing import Callable

import ast
import datetime
import os
import sys
import time
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch

from tools.fast_math import batch_mahalanobis_for_rle
torch.distributions.multivariate_normal._batch_mahalanobis = batch_mahalanobis_for_rle

from ultralytics import YOLO

on_add_extra_arguments: Callable[[ArgumentParser], None] = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--delay", type=int, default=0)

    parser.add_argument("--task", type=str, default="multi-head")
    parser.add_argument("--model", type=str, default="cfg/ball_sports/models/yolov8s_25kpts_2H.yaml")
    parser.add_argument("--pretrained", type=str, default="", help="a .pt file or a .state_dict.pt file")
    parser.add_argument("--data", type=str, default="cfg/ball_sports/datasets/25kpts_2H_Solid.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"])

    parser.add_argument("--nkpts", type=int, default=25)
    parser.add_argument("--teacher_view", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", nargs="+", type=int, default=[448, 768])
    parser.add_argument("--conf", type=float, default=0.001, help="0.001 is 0.25")
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--override_hyp", type=str, default=r"{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 30 }")

    parser.add_argument("--act", type=str, default="relu", choices=["relu", "relu6"])

    parser.add_argument("--sparse", action="store_true", default=False)
    parser.add_argument("--sparse_mode", type=int, default=0, choices=[0, 1, 2, 3], help="0: 4:2 input, 1: 4:2 output, 2: 16:4 input-output, 3: 16:4 output-input")

    parser.add_argument("--logging", type=str, default=None, choices=["tensorboard", "wandb"])
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="If the experiment folder exists, overwrite it.")

    if on_add_extra_arguments is not None:
        on_add_extra_arguments(parser)

    args = parser.parse_args()
    args.override_hyp = ast.literal_eval(args.override_hyp)
    return args

def initlize_global_args(args):
    if args.delay:
        print(f"==> Task will start after {args.delay} seconds.")
        for _ in tqdm(range(args.delay), total=args.delay, desc="delay", ncols=160):
            time.sleep(1)

    # Set logging.
    from ultralytics import settings
    if args.logging == "tensorboard":
        settings["tensorboard"] = True
        settings["wandb"] = False
        print(f"==> Set settings['tensorboard'] = True")
    elif args.logging == "wandb":
        settings["wandb"] = True
        print(f"==> Set settings['wandb'] = True")

    # Set oks sigma and dataset path for different tasks.
    from ultralytics.utils import loss
    _OKS_SIGMA = {
        17: np.array([
                0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89
            ], dtype=np.float32) / 10.0,
        19: np.array([
                0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 
                0.85, 0.85
            ], dtype=np.float32) / 10.0,
        25: np.array([
                0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 
                0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.85, 0.85
            ], dtype=np.float32) / 10.0,
    }

    if args.nkpts == 17:
        loss.OKS_SIGMA = _OKS_SIGMA[args.nkpts]
        os.environ["__global_args__oks_sigma"] = np.array2string(loss.OKS_SIGMA)
    elif args.nkpts == 19:
        loss.OKS_SIGMA = _OKS_SIGMA[args.nkpts]
        os.environ["__global_args__oks_sigma"] = np.array2string(loss.OKS_SIGMA)
        # class_ranges = np.array([[0, 0], [1, 5], [2, 5], [6, 6], [7, 10], [11, 11]], dtype=np.int32)
        class_ranges = np.array([[0, 0], [1, 5], [2, 3], [4, 5], [6, 6], [7, 10], [11, 11]], dtype=np.int32)
        os.environ["__global_args__multi_head_class_ranges"] = np.array2string(class_ranges.reshape(-1))
        os.environ["__global_args__img2label_paths_sa"] = "zhaolixiang/dataset/multi_task_dataset/trainval/trainval_multi_task_with_action_labels/images"
        os.environ["__global_args__img2label_paths_sb"] = "yuanchengzhi/datasets/SmartClassroom/StandUp_remarked/labels"
    elif args.nkpts == 25:
        loss.OKS_SIGMA = _OKS_SIGMA[args.nkpts]
        os.environ["__global_args__oks_sigma"] = np.array2string(loss.OKS_SIGMA)
    else:
        raise ValueError(f"Unsupported number of keypoints: {args.nkpts}")

    # Set default activation function for Conv.
    from ultralytics.nn.modules.conv import Conv, ConvTranspose
    _DEFAULT_ACT = {
        "relu": torch.nn.ReLU(),
        "relu6": torch.nn.ReLU6()
    }
    Conv.default_act = _DEFAULT_ACT[args.act]
    ConvTranspose.default_act = _DEFAULT_ACT[args.act]
    print(f"==> Set default_act = {_DEFAULT_ACT[args.act]}")

    # Set validation period and last epochs.
    os.environ["__global_args__val_period"] = "5"
    os.environ["__global_args__val_last_epochs"] = "15"

    return args

def run(args):
    # Initialize YOLO model and parameters.
    model = YOLO(args.model, task=args.task)
    if args.sparse:
        from tools.sparsity import sparsity_model
        always_disallowed_layer_names = [
            "model.21.flow_model.s.0.2", "model.21.flow_model.s.0.4", 
            "model.21.flow_model.s.1.2", "model.21.flow_model.s.1.4",
            "model.21.flow_model.s.2.2", "model.21.flow_model.s.2.4",
            "model.21.flow_model.s.3.2", "model.21.flow_model.s.3.4",
            "model.21.flow_model.s.4.2", "model.21.flow_model.s.4.4",
            "model.21.flow_model.s.5.2", "model.21.flow_model.s.5.4",
            "model.21.flow_model.t.0.2", "model.21.flow_model.t.0.4",
            "model.21.flow_model.t.1.2", "model.21.flow_model.t.1.4",
            "model.21.flow_model.t.2.2", "model.21.flow_model.t.2.4",
            "model.21.flow_model.t.3.2", "model.21.flow_model.t.3.4",
            "model.21.flow_model.t.4.2", "model.21.flow_model.t.4.4",
            "model.21.flow_model.t.5.2", "model.21.flow_model.t.5.4",
            "model.21.cv4_sigma.0", "model.21.cv4_sigma.1", "model.21.cv4_sigma.2",
        ] 
        custom_disallowed_layer_names = [
            # "model.21.cv3.1.0.1.conv", "model.21.cv3.1.1.1.conv", "model.21.cv3.1.2.1.conv", 
            # "model.21.cv3.1.0.2", "model.21.cv3.1.1.2", "model.21.cv3.1.2.2",
            # "model.21.cv4.0.1.conv", "model.21.cv4.1.1.conv", "model.21.cv4.2.1.conv", 
            # "model.21.cv4_kpts.0", "model.21.cv4_kpts.1", "model.21.cv4_kpts.2",
        ]
        def on_pretrain_routine_end(trainer) -> None:
            print("==> Sparsity model ...")
            sparsity_model(
                trainer.model, trainer.optimizer, 
                mode=args.sparse_mode, verbose=2, 
                disallowed_layer_names=always_disallowed_layer_names + custom_disallowed_layer_names, 
                fast=True
            )
        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

    kwargs = {
        "data": args.data, "val": True,
        "epochs": args.epochs, "device": args.device, 
        "batch": args.batch, "workers": args.workers, "imgsz": args.imgsz,
        "conf": args.conf, "iou": args.iou, 
        "cache": False, "patience": 0, "plots": True, "save": True, "save_period": 20, "verbose": True, 
        "exist_ok": args.overwrite,
        "project": args.project or f"{args.mode}", 
        "name": datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + (args.name or args.model.split("/")[-1].split(".")[0]),
    }
    kwargs.update(args.override_hyp)

    # Train or val.
    if args.mode == "train":
        custom_state_dict = None
        if args.pretrained:
            if args.pretrained.endswith(".state_dict.pt"):
                assert args.model.rsplit(".")[-1] == "yaml", "Make sure the model is a .yaml file."
                custom_state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=True)["state_dict"]
            elif args.pretrained.endswith(".pt"):
                kwargs["pretrained"] = args.pretrained
            else:
                print(f"==> Pretrained model {args.pretrained} is not a .state_dict.pt or .pt file.")
        model.train(**kwargs, state_dict=custom_state_dict)
    elif args.mode == "val":
        model.val(**kwargs)

def override_debug_params(args):
    print("==> Debug mode.")
    student_qrcode_args = {
        "model": "cfg/qrcode/models/yolov8s_19kpts_MergeRaiseHandAndStandUp.yaml",
        "data": "cfg/qrcode/datasets/19kpts_MergeRaiseHandAndStandUp.yaml",
        "pretrained": "cfg/qrcode/models/converted_qrcode_0112.fp16.state_dict.pt",
        "nkpts": 19,
        "device": "3",
        "override_hyp": ast.literal_eval(r"{ 'plots': False, 'mosaic': 0 }")
    }
    ball_sports_args = {
        # "model": "cfg/ball_sports/models/yolov8s_25kpts_2H.yaml",
        "model": "runs/multi-head/train/20260131_165216_best/weights/best.pt",
        "data": "cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml",
        "nkpts": 25,
        "act": "relu6",
        "device": "2",
        "override_hyp": ast.literal_eval(r"{ 'plots': False, 'scale': 0.6, 'albumentations': 1.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }"),
        "sparse": True,
        "sparse_mode": 2,
    }
    running_args = {
        "model": "cfg/running/models/yolov8s-25-poseAngles-m.yaml",
        "data": "cfg/running/datasets/25kpts.yaml",
        "pretrained": "/data5/xieyangyang/MultiTaskDetector/runs/SportsDets_poseAngles_m/25points6/weights/best.fp16.state_dict.pt",
        "nkpts": 25,
        "act": "relu6",
        "device": "2",
        "override_hyp": ast.literal_eval(r"{ 'plots': False, }"),
        "sparse": False,
        "sparse_mode": 0,
        "epochs": 1,
    }
    args.__dict__.update(running_args)
    args.__dict__.update({
        "project": ".experiments"
    })

if __name__ == "__main__":
    args = parse_args()
    if sys.gettrace() is not None or False:
        override_debug_params(args)

    initlize_global_args(args)
    run(args)
