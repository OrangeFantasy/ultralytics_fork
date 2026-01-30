import ast
import datetime
import os
import time
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch

from ultralytics import YOLO

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
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", nargs="+", type=int, default=[448, 768])
    parser.add_argument("--conf", type=float, default=0.001, help="0.001 is 0.25")
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--override_hyp", type=str, default=r"{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 30 }")

    parser.add_argument("--act", type=str, default="relu", choices=["relu", "relu6"])

    parser.add_argument("--logging", type=str, default=None, choices=["tensorboard", "wandb"])
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="If the experiment folder exists, overwrite it.")

    args = parser.parse_args()
    args.override_hyp = ast.literal_eval(args.override_hyp)
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.delay:
        print(f"==> Task will start after {args.delay} seconds.")
        for _ in tqdm(range(args.delay), total=args.delay, desc="delay", ncols=160):
            time.sleep(1)

    os.environ["__global_args__val_period"] = "5"
    os.environ["__global_args__val_last_epochs"] = "15"

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
        class_ranges = np.array([[0, 0], [1, 5], [2, 3], [4, 5], [6, 6], [7, 10]], dtype=np.int32)
        os.environ["__global_args__multi_head_class_ranges"] = np.array2string(class_ranges.reshape(-1))
        os.environ["__global_args__img2label_paths_sa"] = "zhaolixiang/dataset/multi_task_dataset/trainval/trainval_multi_task_with_action_labels/images"
        os.environ["__global_args__img2label_paths_sb"] = "yuanchengzhi/datasets/SmartClassroom/StandUp_remarked/labels"
    elif args.nkpts == 25:
        loss.OKS_SIGMA = _OKS_SIGMA[args.nkpts]
        os.environ["__global_args__oks_sigma"] = np.array2string(loss.OKS_SIGMA)
    else:
        raise ValueError(f"Unsupported number of keypoints: {args.nkpts}")

    from ultralytics.nn.modules.conv import Conv
    _DEFAULT_ACT = {
        "relu": torch.nn.ReLU(),
        "relu6": torch.nn.ReLU6()
    }
    Conv.default_act = _DEFAULT_ACT[args.act]
    print(f"==> Set Conv.default_act = {_DEFAULT_ACT[args.act]}")

    from ultralytics import settings
    if args.logging == "tensorboard":
        settings["tensorboard"] = True
        print(f"==> Set settings['tensorboard'] = True")
    elif args.logging == "wandb":
        settings["wandb"] = True
        print(f"==> Set settings['wandb'] = True")
    else:
        pass

    model = YOLO(args.model, task=args.task)
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
