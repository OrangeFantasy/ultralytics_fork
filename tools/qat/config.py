from functools import partial

import datetime
import torch

from .pipeline import CalibrateConfig, ExportConfig, ValConfig, QAT_Config, QAT_Functions

def get_overrides(args):
    kwargs = {
        "task"  : args.task,   "model" : args.model,  "data" : args.data,
        "device": args.device, "epochs": args.epochs, "batch": args.batch, "workers": args.workers,
        "imgsz" : args.imgsz,  "conf"  : args.conf,   "iou"  : args.iou,   "max_det": 300,
        "warmup_epochs": 0,    "single_cls": False,
        "amp"   : False,       "ema"   : False,        
        "optimizer": "SGD",    "lr0"   : 0.01 * 0.01, "lrf"  : 0.05,
        "mosaic": 0.0,         "fliplr": 0.5,         "scale": 0.6,         
        "plots" : True,        "save_period": 5,      "val"  : False,
        "project": args.project if args.project is not None else "qat",
        "name": datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + (args.name or args.model.split("/")[-1].split(".")[0]),
    }
    kwargs.update(args.override_hyp)
    return kwargs

def _default_forward_function(self, x: list[torch.Tensor], ff_cat: bool = False) -> tuple[torch.Tensor]:
    cat_function = self.ff_cat if ff_cat else torch.cat

    boxes = [
        [self.cv2[head_idx][i](x[i]) for i in range(self.nl)] 
        for head_idx in range(self.n_heads)
    ]
    scores  = [
        [self.cv3[head_idx][i](x[i]) for i in range(self.nl)] 
        for head_idx in range(self.n_heads)
    ]

    kpts = None
    if self.n_pose_heads > 0:
        pose_feats = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpts = [self.cv4_kpts[i](pose_feats[i]) for i in range(self.nl)]

    angles = None
    if self.n_angle_heads > 0:
        angles = [
            [self.cv5[h][i](x[i]) for i in range(self.nl)] 
            for h in range(self.n_angle_heads)
        ]

    preds = []
    angle_idx = 0
    for head_idx, head_type in enumerate(self.heads):
        for i in range(self.nl):
            pred = [boxes[head_idx][i], scores[head_idx][i]]
            if   head_type == "pose":
                pred.append(kpts[i])
            elif head_type == "angle":
                pred.append(angles[angle_idx][i])
            elif head_type == "pose-angle":
                pred.append(kpts[i])
                pred.append(angles[angle_idx][i])
            preds.append(cat_function(pred, 1))
        if head_type in ["angle", "pose-angle"]:
            angle_idx += 1

    if self.export:
        return (*preds, )
    return (*x, *preds)

def _default_forward_qat(self, preds: tuple[torch.Tensor]) -> dict[str, torch.Tensor | None]:
    bs = preds[0].shape[0]

    feats = preds[:self.nl]
    head_preds = [
        preds[self.nl + i*self.nl : self.nl + (i+1)*self.nl]
        for i in range(self.n_heads)
    ]

    boxes, scores, kpts, angles = [], [], [], []
    for head_idx, head_type in enumerate(self.heads):
        curr_head_preds = head_preds[head_idx]
        split_sizes = [4 * self.reg_max, self.nc_per_head[head_idx]]
        if head_type in ["pose", "pose-angle"]:
            split_sizes.append(self.nk)
        if head_type in ["angle", "pose-angle"]:
            split_sizes.append(self.n_angles)
        splitted = [pred.split(split_sizes, 1) for pred in curr_head_preds]  # box, cls, kpts (optional), angle (optional)

        boxes.append(torch.cat([splitted[i][0].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1))
        scores.append(torch.cat([splitted[i][1].view(bs, self.nc_per_head[head_idx], -1) for i in range(self.nl)], dim=-1))
        if head_type == "angle":
            angles.append(torch.cat([splitted[i][2].view(bs, self.n_angles, -1).sigmoid() for i in range(self.nl)], dim=-1))
        elif head_type == "pose":
            kpts.append(torch.cat([splitted[i][2].view(bs, self.nk, -1) for i in range(self.nl)], dim=-1))
        elif head_type == "pose-angle":
            kpts.append(torch.cat([splitted[i][2].view(bs, self.nk, -1) for i in range(self.nl)], dim=-1))
            angles.append(torch.cat([splitted[i][3].view(bs, self.n_angles, -1).sigmoid() for i in range(self.nl)], dim=-1))

    return dict(
        feats=list(feats),
        boxes=torch.cat(boxes, dim=1),
        scores=torch.cat(scores, dim=1),
        kpts=torch.cat(kpts, dim=1) if kpts else None,
        angles=torch.cat(angles, dim=1) if angles else None,
    )

def _default_inference_qat(self, preds):
    preds = self.forward_qat(preds)
    y = self._inference(preds)
    return (y, preds)

def get_qat_config__BallSports(args):
    config = QAT_Config(
        platform=args.platform or "rknn",
        overrides=get_overrides(args),
        model_fp_weights=args.pretrained or "runs/multi-head/train/20260131_165216_best/weights/best.pt",
        # model_qat_weights="/data4/yuanchengzhi/projects/ultralytics_fork/runs/multi-head/.experiments/20260211_180945_yolov8s_25kpts_2H/weights/last.pt",
        skip_train=False,
        val_config=ValConfig(
            enable=True,
        ),
        export_config=ExportConfig(
            enable=True,
            input_names=["input"],
            output_names=[
                "head_body_56x96", "head_body_28x48", "head_body_14x24",
                "ball_pole_56x96", "ball_pole_28x48", "ball_pole_14x24",
            ],
        ),
        qat_functions=QAT_Functions(
            forward=_default_forward_function,
            forward_qat=_default_forward_qat,
            inference_qat=_default_inference_qat,
        ),
    )
    return config

def get_qat_config__Code_Match_MergeActions(args):
    config = QAT_Config(
        platform=args.platform or "Ascend",
        overrides=get_overrides(args),
        model_fp_weights=args.pretrained or "../runs/multi-head/train/20260207_231833_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt",
        # model_qat_weights="/data4/yuanchengzhi/projects/yolo/MultiTaskDetector/runs/multi-head/qat/20260209_165633_yolov8s_25kpts_2H/weights/last.pt",
        val_config=ValConfig(
            enable=False,
        ),
        export_config=ExportConfig(
            enable=True,
            input_names=["input"],
            output_names=[
                "head_96", "head_48", "head_24",
                "body_96", "body_48", "body_24",
                "action_96", "action_48", "action_24",
                "face_96", "face_48", "face_24",
                "code_96", "code_48", "code_24",
                "match_96", "match_48", "match_24",
            ],
        ),
        qat_functions=QAT_Functions(
            forward=_default_forward_function,
            forward_qat=_default_forward_qat,
            inference_qat=_default_inference_qat,
        ),
    )

    return config

def get_qat_config__Code_Match(args):
    config = QAT_Config(
        platform=args.platform or "Ascend",
        overrides=get_overrides(args),
        model_fp_weights=args.pretrained or "../runs/multi-head/train/20260207_231833_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt",
        model_qat_weights="/data4/yuanchengzhi/projects/yolo/MultiTaskDetector/runs/multi-head/qat/20260210_200148_yolov8s_19kpts/weights/last.pt",
        val_config=ValConfig(
            enable=False,
        ),
        export_config=ExportConfig(
            enable=True,
            input_names=["input"],
            output_names=[
                "head_96", "head_48", "head_24",
                "body_96", "body_48", "body_24",
                "raise_hand_96", "raise_hand_48", "raise_hand_24",
                "stand_up_96", "stand_up_48", "stand_up_24",
                "face_96", "face_48", "face_24",
                "code_96", "code_48", "code_24",
                "match_96", "match_48", "match_24",
            ],
        ),
        qat_functions=QAT_Functions(
            forward=_default_forward_function,
            forward_qat=_default_forward_qat,
            inference_qat=_default_inference_qat,
        ),
    )
    return config

def get_qat_config__Student_MergeActions(args):
    config = QAT_Config(
        platform=args.platform,
        overrides=get_overrides(args),
        model_fp_weights=args.pretrained,
        # model_qat_weights="runs/multi-head/.experiments/20260312_171516_yolov8s-19kpts_Student_MergeActions/weights/last.pt",
        val_config=ValConfig(
            enable=True,
        ),
        export_config=ExportConfig(
            enable=True,
            input_names=["input"],
            output_names=[
                "head_96", "head_48", "head_24",
                "body_96", "body_48", "body_24",
                "action_96", "action_48", "action_24",
                "face_96", "face_48", "face_24",
            ],
        ),
        qat_functions=QAT_Functions(
            forward=_default_forward_function,
            forward_qat=_default_forward_qat,
            inference_qat=_default_inference_qat,
        ),
    )
    if args.platform.lower() in ["rknn", "torchao"]:
        config.qat_functions.forward = partial(_default_forward_function, ff_cat=True)

    if args.platform.lower() in ["sophgo", "nvidia", "rknn", "torchao"]:
        config.calibrate_config = CalibrateConfig(
            enable=True,
            batch_size=64,
            num_batch=50,
        )
    if args.platform.lower() == "sophgo":
        config.custom_kwargs = {
            "prepare_custom_config_dict": {
                "quant_dict": {
                    "chip": args.sophgo_chip,
                    "quantmode": "weight_activation",
                    "strategy": "CNN",
                },
                "extra_quantizer_dict": {
                    "exclude_node_name": [
                        "cat_12", "cat_13", "cat_14", 
                        "cat_15", "cat_16", "cat_17", 
                        "cat_18", "cat_19", "cat_20", 
                        "cat_21", "cat_22", "cat_23", 
                    ]
                }
            },
            "chip": args.sophgo_chip,
        }

    return config

def get_qat_config__Teacher(args):
    config = QAT_Config(
        platform=args.platform,
        overrides=get_overrides(args),
        model_fp_weights=args.pretrained,
        # model_qat_weights="/data4/yuanchengzhi/projects/yolo/MultiTaskDetector/runs/multi-head/qat/20260209_165633_yolov8s_25kpts_2H/weights/last.pt",
        val_config=ValConfig(
            enable=True,
        ),
        export_config=ExportConfig(
            enable=True,
            input_names=["input"],
            output_names=[
                "box_cls_kpt_angle_96", "box_cls_kpt_angle_48", "box_cls_kpt_angle_24",
            ],
        ),
        qat_functions=QAT_Functions(
            forward=_default_forward_function,
            forward_qat=_default_forward_qat,
            inference_qat=_default_inference_qat,
        ),
    )

    if args.platform.lower() in ["sophgo", "nvidia"]:
        config.calibrate_config = CalibrateConfig(
            enable=True,
            batch_size=64,
            num_batch=100,
        )
    if args.platform.lower() == "sophgo":
        config.custom_kwargs = {
            "prepare_custom_config_dict": {
                "quant_dict": {
                    "chip": args.sophgo_chip,
                    "quantmode": "weight_activation",
                    "strategy": "CNN",
                },
                "extra_quantizer_dict": {
                    "exclude_node_name": [
                        "cat_12", "cat_13", "cat_14" 
                    ]
                }
            },
            "chip": args.sophgo_chip,
        }

    return config
