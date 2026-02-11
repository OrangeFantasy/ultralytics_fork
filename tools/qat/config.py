from typing import List

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
        "project": args.project or "qat",
        "name": datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + (args.name or args.model.split("/")[-1].split(".")[0]),
    }
    kwargs.update(args.override_hyp)
    return kwargs

def get_ball_sports_qat_config(args):
    def _forward_function(self, x: List[torch.Tensor]):
        box0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
        box1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
        cls0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
        cls1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]

        pose = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpts = [self.cv4_kpts[i](pose[i]) for i in range(self.nl)]
        angles = [self.cv5[0][i](x[i]) for i in range(self.nl)]

        x0 = [self.float_cat([box0[l], cls0[l], kpts[l], angles[l]], dim=1) for l in range(self.nl)]
        x1 = [self.float_cat([box1[l], cls1[l]], dim=1) for l in range(self.nl)]
        if self.export:
            return (*x0, *x1)
        return (*x, *x0, *x1)
    
    def _forward_qat(self, preds):
        feats = preds[0:3]
        x0, x1 = (
            tuple(preds[i * self.nl : (i + 1) * self.nl])
            for i in range(1, 1 + len(preds[3:]) // self.nl)
        )

        box0, cls0, kpts, angs, box1, cls1 = map(list, zip(*[
            (
                *a.split((4 * self.reg_max, self.nc_per_head[0], self.nk, self.n_angles), 1),
                *b.split((4 * self.reg_max, self.nc_per_head[1]), 1),
            )
            for a, b in zip(x0, x1)
        ]))

        bs = feats[0].shape[0]
        box0 = torch.cat([box0[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box1 = torch.cat([box1[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        cls0 = torch.cat([cls0[i].view(bs, self.nc_per_head[0], -1) for i in range(self.nl)], dim=-1)
        cls1 = torch.cat([cls1[i].view(bs, self.nc_per_head[1], -1) for i in range(self.nl)], dim=-1)
        kpts = torch.cat([kpts[i].view(bs, self.nk, -1) for i in range(self.nl)], dim=-1)
        angs = torch.cat([angs[i].view(bs, self.n_angles, -1) for i in range(self.nl)], dim=-1)

        return dict(
            feats=list(feats),
            boxes=torch.cat([box0, box1], 1),
            scores=torch.cat([cls0, cls1], 1),
            kpts=kpts,
            angles=angs
        )

    def _inference_qat(self, preds):
        preds = self.forward_qat(self, preds)
        y = self._inference(preds)
        return (y, preds)
    
    config = QAT_Config(
        platform=args.platform or "rknn",
        overrides=get_overrides(args),
        model_fp_weights=args.pretrained or "runs/multi-head/train/20260131_165216_best/weights/best.pt",
        # model_qat_weights="runs/multi-head/.experiments/20260205_191746_yolov8s_25kpts_2H/weights/last.pt",
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
            forward=_forward_function,
            forward_qat=_forward_qat,
            inference_qat=_inference_qat,
        ),
    )
    return config

def get_student_qrcode_qat_config(args):
    def _forward_function(self, x: List[torch.Tensor]):
        box0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
        box1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
        box2 = [self.cv2[2][i](x[i]) for i in range(self.nl)]
        box3 = [self.cv2[3][i](x[i]) for i in range(self.nl)]
        box4 = [self.cv2[4][i](x[i]) for i in range(self.nl)]
        box5 = [self.cv2[5][i](x[i]) for i in range(self.nl)]

        cls0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
        cls1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]
        cls2 = [self.cv3[2][i](x[i]) for i in range(self.nl)]
        cls3 = [self.cv3[3][i](x[i]) for i in range(self.nl)]
        cls4 = [self.cv3[4][i](x[i]) for i in range(self.nl)]
        cls5 = [self.cv3[5][i](x[i]) for i in range(self.nl)]

        pose = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpts = [self.cv4_kpts[i](pose[i]) for i in range(self.nl)]

        ang0 = [self.cv5[0][i](x[i]) for i in range(self.nl)]
        ang1 = [self.cv5[1][i](x[i]) for i in range(self.nl)]

        x0 = [torch.cat((box0[i], cls0[i], ang0[i]), 1) for i in range(self.nl)]
        x1 = [torch.cat((box1[i], cls1[i], kpts[i]), 1) for i in range(self.nl)]
        x2 = [torch.cat((box2[i], cls2[i]), 1) for i in range(self.nl)]
        x3 = [torch.cat((box3[i], cls3[i], ang1[i]), 1) for i in range(self.nl)]
        x4 = [torch.cat((box4[i], cls4[i]), 1) for i in range(self.nl)]
        x5 = [torch.cat((box5[i], cls5[i]), 1) for i in range(self.nl)]
        if self.export:
            return (*x0, *x1, *x2, *x3, *x4, *x5)
        return (*x, *x0, *x1, *x2, *x3, *x4, *x5)

    def _forward_qat(self, preds):
        feats = preds[0:3]
        x0, x1, x2, x3, x4, x5 = (
            tuple(preds[i * self.nl : (i + 1) * self.nl])
            for i in range(1, 1 + len(preds[3:]) // self.nl)
        )
        
        box0, cls0, ang0, box1, cls1, kpts, box2, cls2, box3, cls3, ang1, box4, cls4, box5, cls5 = map(list, zip(*[
            (
                *x0_ls.split((4 * self.reg_max, self.nc_per_head[0], self.n_angles), 1),
                *x1_ls.split((4 * self.reg_max, self.nc_per_head[1], self.nk), 1),
                *x2_ls.split((4 * self.reg_max, self.nc_per_head[2]), 1),
                *x3_ls.split((4 * self.reg_max, self.nc_per_head[3], self.n_angles), 1),
                *x4_ls.split((4 * self.reg_max, self.nc_per_head[4]), 1),
                *x5_ls.split((4 * self.reg_max, self.nc_per_head[5]), 1),
            )
            for x0_ls, x1_ls, x2_ls, x3_ls, x4_ls, x5_ls in zip(x0, x1, x2, x3, x4, x5)
        ]))

        bs = feats[0].shape[0]
        box0 = torch.cat([box0[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box1 = torch.cat([box1[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box2 = torch.cat([box2[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box3 = torch.cat([box3[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box4 = torch.cat([box4[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box5 = torch.cat([box5[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        cls0 = torch.cat([cls0[i].view(bs, self.nc_per_head[0], -1) for i in range(self.nl)], dim=-1)
        cls1 = torch.cat([cls1[i].view(bs, self.nc_per_head[1], -1) for i in range(self.nl)], dim=-1)
        cls2 = torch.cat([cls2[i].view(bs, self.nc_per_head[2], -1) for i in range(self.nl)], dim=-1)
        cls3 = torch.cat([cls3[i].view(bs, self.nc_per_head[3], -1) for i in range(self.nl)], dim=-1)
        cls4 = torch.cat([cls4[i].view(bs, self.nc_per_head[4], -1) for i in range(self.nl)], dim=-1)
        cls5 = torch.cat([cls5[i].view(bs, self.nc_per_head[5], -1) for i in range(self.nl)], dim=-1)
        kpts = torch.cat([kpts[i].view(bs, self.nk, -1) for i in range(self.nl)], dim=-1)
        ang0 = torch.cat([ang0[i].view(bs, self.n_angles, -1) for i in range(self.nl)], dim=-1)
        ang1 = torch.cat([ang1[i].view(bs, self.n_angles, -1) for i in range(self.nl)], dim=-1)

        return dict(
            feats=list(feats),
            boxes=torch.cat([box0, box1, box2, box3, box4, box5], 1),
            scores=torch.cat([cls0, cls1, cls2, cls3, cls4, cls5], 1),
            kpts=kpts,
            angles=torch.cat([ang0, ang1], 1),
        )

    def _inference_qat(self, preds):
        preds = self.forward_qat(preds)
        y = self._inference(preds)
        return (y, preds)

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
            forward=_forward_function,
            forward_qat=_forward_qat,
            inference_qat=_inference_qat,
        ),
    )
    return config

def get_student_qrcode_qat_config_2(args):
    def _forward_function(self, x: List[torch.Tensor]):
        box0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
        box1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
        box2 = [self.cv2[2][i](x[i]) for i in range(self.nl)]
        box3 = [self.cv2[3][i](x[i]) for i in range(self.nl)]
        box4 = [self.cv2[4][i](x[i]) for i in range(self.nl)]
        box5 = [self.cv2[5][i](x[i]) for i in range(self.nl)]
        box6 = [self.cv2[6][i](x[i]) for i in range(self.nl)]

        cls0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
        cls1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]
        cls2 = [self.cv3[2][i](x[i]) for i in range(self.nl)]
        cls3 = [self.cv3[3][i](x[i]) for i in range(self.nl)]
        cls4 = [self.cv3[4][i](x[i]) for i in range(self.nl)]
        cls5 = [self.cv3[5][i](x[i]) for i in range(self.nl)]
        cls6 = [self.cv3[6][i](x[i]) for i in range(self.nl)]

        pose = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpts = [self.cv4_kpts[i](pose[i]) for i in range(self.nl)]

        ang0 = [self.cv5[0][i](x[i]) for i in range(self.nl)]
        ang1 = [self.cv5[1][i](x[i]) for i in range(self.nl)]

        x0 = [torch.cat((box0[i], cls0[i], ang0[i]), 1) for i in range(self.nl)]
        x1 = [torch.cat((box1[i], cls1[i], kpts[i]), 1) for i in range(self.nl)]
        x2 = [torch.cat((box2[i], cls2[i]), 1) for i in range(self.nl)]
        x3 = [torch.cat((box3[i], cls3[i]), 1) for i in range(self.nl)]
        x4 = [torch.cat((box4[i], cls4[i], ang1[i]), 1) for i in range(self.nl)]
        x5 = [torch.cat((box5[i], cls5[i]), 1) for i in range(self.nl)]
        x6 = [torch.cat((box6[i], cls6[i]), 1) for i in range(self.nl)]
        if self.export:
            return (*x0, *x1, *x2, *x3, *x4, *x5, *x6)
        return (*x, *x0, *x1, *x2, *x3, *x4, *x5, *x6)

    def _forward_qat(self, preds):
        feats = preds[0:3]
        x0, x1, x2, x3, x4, x5, x6 = (
            tuple(preds[i * self.nl : (i + 1) * self.nl])
            for i in range(1, 1 + len(preds[3:]) // self.nl)
        )
        
        box0, cls0, ang0, box1, cls1, kpts, box2, cls2, box3, cls3, box4, cls4, ang1, box5, cls5, box6, cls6 = map(list, zip(*[
            (
                *x0_ls.split((4 * self.reg_max, self.nc_per_head[0], self.n_angles), 1),
                *x1_ls.split((4 * self.reg_max, self.nc_per_head[1], self.nk), 1),
                *x2_ls.split((4 * self.reg_max, self.nc_per_head[2]), 1),
                *x3_ls.split((4 * self.reg_max, self.nc_per_head[3]), 1),
                *x4_ls.split((4 * self.reg_max, self.nc_per_head[4], self.n_angles), 1),
                *x5_ls.split((4 * self.reg_max, self.nc_per_head[5]), 1),
                *x6_ls.split((4 * self.reg_max, self.nc_per_head[6]), 1),
            )
            for x0_ls, x1_ls, x2_ls, x3_ls, x4_ls, x5_ls, x6_ls in zip(x0, x1, x2, x3, x4, x5, x6)
        ]))

        bs = feats[0].shape[0]
        box0 = torch.cat([box0[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box1 = torch.cat([box1[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box2 = torch.cat([box2[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box3 = torch.cat([box3[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box4 = torch.cat([box4[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box5 = torch.cat([box5[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        box6 = torch.cat([box6[i].view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        cls0 = torch.cat([cls0[i].view(bs, self.nc_per_head[0], -1) for i in range(self.nl)], dim=-1)
        cls1 = torch.cat([cls1[i].view(bs, self.nc_per_head[1], -1) for i in range(self.nl)], dim=-1)
        cls2 = torch.cat([cls2[i].view(bs, self.nc_per_head[2], -1) for i in range(self.nl)], dim=-1)
        cls3 = torch.cat([cls3[i].view(bs, self.nc_per_head[3], -1) for i in range(self.nl)], dim=-1)
        cls4 = torch.cat([cls4[i].view(bs, self.nc_per_head[4], -1) for i in range(self.nl)], dim=-1)
        cls5 = torch.cat([cls5[i].view(bs, self.nc_per_head[5], -1) for i in range(self.nl)], dim=-1)
        cls6 = torch.cat([cls6[i].view(bs, self.nc_per_head[6], -1) for i in range(self.nl)], dim=-1)
        kpts = torch.cat([kpts[i].view(bs, self.nk, -1) for i in range(self.nl)], dim=-1)
        ang0 = torch.cat([ang0[i].view(bs, self.n_angles, -1) for i in range(self.nl)], dim=-1)
        ang1 = torch.cat([ang1[i].view(bs, self.n_angles, -1) for i in range(self.nl)], dim=-1)

        return dict(
            feats=list(feats),
            boxes=torch.cat([box0, box1, box2, box3, box4, box5, box6], 1),
            scores=torch.cat([cls0, cls1, cls2, cls3, cls4, cls5, cls6], 1),
            kpts=kpts,
            angles=torch.cat([ang0, ang1], 1),
        )

    def _inference_qat(self, preds):
        preds = self.forward_qat(preds)
        y = self._inference(preds)
        return (y, preds)

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
            forward=_forward_function,
            forward_qat=_forward_qat,
            inference_qat=_inference_qat,
        ),
    )
    return config
