# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import itertools
from copy import copy
from pathlib import Path
from typing import Any

import json
import numpy as np
import torch

__fast_coco_eval = False
try:
    from faster_coco_eval import COCO, COCOeval_faster
    __fast_coco_eval = True
except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops, nms
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class MultiHeadValidator(DetectionValidator):    
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.heads = None
        self.n_heads = None
        self.nc_per_head = None
        self.sigma = None
        self.kpt_shape = None
        self.n_angles = None
        self.args.task = "multi-head"
        self.metrics = PoseMetrics()

    def build_dataset(self, img_path, mode = "val", batch = None):
        heads = self.data["heads"]
        if "pose-angle" in heads or ("pose" in heads and "angle" in heads):
            dataset_task = "pose-angle"
        elif "pose" in heads:
            dataset_task = "pose"
        else:
            dataset_task = "detect"     
        dataset_args = copy(self.args)
        dataset_args.task = dataset_task
        return build_yolo_dataset(dataset_args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def init_metrics(self, model):
        super().init_metrics(model)
        if hasattr(model, "yaml"):
            self.reg_max = model.yaml.get("reg_max", 1)
        else:
            self.reg_max = model.model.yaml.get("reg_max", 1)

        self.heads = self.data["heads"]
        self.n_heads = len(self.heads)
        self.nc_per_head = self.data["nc_per_head"]

        self.n_pose_heads  = len([x for x in self.heads if x in ["pose", "pose-angle"]])
        self.kpt_shape = self.data["kpt_shape"]
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]
        self.sigma = np.fromstring(os.environ.get("__global_args__oks_sigma", np.array2string(OKS_SIGMA)).strip("[]"), sep=" ")
        self.n_angle_heads = len([x for x in self.heads if x in ["angle", "pose-angle"]])
        self.n_angles = self.data["n_angles"]

        cs = [0] + list(itertools.accumulate(self.nc_per_head))
        self.class_ranges = [[cs[i], cs[i + 1] - 1] for i in range(len(self.nc_per_head))]

        self.coco_category_id = self.data["coco_category_id"]
        self.coco_annotations = self.data["coco_annotations"]
        self.coco_jdict_index = self.data["coco_jdict_index"]
        self.jdict = [[] for _ in range(self.n_heads)]

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        batch["angles"] = batch["angles"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        if isinstance(preds, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            preds = preds[0]  # select only inference output

        nreg = (self.reg_max * 4) * self.n_heads
        nks = self.nk * self.n_pose_heads
        nas = self.n_angles * self.n_angle_heads
        assert preds.shape[1] == nreg + self.nc + nks + nas, \
            f"Pred shape {preds.shape[1]} does not match expected shape {nreg + self.nc + nks + nas}"

        # Split predictions into different tasks.
        box, cls, kpt, ang = preds.split((nreg, self.nc, nks, nas), dim=1)
        box_split = list(box.chunk(self.n_heads, 1))
        cls_split = list(cls.split(self.nc_per_head, 1))
        kpt_split = list(kpt.chunk(self.n_pose_heads, 1))
        ang_split = list(ang.chunk(self.n_angle_heads, 1))

        preds = [_ for _ in range(self.n_heads)]
        for head_idx in reversed(range(self.n_heads)):
            curr_head = self.heads[head_idx]
            if curr_head == "angle":
                prediction = torch.cat((box_split.pop(), cls_split.pop(), ang_split.pop()), 1)
            elif curr_head == "pose":
                prediction = torch.cat((box_split.pop(), cls_split.pop(), kpt_split.pop()), 1)
            elif curr_head == "pose-angle":
                prediction = torch.cat((box_split.pop(), cls_split.pop(), kpt_split.pop(), ang_split.pop()), 1)
            elif curr_head == "detect":
                prediction = torch.cat((box_split.pop(), cls_split.pop()), 1)
            else:
                raise ValueError(f"Invalid head: {curr_head}")

            nms_ouput = nms.non_max_suppression(
                prediction,
                self.args.conf, 
                self.args.iou,
                nc=self.nc_per_head[head_idx], 
                multi_label=True, 
                max_det=self.args.max_det, 
                end2end=self.end2end,
                rotated=False,
            )

            if curr_head == "angle":
                nms_ouput = [
                    {
                        "bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], 
                        "angles": x[:, 6:]
                    } for x in nms_ouput
                ]
            elif curr_head == "pose":
                nms_ouput = [
                    {
                        "bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], 
                        "keypoints": x[:, 6:].view(-1, *self.kpt_shape)
                    } for x in nms_ouput
                ]
            elif curr_head == "pose-angle":
                nms_ouput = [
                    {
                        "bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], 
                        "keypoints": x[:, 6: 6 + self.nk].view(-1, *self.kpt_shape),
                        "angles": x[:, 6 + self.nk:]
                    } for x in nms_ouput
                ]
            elif curr_head == "detect":
                nms_ouput = [
                    {
                        "bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]
                    } for x in nms_ouput
                ]
            else:
                raise ValueError(f"Invalid task: {curr_head}")

            preds[head_idx] = nms_ouput
        return preds

    def update_metrics(self, preds: list[list[dict[str, torch.Tensor]]], batch: dict[str, Any]) -> None:
        for head_idx, preds_detector in enumerate(preds):
            self._update_metrics_impl(preds_detector, batch, head_idx, self.class_ranges[head_idx])

    def _update_metrics_impl(self, preds, batch, head_idx, class_range):
        for si, pred in enumerate(preds):
            self.seen += 1
            
            pbatch = self._prepare_batch(si, batch)
            pbatch = self._select_pbatch(pbatch, class_range)
            predn = self._prepare_pred(pred)
            predn["cls"] = predn["cls"] + torch.full_like(predn["cls"], fill_value=class_range[0], device=self.device)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)

            if no_pred:
                continue

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)     
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch, head_idx)
                # self.pred_to_json(predn, batch["im_file"][si], detector_idx, class_range[0])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )
    
    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        pbatch["angles"] = batch["angles"][batch["batch_idx"] == si]
        return pbatch

    def _select_pbatch(self, pbatch: dict[str, torch.Tensor], class_range: tuple[int, int]) -> dict[str, torch.Tensor]:
        class_mask = (pbatch["cls"] >= class_range[0]) & (pbatch["cls"] <= class_range[1])
        pbatch["cls"] = pbatch["cls"][class_mask]
        pbatch["bboxes"] = pbatch["bboxes"][class_mask]
        pbatch["keypoints"] = pbatch["keypoints"][class_mask]
        pbatch["angles"] = pbatch["angles"][class_mask]
        return pbatch

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predn = super()._prepare_pred(pred)
        if "angles" in pred:
            predn["angles"] = pred["angles"]
        return predn

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if "keypoints" not in preds or len(gt_cls) == 0 or len(preds["cls"]) == 0:
            tp_p = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        predn = super().scale_preds(predn, pbatch)
        if "keypoints" in predn:
            predn["kpts"] = ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            )
        return predn

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[list[dict[str, torch.Tensor]]], ni: int, max_det: int | None = None
    ) -> None:
        batch_size = len(preds[0])
        preds_concat = [
            {
                "bboxes": torch.cat([
                    preds[head_idx][i]["bboxes"] 
                    for head_idx in range(self.n_heads)
                ]),
                "conf": torch.cat([
                    preds[head_idx][i]["conf"]
                    for head_idx in range(self.n_heads)
                ]),
                "cls": torch.cat([
                    preds[head_idx][i]["cls"] + self.class_ranges[head_idx][0]
                    for head_idx in range(self.n_heads)
                ]),
            }
            for i in range(batch_size)
        ]
        super().plot_predictions(batch, preds_concat, ni, max_det)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any], head_idx: int) -> None:      
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem    
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for i, (b, s, c) in enumerate(zip(
            box.tolist(), 
            predn["conf"].tolist(), 
            predn["cls"].tolist()
        )):
            c = int(c)         
            pred = {
                "image_id": image_id,
                "file_name": path.name,
                "category_id": self.coco_category_id[c],
                "bbox": [round(x, 3) for x in b],
                "score": round(s, 5),
            }

            curr_head = self.heads[head_idx]
            if curr_head in ["angle", "pose-angle"]:
                [pitch, yaw, roll] = predn["angles"][i].tolist()
                pred["pitch"] = round((pitch - 0.5) * 180, 3)
                pred["yaw"]   = round((yaw - 0.5) * 360, 3)
                pred["roll"]  = round((roll - 0.5) * 180, 3)
            elif curr_head in ["pose", 'pose-angle']:
                kpts = predn["keypoints"][i].view(self.kpt_shape[0] * self.kpt_shape[1])
                pred["keypoints"] = kpts.tolist()
            elif curr_head == "detect":
                pass
            else:
                raise ValueError(f"Unknown task {curr_head} for class {c}")
                 
            self.jdict[self.coco_jdict_index[c]].append(pred)

    def eval_json(self, stats):
        coco_results = open(str(self.save_dir / f"evel_result.csv"), "w", encoding="utf-8")

        imgIds = [int(Path(x).stem) if Path(x).stem.isnumeric() else Path(x).stem for x in self.dataloader.dataset.im_files]  # images to eval
        for head_idx in range(self.n_heads):
            try:
                anno_json = self.data["path"] / self.coco_annotations[head_idx]  # annotations
                pred_json = self.save_dir / f"predictions_{head_idx}.json"  # predictions

                LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                if __fast_coco_eval:
                    val = COCOeval_faster(anno, pred, "bbox", print_function=LOGGER.info)
                else:
                    val = COCOeval(anno, pred, "bbox")
                val.params.imgIds = imgIds
                val.evaluate()
                val.accumulate()
                val.summarize()
                
                if __fast_coco_eval:
                    stats_key = [
                        "AP_all", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
                        "AR_all", "AR_second", "AR_third", "AR_small", "AR_medium", "AR_large",
                    ]
                    coco_stats = [val.stats_as_dict[key] for key in stats_key]
                else:
                    coco_stats = val.stats

                coco_results.write(
                    f"Head_{head_idx}, iou_type=bbox\n"
                    f"Average Precision  (AP) @[ IoU=0.50:0.95  area=   all  maxDets=100 ], {coco_stats[0]}\n"
                    f"Average Precision  (AP) @[ IoU=0.50       area=   all  maxDets=100 ], {coco_stats[1]}\n"
                    f"Average Precision  (AP) @[ IoU=0.75       area=   all  maxDets=100 ], {coco_stats[2]}\n"
                    f"Average Precision  (AP) @[ IoU=0.50:0.95  area= small  maxDets=100 ], {coco_stats[3]}\n"
                    f"Average Precision  (AP) @[ IoU=0.50:0.95  area=medium  maxDets=100 ], {coco_stats[4]}\n"
                    f"Average Precision  (AP) @[ IoU=0.50:0.95  area= large  maxDets=100 ], {coco_stats[5]}\n"
                    f"Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets=  1 ], {coco_stats[6]}\n"
                    f"Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets= 10 ], {coco_stats[7]}\n"
                    f"Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets=100 ], {coco_stats[8]}\n"
                    f"Average Recall     (AR) @[ IoU=0.50:0.95  area= small  maxDets=100 ], {coco_stats[9]}\n"
                    f"Average Recall     (AR) @[ IoU=0.50:0.95  area=medium  maxDets=100 ], {coco_stats[10]}\n"
                    f"Average Recall     (AR) @[ IoU=0.50:0.95  area= large  maxDets=100 ], {coco_stats[11]}\n"         
                )
                coco_results.flush()

                curr_head = self.heads[head_idx]
                if curr_head in ["angle", "pose-angle"]:
                    total_num, pose_matrix, left_num, pose_matrix_b, left_num_b, pose_matrix_f, left_num_f = \
                        mae.mean_absolute_error_calculate_v2(anno_json, pred_json, frontal_face=False)

                    [pitch_error, yaw_error, roll_error] = pose_matrix
                    MAE = np.mean(pose_matrix)
                    mae_result_1 = "left bbox number: %d / %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s" % (
                        left_num, total_num, round(MAE, 4), round(pitch_error, 4), round(yaw_error, 4), round(roll_error, 4))
                    mae_result_2 = "left backward bbox number: %d / %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s" % (
                        left_num_b, left_num, round(np.mean(pose_matrix_b), 4), round(pose_matrix_b[0], 4), round(pose_matrix_b[1], 4), round(pose_matrix_b[2], 4))
                    mae_result_3 = "left frontal bbox number: %d / %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s" % (
                        left_num_f, left_num, round(np.mean(pose_matrix_f), 4), round(pose_matrix_f[0], 4), round(pose_matrix_f[1], 4), round(pose_matrix_f[2], 4))
                    LOGGER.info(mae_result_1)
                    LOGGER.info(mae_result_2)
                    LOGGER.info(mae_result_3)

                    coco_results.write(
                        f"detector_{head_idx}, angle\n"
                        f"left MAE, {round(MAE, 4)}\n"
                        f"left backward MAE, {round(np.mean(pose_matrix_b), 4)}\n"
                        f"left frontal MAE, {round(np.mean(pose_matrix_f), 4)}\n"
                    )
                    coco_results.flush()

                elif curr_head in ["pose", "pose-angle"]:
                    if __fast_coco_eval:
                        val = COCOeval_faster(anno, pred, "keypoints", print_function=LOGGER.info)
                    else:
                        val = COCOeval(anno, pred, "keypoints")
                    val.params.imgIds = imgIds
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    if __fast_coco_eval:
                        stats_key = [
                            "AP_all", "AP_50", "AP_75", "AP_medium", "AP_large",
                            "AR_all", "AR_50", "AR_75", "AR_medium", "AR_large",
                        ]
                        coco_stats = [val.stats_as_dict[key] for key in stats_key]
                    else:
                        coco_stats = val.stats

                    coco_results.write(
                        f"Head_{head_idx}, iou_type=keypoints\n"
                        f"Average Precision  (AP) @[ IoU=0.50:0.95  area=   all  maxDets= 20 ], {coco_stats[0]}\n"
                        f"Average Precision  (AP) @[ IoU=0.50       area=   all  maxDets= 20 ], {coco_stats[1]}\n"
                        f"Average Precision  (AP) @[ IoU=0.75       area=   all  maxDets= 20 ], {coco_stats[2]}\n"
                        f"Average Precision  (AP) @[ IoU=0.50:0.95  area=medium  maxDets= 20 ], {coco_stats[3]}\n"
                        f"Average Precision  (AP) @[ IoU=0.50:0.95  area= large  maxDets= 20 ], {coco_stats[4]}\n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets= 20 ], {coco_stats[5]}\n"
                        f"Average Recall     (AR) @[ IoU=0.50       area=   all  maxDets= 20 ], {coco_stats[6]}\n"
                        f"Average Recall     (AR) @[ IoU=0.75       area=   all  maxDets= 20 ], {coco_stats[7]}\n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95  area=medium  maxDets= 20 ], {coco_stats[8]}\n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95  area= large  maxDets= 20 ], {coco_stats[9]}\n"
                    )
                    coco_results.flush()
                        
            except Exception as e:
                LOGGER.info(f"Error evaluating {pred_json}: {e}")
                continue

        coco_results.close()

    def save_json(self):
        for head_idx in range(self.n_heads):
            with open(str(self.save_dir / f"predictions_{head_idx}.json"), "w", encoding="utf-8") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict[head_idx], f)
