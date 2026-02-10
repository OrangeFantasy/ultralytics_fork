from typing import Any, Callable, Dict, List

import argparse
import os
from tqdm import tqdm
from copy import deepcopy

import onnx
import onnxslim
import torch
from torch.utils.data import DataLoader
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

from .ops import apply_quantization_ops
from ..pipeline import QAT_Pipeline, register_platform

class Nvidia_Pipeline(QAT_Pipeline):
    @staticmethod
    def initialize_env(*args, **kwargs):
        from absl import logging
        logging.set_verbosity(logging.INFO)

        # torch.use_deterministic_algorithms(mode=False)
        # quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method="histogram"))
        # quant_nn.QuantConv2d.set_default_quant_desc_weight(QuantDescriptor(calib_method="histogram"))  # NOTE: should be "max"?
        quant_modules.initialize()
        print("==> Native: apply_quantization_ops ...")

    def prepare(self, model_fp: torch.nn.Module, qat_weights: str | None):       
        print("==> Native: apply_quantization_ops ...")
        model_qat = apply_quantization_ops(model_fp)
        if qat_weights is not None:
            weights = torch.load(qat_weights, map_location="cpu", weights_only=True)
            model_qat.load_state_dict(weights["model"])
        return model_fp, model_qat

    def calibrate(self, batch_size, num_batch, method: str = "percentile", percentile: float = 99.99):
        def collect_stats(model: torch.nn.Module, data_loader: DataLoader, num_batch: int) -> None:
            """Feed data to the network and collect statistics"""

            # Enable calibrators
            model.eval()
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()

            # Feed data to the network for collecting stats
            for batch_idx, batch in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats", ncols=120):
                imgs = preprocess_function(batch)
                # NOTE: If the message “Calibrator encountered negative values. It shouldn't happen after ReLU.” appears, 
                #       it is most likely caused by `MaxCalibrator` during weight collection, and there’s no need to worry.
                model(imgs)
                if batch_idx >= num_batch:
                    break

            # Disable calibrators
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        module.disable_calib()
                    else:
                        module.enable()

        def compute_amax(model: torch.nn.Module, **kwargs) -> None:
            device = next(model.parameters()).device
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        if isinstance(module._calibrator, calib.MaxCalibrator):
                            # NOTE: strict default is True, strict=False avoid Exception when some quantizer are never used?
                            module.load_calib_amax()
                        else:
                            module.load_calib_amax(**kwargs)
                        module._amax = module._amax.to(device)

        print(f"==> Nvidia: calibrate with {num_batch} batches ...")
        calibration_loader, preprocess_function = self.get_calibration_dataloader(batch_size)
        with torch.no_grad():
            collect_stats(self.model, calibration_loader, num_batch)
            compute_amax(self.model, method=method, percentile=percentile)  # NOTE: method can be 'entropy', 'max' or 'percentile'

    def export(
        self, 
        input_names: list[str], 
        output_names: list[str], 
        dynamic_axes: dict[str, dict] = None
    ) -> None:
        # Export the quantization node with torch Q/DQ format.
        prev_use_fb_fake_quant = quant_nn.TensorQuantizer.use_fb_fake_quant
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        dummy_input = torch.randn([1, 3, *self.args.imgsz], device=self.device)
        save_path = str(self.wdir / "nvidia")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Save quantized model.
        onnx_qat_path = os.path.join(save_path, "model_nvidia_qat.onnx")
        torch.onnx.export(
            self.model, dummy_input, onnx_qat_path, verbose=False, opset_version=14,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
        )

        # Checks and simplifies.
        onnx_qat = onnx.load(onnx_qat_path)  # load onnx model
        onnx_qat = onnxslim.slim(onnx_qat)
        os.remove(onnx_qat_path)
        onnx.save(onnx_qat, onnx_qat_path)
        print(f"==> Nvidia: convert done, model saved to {onnx_qat_path}")

        quant_nn.TensorQuantizer.use_fb_fake_quant = prev_use_fb_fake_quant

def export_and_quantation_balls_2H(self, x):
    cv2_0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
    cv3_0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
    ang_0 = [self.cv5[0][i](x[i]) for i in range(self.nl)]
    kpt_0 = [self.cv4[0][i](x[i]) for i in range(self.nl)]

    cv2_1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
    cv3_1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]
    
    x0 = [torch.cat((cv2_0[i], cv3_0[i], kpt_0[i], ang_0[i], ), 1) for i in range(self.nl)]
    x1 = [torch.cat((cv2_1[i], cv3_1[i]), 1) for i in range(self.nl)]

    return (*x0, *x1)

def main(args: argparse.Namespace, overrides: Dict[str, Any]):
    training = (not args.val) and (not args.export)

    # Initialize pipeline.
    Nvidia_Pipeline.initialize()
    pipe = QAT_Pipeline(platform="nvidia", overrides=overrides)
    model = pipe.load_floating_point_model(cfg=args.model, weights=args.weight if training else None)
    model.model[-1].quantation = True
    model.model[-1].quantation_function = export_and_quantation_balls_2H
    print("==> Set 'head'.quantation to True.")

    # Create quantized model.
    model_quantized = Nvidia_Pipeline.prepare(deepcopy(model))
    if args.val or args.export:
        weights = torch.load(args.weight, map_location="cpu", weights_only=True)
        model_quantized.load_state_dict(weights["model"])

    # Set model to pipeline.
    pipe.set_original_model(model)
    pipe.set_quantized_model(model_quantized)
    if training:
        calibration_loader, prerocess_function = pipe.get_calibration_dataloader()
        Nvidia_Pipeline.calibrate(model_quantized, calibration_loader, prerocess_function, args.num_calibration)
            
    # Train, validate and export.
    if training:
        pipe.train()

    if training or args.val:
        kwargs = {
            "conf": 0.001, "iou": 0.65, "max_det": 300, 
            "save_json": False, "plots": True, "verbose": True
        }
        pipe.val(**kwargs)

    if training or args.export:
        if args.export_with_postprocess:
            pipe.model.model[-1].export_with_postprocess = True

        input_names = ["input"]
        if os.environ.get("TASK") == "multi-detectors":
            if args.export_with_postprocess:
                output_names=[
                    "head_8_box_cls_angle", "body_56_box_cls_kpt", "raise_hand_6_box_cls", "stand_up_6_box_cls", "face_8_box_cls_angle",
                ]
                dynamic_axes={
                    "input": { 0: "batch" },
                    "head_8_box_cls_angle": { 0: "batch" },
                    "body_56_box_cls_kpt": { 0: "batch" },
                    "raise_hand_6_box_cls": { 0: "batch" },
                    "stand_up_6_box_cls": { 0: "batch" },
                    "face_8_box_cls_angle": { 0: "batch" },
                }
            else:
                output_names=[
                    "head_96", "head_48", "head_24",
                    "body_96", "body_48", "body_24",
                    "raise_hand_96", "raise_hand_48", "raise_hand_24",
                    "stand_up_96", "stand_up_48", "stand_up_24",
                    "face_96", "face_48", "face_24",
                ]
                dynamic_axes = {
                    "input": { 0: "batch" },
                    "head_96": { 0: "batch" }, "head_48": { 0: "batch" }, "head_24": { 0: "batch" },
                    "body_96": { 0: "batch" }, "body_48": { 0: "batch" }, "body_24": { 0: "batch" },
                    "raise_hand_96": { 0: "batch" }, "raise_hand_48": { 0: "batch" }, "raise_hand_24": { 0: "batch" },
                    "stand_up_96": { 0: "batch" }, "stand_up_48": { 0: "batch" }, "stand_up_24": { 0: "batch" },
                    "face_96": { 0: "batch" }, "face_48": { 0: "batch" }, "face_24": { 0: "batch" },
                }
                output_names=[
                    "head_body_96", "head_body_48", "head_body_24",
                    "ball_96", "ball_48", "ball_24",
                ]
                dynamic_axes = None          

        elif os.environ.get("TASK") == "pose-angle":
            if args.export_with_postprocess:
                output_names=[
                    "63_box_cls_kpt_angle"
                ]
                dynamic_axes={
                    "input": { 0: "batch" },
                    "63_box_cls_kpt_angle": { 0: "batch" },
                }
            else:
                output_names=[
                    "box_cls_kpt_angle_96", "box_cls_kpt_angle_48", "box_cls_kpt_angle_24",
                ]
                dynamic_axes = {
                    "input": { 0: "batch" },
                    "box_cls_kpt_angle_96": { 0: "batch" }, "box_cls_kpt_angle_48": { 0: "batch" }, "box_cls_kpt_angle_24": { 0: "batch" },
                }
        else:
            raise ValueError("TASK not defined.")

        pipe.to("cpu")
        dummy_input = torch.randn([1, 3, args.export_img_h, args.export_img_w], device=pipe.device) 

        print(f"==> Exporting model with output_names: {output_names}, input_shape: [ {args.export_img_h}, {args.export_img_w} ] ...")
        Nvidia_Pipeline.export(
            pipe.model, dummy_input, str(pipe.wdir / "nvidia"),
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
        )
