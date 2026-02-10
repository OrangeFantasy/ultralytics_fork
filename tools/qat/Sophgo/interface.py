from typing import Any, Callable, Dict, List
from tqdm import tqdm

import os
import argparse

import torch
from torch import fx
from torch.utils.data import DataLoader

from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_calibration, enable_quantization

from ..pipeline import QAT_Pipeline

class Sophgo_Pipeline(QAT_Pipeline):
    def prepare(self, model_fp: torch.nn.Module, qat_weights: str | None, extra_prepare_dict: dict[str, Any]) -> fx.graph_module.GraphModule:
        print("==> SOPHGO step1: prepare_by_platform ...")
        model_fp.train()
        input_shapes = [[1, 3, *self.overrides["imgsz"]]]
        # NOTE: prepare_by_platform params 'input_shape_dict' is a list, not a dict
        model_qat = prepare_by_platform(model_fp, input_shapes, extra_prepare_dict)
        enable_quantization(model_fp)
        if qat_weights is not None:
            weights = torch.load(qat_weights, map_location="cpu", weights_only=True)
            model_qat.load_state_dict(weights["model"], strict=True)
        return model_fp, model_qat

    def calibrate(
        self, num_cbatch: int = 100
    ) -> None:
        print("==> SOPHGO step2: calibrate ...")
        calibration_loader, preprocess_function = self.get_calibration_loader()
        enable_calibration(self.model)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(calibration_loader), total=num_cbatch, ncols=120):
                batch = preprocess_function(batch)
                self.model(batch)
                if batch_idx + 1 == num_cbatch:
                    break
        enable_quantization(self.model_fp)
    
    @staticmethod
    def enbale_quantization(model: torch.nn.Module) -> None:
        print("==> SOPHGO step3: enbale_quantization ...")
        enable_quantization(model)

    @staticmethod
    def export(
        model: torch.nn.Module, input_shape_dict: Dict[str, List[int]], save_path: str, model_name: str, 
        input_names: List[str], output_names: List[int], dynamic_axes: Dict[str, Dict] = None, chip: str = "BM1688"
    ) -> None:
        os.makedirs(save_path, exist_ok=True)

        model.eval()
        convert_deploy(
            model, "CNN", input_shape_dict,
            output_path=save_path, model_name=model_name, deploy=True, chip=chip, 
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
        )
        print(f"==> SOPHGO: convert_deploy to {save_path} ...")

def main(args: argparse.Namespace, overrides: Dict[str, Any]):
    training = (not args.val) and (not args.export)

    from ..pipeline import QAT_Pipeline
    pipe = QAT_Pipeline(platform="sophgo", overrides=overrides)
    model = pipe.load_floating_point_model(cfg=args.model, weights=args.weight if training else None)
    model.model[-1].quantation = True
    model.model[-1].export_with_postprocess = args.export_with_postprocess
    print( "==> Set 'head'.quantation to True.")
    print(f"==> Set 'head'.export_with_postprocess to {args.export_with_postprocess}.")

    # Configure SOPHGO
    extra_prepare_dict = {
        "quant_dict": {
            "chip": args.chip,
            "quantmode": "weight_activation",
            "strategy": "CNN",
        },
    }
    if os.environ.get("TASK") == "multi-detectors":
        extra_prepare_dict["extra_quantizer_dict"] = { }
        extra_prepare_dict["extra_quantizer_dict"]["exclude_node_name"] = [
            "cat_12", "cat_13", "cat_14", 
            "cat_15", "cat_16", "cat_17", 
            "cat_18", "cat_19", "cat_20", 
            "cat_21", "cat_22", "cat_23", 
            "cat_24", "cat_25", "cat_26",
        ]
        if args.export_with_postprocess:
            extra_prepare_dict["extra_quantizer_dict"]["exclude_node_name"] = [
                # "_tensor_constant1", "_tensor_constant1_1", "_tensor_constant1_2", 
                # "_tensor_constant1_3", "_tensor_constant1_4", "_tensor_constant1_5", 
                # "_tensor_constant2", 
                # "_tensor_constant3", "_tensor_constant3_1",
                # "_tensor_constant4", "_tensor_constant4_1", 
                # "_tensor_constant5", "_tensor_constant5_1",
                # "_tensor_constant6", "_tensor_constant6_1",

                # "view", "view_1", "view_2", "cat_12", "sigmoid",
                # "cat_13", "view_3", "cat_14", "view_4", "cat_15", "view_5", "cat_16",
                # "add_6", "truediv", "mul", "mul_1", "sub", "add_7", "add_8", "sub_1", "truediv_1", "cat_17",
                # "mul_2", "sigmoid_2", "cat_18", "cat_19",

                # "view_6", "view_7", "view_8", "cat_20",
                # "view_9", "sigmoid_2", "mul_3", "add_9", "mul_4", "cat_21", "view_10",
                # "cat_22", "view_11", "cat_23", "view_12", "cat_24", "view_13", "cat_25", 
                # "add_10", "truediv_2", "mul_5", "mul_6", "sub_2", "add_11", "add_12", "sub_3", "truediv_3", "cat_26",
                # "mul_7", "sigmoid_3", "cat_27", "cat_28",

                # "cat_29", "view_14", "cat_30", "view_15", "cat_31", "view_16", "cat_32",
                # "add_13", "truediv_4", "mul_8", "mul_9", "sub_4", "add_14", "add_15", "sub_5", "truediv_5", "cat_33",
                # "mul_10", "sigmoid_4", "cat_34",

                # "cat_35", "view_17", "cat_36", "view_18", "cat_37", "view_19", "cat_38", 
                # "add_16", "truediv_6", "mul_11", "mul_12", "sub_6", "add_17", "add_18", "sub_7", "truediv_7", "cat_39",
                # "mul_13", "sigmoid_5", "cat_40",

                # "view_20", "view_21", "view_22", "cat_41", "sigmoid_6",
                # "cat_42", "view_23", "cat_43", "view_24", "cat_44", "view_25", "cat_45",
                # "add_19", "truediv_8", "mul_14", "mul_15", "sub_8", "add_20", "add_21", "sub_9", "truediv_9", "cat_46",
                # "mul_16", "sigmoid_7", "cat_47", "cat_48",
            ]
    elif os.environ.get("TASK") == "pose-angle":
        extra_prepare_dict["extra_quantizer_dict"] = { }
        extra_prepare_dict["extra_quantizer_dict"]["exclude_node_name"] = [
            "cat_12", "cat_13", "cat_14" 
        ]
        if args.export_with_postprocess:
            extra_prepare_dict["extra_quantizer_dict"]["exclude_function_type"] = [
                torch.reshape
            ]
            extra_prepare_dict["extra_quantizer_dict"]["exclude_node_name"] = []
            #     "_tensor_constant0", "_tensor_constant1",
            #     # "view", "view_1", "view_2", 
            #     # "cat_12", 
            #     # "view_6", 
            #     "sigmoid_1", "mul", "add_6", "mul_1", 
            #     # "cat_14", "view_7",
            #     # "view_3", "view_4", "view_5", 
            #     # "cat_13", 
            #     # "cat_15", "view_8", "cat_16", "view_9", "cat_17", "view_10", "cat_18",
            #     "add_7", "truediv", "mul_2", "mul_3", "sub", "add_8", "add_9", "sub_1", "truediv_1", 
            #     # "cat_19",
            #     "mul_4", "sigmoid_2",
            #     #   "cat_20", "cat_21",

            #     "sigmoid",
            #     #   "cat_22", "cat_23", "cat_24",
            #     # "reshape", "reshape_1", "reshape_2", "reshape_3", "reshape_4", "reshape_5"
            # ]
    else:
        raise ValueError("TASK not defined.")
    print(f"==> extra_prepare_dict: \n{extra_prepare_dict}")
    
    # Get quantized model.
    if training or args.val:
        input_shapes = [[1, 3, overrides["imgsz"], overrides["imgsz"]]]
    else:
        input_shapes = [[1, 3, args.export_img_h, args.export_img_w]]
    
    model_quantized = Sophgo_Pipeline.prepare(model, extra_prepare_dict, input_shapes)
    if args.val or args.export:
        weights = torch.load(args.weight, map_location="cpu", weights_only=True)
        model_quantized.load_state_dict(weights["model"], strict=True)

    # Set model to pipeline.     
    pipe.set_original_model(model)
    pipe.set_quantized_model(model_quantized)
    if training:
        calibration_loader, prerocess_function = pipe.get_calibration_dataloader()
        Sophgo_Pipeline.calibrate(model_quantized, calibration_loader, prerocess_function, args.num_calibration)
    Sophgo_Pipeline.enbale_quantization(model_quantized)

    # Train, validate and export.
    if training:
        pipe.train()
    
    if training or args.val:
        kwargs = {
            "conf": 0.001, "iou": 0.65, "max_det": 300, 
            "save_json": True, "plots": True, "verbose": True
        }
        pipe.val(**kwargs)

    if training or args.export:
        if os.environ.get("TASK") == "multi-detectors":
            if args.export_with_postprocess:
                output_names=[
                    "head_8_box_cls_angle", "body_56_box_cls_kpt", "raise_hand_6_box_cls", "stand_up_6_box_cls", "face_8_box_cls_angle",
                ]
            else:
                output_names=[
                    "head_96", "head_48", "head_24",
                    "body_96", "body_48", "body_24",
                    "raise_hand_96", "raise_hand_48", "raise_hand_24",
                    "stand_up_96", "stand_up_48", "stand_up_24",
                    "face_96", "face_48", "face_24",
                ]
        elif os.environ.get("TASK") == "pose-angle":
            if args.export_with_postprocess:
               output_names=["63_box_cls_kpt_angle"]
            else:
                output_names=[
                    "box_cls_kpt_angle_96", "box_cls_kpt_angle_48", "box_cls_kpt_angle_24",
                ]
        else:
            raise ValueError("TASK not defined.")

        input_shape_dict = { "input": [1, 3, args.export_img_h, args.export_img_w] }
        print(f"==> Exporting model with output_names: {output_names}, input_shape: [ {args.export_img_h}, {args.export_img_w} ] ...")
        Sophgo_Pipeline.export(
            pipe.model, input_shape_dict, str(pipe.wdir / "sophgo"), "sophgo", chip=args.chip,
            input_names=["input"], output_names=output_names
        )
