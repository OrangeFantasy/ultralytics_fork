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
from ..pipeline import QAT_Pipeline

class Nvidia_Pipeline(QAT_Pipeline):
    def initialize_env(
        self, 
        **ignore_kwargs
    ):
        from absl import logging
        logging.set_verbosity(logging.INFO)

        # torch.use_deterministic_algorithms(mode=False)
        # quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method="histogram"))
        # quant_nn.QuantConv2d.set_default_quant_desc_weight(QuantDescriptor(calib_method="histogram"))  # NOTE: should be "max"?
        quant_modules.initialize()

    def prepare(
        self, 
        model_fp: torch.nn.Module, 
        qat_weights: str | None,
        **ignore_kwargs
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        print("==> [Nvidia] apply_quantization_ops ...")
        model_qat = apply_quantization_ops(model_fp)
        if qat_weights is not None:
            weights = torch.load(qat_weights, map_location="cpu", weights_only=True)
            model_qat.load_state_dict(weights["model"], state_dict=True)
        return model_fp, model_qat

    def calibrate(
        self, 
        batch_size, 
        num_batch, 
        method: str = "percentile", 
        percentile: float = 99.99,
        **ignore_kwargs
    ):
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

        calibration_loader, preprocess_function = self.get_calibration_dataloader(batch_size)
        print("==> [Nvidia] calibrate ...")
        print("==> [Nvidia] If the message “Calibrator encountered negative values. It shouldn't happen after ReLU.” appears, "
            "it is most likely caused by `MaxCalibrator` during weight collection, and there’s no need to worry. "
        )
        with torch.no_grad():
            collect_stats(self.model, calibration_loader, num_batch)
            compute_amax(self.model, method=method, percentile=percentile)  # NOTE: method can be 'entropy', 'max' or 'percentile'

    def export(
        self, 
        input_names: list[str], 
        output_names: list[str], 
        dynamic_axes: dict[str, dict] = None,
        opset_version: int = 13,
        **ignore_kwargs
    ) -> None:
        save_path = str(self.wdir / "Nvidia")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Export the quantization node with torch Q/DQ format.
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        model_qat = deepcopy(self.model)
        model_qat.model[-1].export = True

        # Export model.
        dummy_input = torch.randn([1, 3, *self.args.imgsz], device=self.device)
        onnx_path = os.path.join(save_path, "model_nvidia_qat.onnx")
        torch.onnx.export(
            model_qat.eval(), dummy_input, onnx_path, verbose=False, opset_version=opset_version,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
        )

        # Checks and simplifies.
        onnx_model = onnx.load(onnx_path)  # load onnx model
        onnx_model = onnxslim.slim(onnx_model)
        os.remove(onnx_path)
        onnx.save(onnx_model, onnx_path)
        print(f"==> [Nvidia] convert done, model saved to {onnx_path}")
