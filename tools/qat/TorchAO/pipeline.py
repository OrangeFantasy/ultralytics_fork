from typing import Any

import copy
import onnx
import onnxslim
import os
from tqdm import tqdm

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from ..pipeline import QAT_Pipeline

class NoQuant_Concat(torch.nn.Module):
    def forward(self, x, dim):
        return torch.cat(x, dim)

class TorchAO_Pipeline(QAT_Pipeline):
    @staticmethod
    def convert(model: torch.nn.Module):
        model = model.to("cpu").eval()
        model = convert_fx(model)
        print("==> TorchAO: convert quantization model to deploy model done")
        return model

    @staticmethod
    def is_converted_fx(model):
        for m in model.modules():
            if isinstance(m, (FakeQuantize, ObserverBase)):
                return False
        return True

    def prepare(self, model_fp: torch.nn.Module, qat_weights: str | None, weight_key: str = "model", *args, **kwargs):
        m = model_fp.model[-1]
        m.op_concat = NoQuant_Concat()

        print("==> TorchAO: prepare_qat_fx...")
        dummy_input = torch.randn([1, 3, *self.args.imgsz], device=self.device)
        qconfig_mapping = (
            QConfigMapping()
                .set_global(get_default_qconfig("qnnpack"))
                .set_module_name_regex("model.*.op_concat", None)  # Disable quantization for Concat
        )
        model_qat = prepare_qat_fx(model_fp, qconfig_mapping, example_inputs=(dummy_input,))   
        with open(self.wdir / "model_qat_graph.txt", "w") as f:
            f.write(str(model_qat.graph))
        print(f"==> TorchAO: prepare_qat_fx done, save graph to {self.wdir / 'model_qat_graph.txt'}")

        if qat_weights:
            model_qat = self.convert(model_qat)
            weights = torch.load(qat_weights, map_location="cpu", weights_only=True)
            model_qat.load_state_dict(weights[weight_key])
            print(f"==> TorchAO: load quantization model weights from {qat_weights}")

        model_fp.train()
        model_qat.train()
        return model_fp, model_qat
    
    def calibrate(self, batch_size: int, num_batch: int, *args, **kwargs):
        calibration_loader, preprocess_function = self.get_calibration_dataloader(batch_size)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(calibration_loader), total=num_batch, desc="Calibrating", ncols=120):
                imgs = preprocess_function(batch)
                self.model(imgs)

    def export(
        self, 
        input_names: list[str], 
        output_names: list[str], 
        dynamic_axes: dict[str, dict] = None,
        model_wrapper: Any = None,
        **kwargs
    ) -> None:
        if not self.is_converted_fx(self.model):
            self.model = self.convert(self.model)
        model = model_wrapper(self.model) if model_wrapper is not None else self.model

        dummy_input = torch.randn([1, 3, *self.args.imgsz], device="cpu")
        save_path = str(self.wdir / "TorchAO")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Save quantized model.
        onnx_path = os.path.join(save_path, "model_torchao_qat.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path, verbose=False, opset_version=13,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            do_constant_folding=False
        )

        onnx_model = onnx.load(onnx_path)  # load onnx model
        onnx_model = onnxslim.slim(onnx_model)
        os.remove(onnx_path)
        onnx.save(onnx_model, onnx_path)
        print(f"==> TorchAO: export quantization model to {onnx_path}")

    def get_qat_model_state_dict(self):
        # NOTE: Save model's state dict directly seem to cause error when loading. Need convert.
        model = copy.deepcopy(self.model).to("cpu").eval()
        model(torch.randn((1, 3, *self.args.imgsz), dtype=torch.float32, device="cpu"))
        model = self.convert(model)
        model(torch.randn((1, 3, *self.args.imgsz), dtype=torch.float32, device="cpu"))
        return model.state_dict()
    
    def val(self, loss_names=None, **kwargs):
        if not self.is_converted_fx(self.model):
            self.model = self.convert(self.model.to("cpu"))

        overrides = copy.deepcopy(self.overrides)
        overrides["device"] = "cpu"
        pipe = TorchAO_Pipeline("TorchAO", overrides=overrides)
        pipe.set_prepared_model(self.model_fp.to("cpu").eval(), self.model)

        return super().val(loss_names=loss_names, **kwargs)
