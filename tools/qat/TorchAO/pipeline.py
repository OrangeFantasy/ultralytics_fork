import onnx
import onnxslim
import os
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import fx
from torch.ao.quantization import (
    FakeQuantize, 
    MovingAverageMinMaxObserver, 
    MovingAveragePerChannelMinMaxObserver, 
    QConfig, 
    QConfigMapping, 
)
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from ..pipeline import QAT_Pipeline

class FloatFunctional_Concat(torch.nn.Module):
    def forward(self, x, dim):
        return torch.cat(x, dim)

class TorchAO_Pipeline(QAT_Pipeline):
    @staticmethod
    def convert(model: fx.GraphModule) -> fx.GraphModule:
        model = model.to("cpu").eval()
        model = convert_fx(model)
        print("==> [TorchAO] convert quantization model to deploy model done")
        return model

    @staticmethod
    def is_converted_fx(model: fx.GraphModule) -> bool:
        for m in model.modules():
            if isinstance(m, (FakeQuantize, ObserverBase)):
                return False
        return True
    
    @staticmethod
    def get_defualt_qconfig() -> QConfig:
        # From Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2
        return QConfig(
            activation=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                reduce_range=False
            ),
            weight = FakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_affine,
                reduce_range=False
            )
        )

    def prepare(
        self, 
        model_fp: torch.nn.Module, 
        qat_weights: str | None, 
        weight_key: str = "model", 
        **ignore_kwargs
    ) -> tuple[torch.nn.Module, fx.GraphModule]:
        m = model_fp.model[-1]
        m.float_cat = FloatFunctional_Concat()

        print("==> [TorchAO] prepare_qat_fx...")
        qconfig_mapping = QConfigMapping()    
        qconfig_mapping.set_global(self.get_defualt_qconfig())
        qconfig_mapping.set_object_type(FloatFunctional_Concat, None)  # Disable quantization for Concat

        dummy_input = torch.randn([1, 3, *self.args.imgsz], device=self.device)
        model_qat = prepare_qat_fx(model_fp, qconfig_mapping, example_inputs=(dummy_input,))

        with open(self.wdir / "model_fx_graph.txt", "w") as f:
            f.write(str(model_qat.graph))
        print(f"==> [TorchAO] prepare_qat_fx done, save graph to {self.wdir / 'model_fx_graph.txt'}")

        if qat_weights:
            model_qat = self.convert(model_qat)
            weights = torch.load(qat_weights, map_location="cpu", weights_only=True)
            model_qat.load_state_dict(weights[weight_key], state_dict=True)
            print(f"==> [TorchAO] load quantization model weights from {qat_weights}")

        return model_fp, model_qat
    
    def calibrate(
        self, 
        batch_size: int, 
        num_batch: int, 
        **ignore_kwargs
    ) -> None:
        calibration_loader, preprocess_function = self.get_calibration_dataloader(batch_size)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(calibration_loader), total=num_batch, desc="Calibrating", ncols=120):
                imgs = preprocess_function(batch)
                self.model(imgs)
                if batch_idx + 1 == num_batch:
                    break

    def get_qat_model_state_dict(self) -> dict[str, torch.Tensor]:
        # NOTE: Save model's state dict directly seem to cause error when loading. Need convert.
        model = deepcopy(self.model).to("cpu").eval()
        # model(torch.randn((1, 3, *self.args.imgsz), dtype=torch.float32, device="cpu"))
        model = self.convert(model)
        # model(torch.randn((1, 3, *self.args.imgsz), dtype=torch.float32, device="cpu"))
        return model.state_dict()
    
    def val(self, loss_names=None, **kwargs):
        if not self.is_converted_fx(self.model):
            model_qat = self.convert(self.model.to("cpu"))
            self.set_prepared_model(self.model_fp, model_qat)

        self.to("cpu")
        return super().val(loss_names=loss_names, **kwargs)

    def export(
        self, 
        input_names: list[str], 
        output_names: list[str], 
        dynamic_axes: dict[str, dict] = None,
        opset_version: int = 13,
        **ignore_kwargs
    ) -> None:
        save_path = str(self.wdir / "TorchAO")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Reprepare model for export.
        model_fp = deepcopy(self.model_fp)
        model_fp.model[-1].export = True

        qat_weights = self.qat_config.model_qat_weights if self.qat_config.skip_train else self.last
        _, model_qat = self.prepare(model_fp, qat_weights, **self.qat_config.custom_kwargs)

        # Export model.
        dummy_input = torch.randn([1, 3, *self.args.imgsz], device="cpu")
        onnx_path = os.path.join(save_path, "model_torchao_qat.onnx")
        torch.onnx.export(
            model_qat.eval(), dummy_input, onnx_path, verbose=False, opset_version=opset_version,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            do_constant_folding=False
        )

        onnx_model = onnx.load(onnx_path)  # load onnx model
        onnx_model = onnxslim.slim(onnx_model)
        os.remove(onnx_path)
        onnx.save(onnx_model, onnx_path)
        print(f"==> [TorchAO] export quantization model to {onnx_path}")
