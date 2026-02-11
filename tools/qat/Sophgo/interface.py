from typing import Any

import os
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import fx

from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.utils.state import enable_calibration, enable_quantization

from ..pipeline import QAT_Pipeline

class Sophgo_Pipeline(QAT_Pipeline):
    def prepare(
        self, 
        model_fp: torch.nn.Module, 
        qat_weights: str | None, 
        state_dict_key: str = "model",
        prepare_custom_config_dict: dict[str, Any] = { },
        **ignore_kwargs
    ) -> tuple[torch.nn.Module, fx.GraphModule]:
        print("==> [SOPHGO] prepare_by_platform ...")
        model_fp.train()

        # NOTE: prepare_by_platform params 'input_shape_dict' is a list, not a dict
        input_shapes = [[1, 3, *self.overrides["imgsz"]]]
        model_qat = prepare_by_platform(model_fp, input_shapes, prepare_custom_config_dict)

        if qat_weights is not None:
            weights = torch.load(qat_weights, map_location="cpu", weights_only=True)
            model_qat.load_state_dict(weights[state_dict_key], strict=True)

        enable_quantization(model_qat)
        return model_fp, model_qat

    def calibrate(
        self, 
        batch_size: int, 
        num_batch: int, 
        **ignore_kwargs
    ) -> None:
        print("==> [SOPHGO] calibrate ...")
        enable_calibration(self.model)
        calibration_loader, preprocess_function = self.get_calibration_dataloader(batch_size)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(calibration_loader), total=num_batch, ncols=120):
                batch = preprocess_function(batch)
                self.model(batch)
                if batch_idx + 1 == num_batch:
                    break
        enable_quantization(self.model)

    def export(
        self,
        input_names: list[str], 
        output_names: list[str], 
        dynamic_axes: dict[str, dict] = None,
        chip: str = "BM1688",
        net_type: str = "CNN",
        **ignore_kwargs
    ) -> None:
        save_path = str(self.wdir / "Sophgo")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Reprepare model for export.
        model_fp = deepcopy(self.model_fp)
        model_fp.model[-1].export = True

        qat_weights = self.qat_config.model_qat_weights if self.qat_config.skip_train else self.last
        _, model_qat = self.prepare(model_fp, qat_weights, **self.qat_config.custom_kwargs)

        # Export model.
        input_shape_dict = { "input": [1, 3, *self.args.imgsz] }
        convert_deploy(
            model_qat, net_type, input_shape_dict,
            output_path=save_path, model_name="sophgo", deploy=True, chip=chip, 
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
        )
        print(f"==> [SOPHGO]: convert_deploy to {save_path} ...")
