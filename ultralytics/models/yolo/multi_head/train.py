# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.data.build import build_yolo_dataset
from ultralytics.nn.tasks import MultiHeadModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class MultiHeadTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = { }
        overrides["task"] = "multi-head"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> MultiHeadModel:
        model = MultiHeadModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )

        try:
            import torchsummaryX
            dummay_input = torch.randn((1, 3, *self.args.imgsz), dtype=torch.float32, device=next(model.parameters()).device)
            torchsummaryX.summary(deepcopy(model).float(), dummay_input)
        except ImportError:
            print("==> torchsummaryX not found, install with `pip install torchsummaryX`")

        with open(self.wdir / "model_arch.txt", "w") as f:
            f.write(str(unwrap_model(model)))

        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.n_angles = self.data["n_angles"]
        self.model.n_extra_props = self.data["n_extra_props"]

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "pose_loss", "kobj_loss", "rle_loss", "ang_loss"
        return yolo.multi_head.MultiHeadValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)

        heads = self.data["heads"]
        if "pose-angle" in heads or ("pose" in heads and "angle" in heads):
            dataset_task = "pose-angle"
        elif "pose" in heads:
            dataset_task = "pose"
        else:
            dataset_task = "detect"

        dataset_args = copy(self.args)
        dataset_args.task = dataset_task
        return build_yolo_dataset(dataset_args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_dataset(self):
        data = super().get_dataset()
        if "kpt_shape" not in data:
            LOGGER.info(f"No `kpt_shape` in the {self.args.data}.")
        if "n_angles" not in data:
            LOGGER.info(f"No `angles` in the {self.args.data}.")
        if "n_extra_props" not in data:
            LOGGER.info(f"No `n_extra_props` in the {self.args.data}.")
        return data
