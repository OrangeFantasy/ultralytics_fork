from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING, Type
from pathlib import Path
from dataclasses import dataclass, field

import importlib
import io
import os
from copy import copy, deepcopy
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import DataLoader

from ultralytics.models import yolo
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import __version__, DEFAULT_CFG, DEFAULT_CFG_DICT, LOCAL_RANK, LOGGER, ops
from ultralytics.utils.torch_utils import unwrap_model

def C2f_forward(self, x: torch.Tensor) -> torch.Tensor:
    # NOTE: Default quantization. For SOPHGO and AMCT.
    # y = list(self.cv1(x).chunk(2, 1))
    y = self.cv1(x)
    y1, y2 = y.split(y.size(1) // 2, dim=1)
    y = [y1, y2]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
C2f.forward = C2f_forward


from ultralytics.nn.tasks import MultiHeadModel
from ultralytics.models.yolo.multi_head import MultiHeadTrainer, MultiHeadValidator, MultiHeadPredictor
_QAT_Model, _QAT_Trainer, _QAT_Validator, _QAT_Predictor = (
    MultiHeadModel, MultiHeadTrainer, MultiHeadValidator, MultiHeadPredictor
)

class QAT_Model(_QAT_Model):
    def forward(self, x):
        return self.predict(x)

    def loss(self, preds, batch):
        raise NotImplementedError("loss() is not should be called in quantization mode.")
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        return self.criterion(preds, batch)

class QAT_Validator(_QAT_Validator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_qat = None

    def postprocess(self, preds):
        preds = self.inference_qat(preds)
        return super().postprocess(preds)


class QAT_Pipeline(_QAT_Trainer):
    registried_platform: Dict[str, str] = { }

    @classmethod
    def register(cls, name, path):
        cls.registried_platform[name] = path

    @classmethod
    def build(cls, name, *args, **kwargs) -> "QAT_Pipeline":
        path = cls.registried_platform[name]
        package = __package__ if path.startswith(".") else None
        module_path, cls_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path, package=package)
        return getattr(module, cls_name)(*args, **kwargs)

    def __init__(self, platform, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, qat_config: 'QAT_Config' = None):
        super().__init__(cfg, overrides, _callbacks)
        self.overrides = overrides
        self.qat_config = qat_config
        self.enable_ema = False

        self.platform = platform
        LOGGER.info(f"==> Build quantization pipeline for {platform} platform.")

        self.model_fp: Optional[torch.nn.Module] = None
        self.model: Optional[torch.nn.Module] = None

        # Aligns outputs from the quantized model's forward (QAT) with the format of the floating-point model's outputs (no QAT).
        # Call `after` the quantized model's forward step.
        self.forward_qat = None

        # Aligns outputs from the quantized model's forward (QAT) with the format of the original post-processing.
        # Call `before` the original post-processing step.
        self.inference_qat = None

    def load_floating_point_model(self, weights=None, verbose=True):
        kwargs = {
            "cfg": self.overrides["model"], "nc": self.data["nc"], "ch": self.data["channels"], "verbose": verbose
        }
        if "kpt_shape" in self.data:
            kwargs["data_kpt_shape"] = self.data["kpt_shape"]
        model = QAT_Model(**kwargs)
        model.to(self.device)
        setattr(model, "args", self.args)

        if weights:  # Except weight is a state dict or a .pt file
            try:
                if isinstance(weights, (str, Path)):
                    weights, _ = load_checkpoint(weights, device=self.device)  # Load from .pt file
                model.load(weights)
            except:
                LOGGER.info("==> Load weights failed. Try to load weights as a state dict.")
                weights = torch.load(weights, map_location="cpu", weights_only=True)["state_dict"]
                model.load_state_dict(weights)
        return model

    def set_prepared_model(self, model_fp, model_qat):
        self.model_fp = model_fp
        self.model = model_qat.to(self.device)

        for k in ["args", "stride", "names", "yaml", "end2end"]: # for dataloader
            setattr(self.model, k, getattr(model_fp, k))

        head_module = model_fp.model[-1]
        setattr(self.model, "nc", head_module.nc)
        self.forward_qat = partial(head_module.forward_qat, head_module)
        self.inference_qat = partial(head_module.inference_qat, head_module)

        with open(self.wdir / "model_fp.txt", "w") as f1, open(self.wdir / "model_qat.txt", "w") as f2:
            f1.write(str(model_fp))
            f2.write(str(model_qat))

    def get_calibration_dataloader(self, batch_size: int | None) -> tuple[DataLoader, Callable[[Any], torch.Tensor]]:
        # NOTE: batch_size = 64
        loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size or self.batch_size, rank=LOCAL_RANK, mode="train"
        )
        preprocess_function = lambda batch: self.preprocess_batch(batch)["img"]
        return loader, preprocess_function

    def to(self, device: str | torch.device):
        self.args.device = str(device)
        self.device = device    

        self.model_fp.to(device)
        self.model.to(device)

    def train(self):
        super().train()

    def setup_model(self):
        pass

    def _setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.args.lr0 * self.args.lrf
        )

    def _forward(self, batch):
        # Forward
        preds = self.model.forward(batch["img"].clone())
        preds = self.forward_qat(preds)

        # Loss
        if getattr(self, "criterion", None) is None:
            self.criterion = self.model_fp.init_criterion()
        return self.criterion(preds, batch)

    def get_qat_model_state_dict(self) -> Dict[str, Any]:
        model_qat = deepcopy(self.model).to("cpu").eval()
        return model_qat.state_dict()

    def save_model(self):
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": self.get_qat_model_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_args": vars(self.args),
                "date": datetime.now().isoformat(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        # if self.best_fitness == self.fitness:
        #     self.best.write_bytes(serialized_ckpt)  # save best.pt
        # if self.epoch <= 10 or ((self.save_period > 0) and (self.epoch % self.save_period == 0)):
        (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_validator(self, loss_names=None, args=None):
        self.loss_names = loss_names or ("box_loss", "cls_loss", "dfl_loss", "pose_loss", "kobj_loss", "rle_loss", "ang_loss")
        validator = QAT_Validator(
            dataloader=getattr(self, "test_loader", None),
            save_dir=self.save_dir, 
            args=args or copy(self.args), 
            _callbacks=self.callbacks
        )
        validator.inference_qat = self.inference_qat
        return validator

    def val(self, loss_names=None, **kwargs):
        self.model.eval()

        args = copy(self.args)
        for k, v in kwargs.items():
            assert hasattr(args, k), f"args has no attribute {k}"
            setattr(args, k, v)

        validator = self.get_validator(loss_names, args)
        validator(model=self.model)
        return validator.metrics.mean_results()

    def initialize_env(self, *args, **kwargs):
        pass

    def prepare(self, model_fp: torch.nn.Module, qat_weights: str | None, *args, **kwargs):
        raise NotImplementedError

    def calibrate(self, batch_size, num_batch, *args, **kwargs):
        raise NotImplementedError

    def export(self, input_names: list[str] = None, output_names: list[str] = None, dynamic_axes: dict[str, dict] = None, *args, **kwargs):
        raise NotImplementedError

QAT_Pipeline.register("ascend", ".Ascend.pipeline.Ascend_Pipeline")
QAT_Pipeline.register("nvidia", ".Nvidia.pipeline.Nvidia_Pipeline")
QAT_Pipeline.register("rknn", ".TorchAO.pipeline.TorchAO_Pipeline")
QAT_Pipeline.register("torchao", ".TorchAO.pipeline.TorchAO_Pipeline")
QAT_Pipeline.register("sophgo", ".Sophgo.pipeline.Sophgo_Pipeline")

@dataclass
class CalibrateConfig:
    enable: bool = False
    batch_size: int = 64
    num_batch: int = 100

@dataclass
class ValConfig:
    enable: bool = False
    conf: float = 0.001
    iou: float = 0.65
    save_json: bool = False
    half: bool = False
    fuse: bool = False
    plots: bool = True
    verbose: bool = True

@dataclass
class ExportConfig:
    enable: bool = False
    input_names: list[str] | None = None
    output_names: list[str] | None = None
    dynamic_axes: dict[str, dict] | None = None
    opset_version: int = 13
    verbose: bool = True

@dataclass
class QAT_Functions:
    forward: Callable | None = None
    forward_qat: Callable | None = None
    inference_qat: Callable | None = None

@dataclass
class QAT_Config:
    platform: str
    overrides: dict[str, Any] = field(default_factory=dict)
    model_fp_weights: str | None = None
    model_qat_weights: str | None = None
    skip_train: bool = False
    calibrate_config: CalibrateConfig = field(default_factory=CalibrateConfig)
    val_config: ValConfig = field(default_factory=ValConfig)
    export_config: ExportConfig = field(default_factory=ExportConfig)
    callbacks: dict[str, Any] = field(default_factory=dict)
    qat_functions: QAT_Functions = field(default_factory=QAT_Functions)
    custom_kwargs: dict[str, Any] = field(default_factory=dict)

def run_qat(config: QAT_Config):
    platform_name = config.platform.lower()
    assert platform_name in QAT_Pipeline.registried_platform.keys(), f"[Error] platform {config.platform} is not supported"
    pipe = QAT_Pipeline.build(platform_name, platform=config.platform, overrides=config.overrides, qat_config=config)

    for event, callback in config.callbacks.items():
        pipe.add_callback(pipe, event, callback)

    pipe.initialize_env(**config.custom_kwargs)

    model_fp = pipe.load_floating_point_model(weights=config.model_fp_weights)
    print(f"==> [QAT] Build floating point model with weights: {config.model_fp_weights}")

    m = model_fp.model[-1]
    if config.qat_functions.forward is not None:
        m.forward_function = config.qat_functions.forward
    if config.qat_functions.forward_qat is not None:
        m.forward_qat = config.qat_functions.forward_qat
    if config.qat_functions.inference_qat is not None:
        m.inference_qat = config.qat_functions.inference_qat

    if config.model_qat_weights is not None:
        print(f"==> [QAT] qat_weights has been provided, skip training")
        config.skip_train = True

    model_fp, model_qat = pipe.prepare(model_fp, qat_weights=config.model_qat_weights, **config.custom_kwargs)
    pipe.set_prepared_model(model_fp, model_qat)

    if not config.skip_train:
        if config.calibrate_config.enable:
            calibrate_params = config.calibrate_config.__dict__
            calibrate_params.pop("enable")
            pipe.calibrate(**calibrate_params, **config.custom_kwargs)
        pipe.train()

    if config.val_config.enable:
        val_params = config.val_config.__dict__
        val_params.pop("enable")
        pipe.val(**val_params, **config.custom_kwargs)

    if config.export_config.enable:
        export_params = config.export_config.__dict__
        export_params.pop("enable")
        pipe.export(**export_params, **config.custom_kwargs)
