from typing import Any, Dict, List, Tuple, Optional

import os
import json
import shutil
from copy import deepcopy

import torch
from hotwheels import amct_pytorch

from ..pipeline import QAT_Pipeline

def create_quant_retrain_config(
    config_file: str, 
    model: torch.nn.Module, 
    input_data: Tuple[torch.Tensor, ...], 
    config_defination: Optional[str] = None
) -> None:
    """
    量化感知训练接口，根据图的结构找到所有可量化的层，自动生成量化配置文件，并将可量化层的量化配置信息写入配置文件。

    Args:
        config_file (str):
            待生成的量化感知训练配置文件存放路径及名称。如果存放路径下已经存在该文件，则调用该接口时会覆盖已有文件。
        model (torch.nn.Module):
            待进行量化感知训练的模型，已加载权重。
        input_data (tuple[torch.Tensor]):
            模型的输入数据。一个torch.tensor会被等价为tuple(torch.tensor)。
        config_defination (str, optional):
            基于retrain_config_pytorch.proto文件生成的简易配置文件quant.cfg，*.proto文件所在路径为：AMCT安装目录/amct_pytorch/proto/。
            *.proto文件参数解释以及生成的quant.cfg简易量化配置文件样例请参见量化感知训练简易配置文件。默认值为 None。

    Returns:
        None: 无返回值。
    """
    return amct_pytorch.create_quant_retrain_config(config_file, model, input_data, config_defination)

def create_quant_retrain_model(
    config_file: str, 
    model: torch.nn.Module, 
    record_file: str, 
    input_data: Tuple[torch.Tensor, ...]
) -> torch.nn.Module:
    """
    量化感知训练接口，将输入的待量化的图结构按照给定的量化配置文件进行量化处理，在传入的图结构中插入量化相关的算子（数据和权重的量化感知训练层以及找N的层），
    生成量化因子记录文件record_file，返回修改后可用于量化感知训练的torch.nn.Module模型。

    Args:
        config_file (str):
            用户生成的量化感知训练配置文件，用于指定模型network中量化层的配置情况。
        model (torch.nn.Module):
            待进行量化感知训练的模型，已加载权重。
        record_file (str):
            量化因子记录文件路径及名称。
        input_data (tuple[torch.Tensor]):
            模型的输入数据。一个torch.tensor会被等价为tuple(torch.tensor)。

    Returns:
        torch.nn.Module: 返回修改后可用于量化感知训练的torch.nn.Module模型。
    """
    return amct_pytorch.create_quant_retrain_model(config_file, model, record_file, input_data)

def restore_quant_retrain_model(
    config_file: str, 
    model: torch.nn.Module, 
    record_file: str, 
    input_data: Tuple[torch.Tensor, ...], 
    pth_file: str, 
    state_dict_name: Optional[str] = None
) -> torch.nn.Module:
    """
    量化感知训练接口，将输入的待量化的图结构按照给定的量化感知训练配置文件进行量化处理，
    在传入的图结构中插入量化感知训练相关的算子（数据和权重的量化感知训练层以及找N的层），生成量化因子记录文件record_file，
    加载训练过程中保存的checkpoint权重参数，返回修改后的torch.nn.Module量化感知训练模型。

    Args:
        config_file (str):
            用户生成的量化感知训练配置文件，用于指定模型network中量化层的配置情况。
        model (torch.nn.Module):
            待进行量化感知训练的原始模型，未加载权重。
        record_file (str):
            量化因子记录文件路径及名称。
        input_data (tuple[torch.Tensor]):
            模型的输入数据。一个torch.tensor会被等价为tuple(torch.tensor)。
        pth_file (str):
            训练过程中保存的权重文件。
        state_dict_name (str, optional):
            权重文件中的权重对应的键值。默认值：None

    Returns:
        torch.nn.Module: 返回修改后的torch.nn.Module量化感知训练模型。
    """
    return amct_pytorch.restore_quant_retrain_model(config_file, model, record_file, input_data, pth_file, state_dict_name)

def save_quant_retrain_model(
    config_file: str,
    model: torch.nn.Module,
    record_file: str,
    save_path: str,
    input_data: Tuple[torch.Tensor, ...],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None, 
    dynamic_axes: Optional[Dict[str, List[int]]] = None
) -> None:
    """
    量化感知训练接口，根据用户最终的重训练好的模型，插入AscendQuant、AscendDequant等算子，生成最终量化精度仿真模型以及量化部署模型。

    Args:
        config_file (str):
            用户生成的量化感知训练配置文件，用于指定模型network中量化层的配置情况。
        model (torch.nn.Module):
            已进行量化感知训练后的量化模型。
        record_file (str):
            量化因子记录文件路径及名称。
        save_path (str):
            量化模型存放路径。该路径需要包含模型名前缀，例如./quantized_model/*model。
        input_data (tuple[torch.Tensor]):
            模型的输入数据。一个torch.tensor会被等价为tuple(torch.tensor)。
        input_names (list[str], optional):
            模型的输入的名称，用于保存的量化onnx模型中显示。默认值：None
        output_names (list[str], optional):
            模型的输出的名称，用于保存的量化onnx模型中显示。默认值：None
        dynamic_axes (dict, optional):
            对模型输入输出动态轴的指定，例如对于输入inputs（NCHW），N、H、W为不确定大小，输出outputs（NL），N为不确定大小，则指定形式为：
            {"inputs": [0,2,3], "outputs": [0]}，其中0,2,3分别表示N，H，W所在位置的索引。
    
    Returns:
        None: 无返回值。
    """
    return amct_pytorch.save_quant_retrain_model(config_file, model, record_file, save_path,  input_data, input_names, output_names, dynamic_axes)

class Ascend_Pipeline(QAT_Pipeline):
    _default_config_file: str = os.path.join("runs/.amct_cache", "config.json")
    _defualt_record_file: str = os.path.join("runs/.amct_cache", "record.txt")

    def initialize_env(self, *args, **kwargs):
        if not os.path.exists("runs/.amct_cache"):
            os.makedirs("runs/.amct_cache", mode=777)

    def prepare(
        self, 
        model_fp: torch.nn.Module, 
        qat_weights: str | None,
        only_init_config: bool = False, 
        state_dict_key: str = "model",
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        for m in model_fp.modules():
            if isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6)):
                m.inplace = False  # NOTE: inplace=True will cause error.

        dummy_input = (torch.randn((1, 3, *self.args.imgsz), device=self.device),)
        config_file = self._default_config_file
        record_file = self._defualt_record_file

        if only_init_config or not os.path.exists(config_file):
            print("==> AMCT: create_quant_retrain_config ...")
            model_fp.train()
            create_quant_retrain_config(config_file, model_fp, dummy_input, config_defination=None)
            print(f"==> Retrain config has been saved to {config_file}. Make sure and then restart the training.")
            exit()

        if qat_weights is not None:
            print(f"==> AMCT: restore_quant_retrain_model from {qat_weights} ...")
            model_qat = restore_quant_retrain_model(config_file, model_fp, record_file, dummy_input, qat_weights, state_dict_key)
        else:   
            print("==> AMCT: create_quant_retrain_model ...")
            model_qat = create_quant_retrain_model(config_file, model_fp, record_file, dummy_input)
        return model_fp, model_qat

    def export(
        self,
        input_names: List[str] = None, 
        output_names: List[str] = None, 
        dynamic_axes: Dict[str, Dict] = None,
        **unused_kwargs
    ) -> None:
        save_path = str(self.wdir / "amct")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Save config and record files.
        shutil.copy2(self._default_config_file, f"{save_path}/config.json")
        shutil.copy2(self._defualt_record_file, f"{save_path}/record.txt")

        record_update_path = self._defualt_record_file.rsplit(".", 1)[0] + "_update.txt"
        if os.path.exists(record_update_path):
            shutil.copy2(record_update_path, f"{save_path}/record_update.txt")

        # Reprepare model for export.
        model_fp = deepcopy(self.model_fp)
        model_fp.model[-1].export = True

        qat_weights = self.config.model_qat_weights if self.config.skip_train else self.last_state_dict
        _, model_qat = self.prepare(model_fp, qat_weights, **self.config.custom_kwargs)

        # Export model.
        dummy_input = (torch.randn((1, 3, *self.args.imgsz), device=self.device),)
        save_quant_retrain_model(
            self._default_config_file, model_qat, self._defualt_record_file, f"{save_path}/amct", dummy_input, input_names, output_names, dynamic_axes
        )
        print(f"==> AMCT: save_quant_retrain_model to {save_path}")
