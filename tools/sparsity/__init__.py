from typing import List, Optional
import torch
import autosparsity

def sparsity_model(
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    mode: int = 0, 
    verbose: int = 2, 
    whitelist: List[torch.nn.Module] = list([torch.nn.Linear, torch.nn.Conv2d]), 
    allowed_layer_names: Optional[List[str]] = None, 
    disallowed_layer_names: list[str] = list(), 
    fast: bool = False
) -> None:
    """
    Args:
        model (torch.nn.Module): 原训练模型.
        optimizer (torch.optim.Optimizer): 原优化器，默认为None.
        mode (int): 稀疏化方式，可选值为0, 1, 2, 3，默认为0. \
            0: 4:2输入方向稀疏化 (50%稀疏率). \
            1: 4:2输出方向稀疏化 (50%稀疏率). \
            2: 16:4输入输出稀疏化 (75%稀疏率). \
            3: 16:4输出输入稀疏化 (75%稀疏率).
        verbose (int): log等级，可选值为0, 1, 2, 3，默认为2. 0: Errors. 1: Errors and Warnings. 2: Errors, warnings and info.
        whitelist (list): 稀疏化支持的module列表，支持1d conv，2d conv，3d conv, linear, MultiheadAttention，默认[torch.nn.Linear, torch.nn.Conv2d].
        allowed_layer_names (list): 允许稀疏化的层名，用户配置时则只稀疏指定层，默认None.
        disallowed_layer_names (list): 不允许稀疏化的层名，用户配置时则会跳过该层，默认 [].
        fast (bool): 设为True代表使用快速方法计算mask，默认为False (默认的mask计算方法针对部分模型会失效，若稀疏化报错可尝试将该参数设为True)
    """

    # NOTE: More details can refer to https://github.com/NVIDIA/apex/blob/master/apex/contrib/sparsity/
    return autosparsity.sparsity.sparsity_model(model, optimizer, mode, verbose, whitelist, allowed_layer_names, disallowed_layer_names, fast)
