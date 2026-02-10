from types import MethodType

import torch
from torch.nn import functional as F
from torch.ao.quantization.fuse_modules import fuse_modules

from pytorch_quantization.nn.modules import _utils
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn

def Concat_quant_forward(self, x):
    if hasattr(self, "op_concat"):
        return self.op_concat(x, self.d)
    return torch.cat(x, self.d)

def Upsample_quant_forward(self, x):
    if hasattr(self, "op_upsample"):
        return self.op_upsample(x)
    return F.interpolate(x, self.size, self.scale_factor, self.mode)

def C2f_quant_forward(self, x):
    if hasattr(self, "op_c2f_chunk"):
        y = list(self.op_c2f_chunk(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))

def Bottleneck_quant_forward(self, x):
    if hasattr(self, "op_add"):
        return self.op_add(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class QuantAdd(torch.nn.Module, _utils.QuantMixin):
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y
    
class QuantC2fChunk(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c

    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)
    
class QuantConcat(torch.nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim) 

class QuantUpsample(torch.nn.Module): 
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        
    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)

def _is_type(module: torch.nn.Module, type_name: str) -> bool:
    return module.__class__.__name__ == type_name

def apply_quantization_ops(model: torch.nn.Module):
    for name, module in model.named_modules():
        if _is_type(module, "Concat"):
            if not hasattr(module, "op_concat"):
                module.op_concat = QuantConcat(module.d)
            module.forward = MethodType(Concat_quant_forward, module)
        elif _is_type(module, "Upsample"):
            if not hasattr(module, "op_upsample"):
                module.op_upsample = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.forward = MethodType(Upsample_quant_forward, module)
        elif _is_type(module, "C2f"):
            if not hasattr(module, "op_c2f_chunk"):
                module.op_c2f_chunk = QuantC2fChunk(module.c)
            module.forward = MethodType(C2f_quant_forward, module)
        elif _is_type(module, "Bottleneck"):
            if module.add:
                if not hasattr(module, "op_add"):
                    module.op_add = QuantAdd(module.add)
                module.forward = MethodType(Bottleneck_quant_forward, module)
    return model
