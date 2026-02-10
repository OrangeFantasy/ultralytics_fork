from types import MethodType

import torch
from torch.nn.quantized import FloatFunctional
from torch.ao import quantization

__all__ = ["apply_quantization_ops", "fuse_ops"]

def Concat_quant_forward(self, x):
    return self.op.cat(x, self.d)

def Bottleneck_quant_forward(self, x):
    return self.op.add(x , self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))

def C2f_quant_forward(self, x):
    x = self.cv1(x)
    x = self.dequant(x)
    y = list(x.split((self.c, self.c), 1)) # ONNX does not support quantized ops for split or chunk
    # y = [self.quant(t) for t in y]
    y = [self.quant0(y[0]), self.quant1(y[1])]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(self.op.cat(y, 1))
    # y = self.cv1(x)
    # y1, y2 = y.split(y.size(1) // 2, dim=1)
    # y = [y1, y2]
    # y.extend(m(y[-1]) for m in self.m)
    # return self.cv2(self.op.cat(y, 1))

def C3_quant_forward(self, x):
    x = self.op.cat([self.m(self.cv1(x)), self.cv2(x)], 1)
    return self.cv3(x)

def SPPF_quant_forward(self,x):
    x = self.cv1(x)
    y1 = self.m(x)
    y2 = self.m(y1)
    return self.cv2(self.op.cat((x, y1, y2, self.m(y2)), 1))

def MultiHead_qat_forward_head(
    self, 
    x: list[torch.Tensor],
    box_head: torch.nn.Module,
    cls_head: torch.nn.Module,
    pose_head: torch.nn.Module,
    kpts_head: torch.nn.Module,
    kpts_sigma_head: torch.nn.Module,  # Unused ?
    angle_head: torch.nn.Module,
):
    boxes, scores = [], []
    for head_idx in range(self.n_heads):
        boxes.append([
            self.dequant(box_head[head_idx][i](x[i]))
            for i in range(self.nl)
        ])
        scores.append([
            self.dequant(cls_head[head_idx][i](x[i]))
            for i in range(self.nl)
        ])

    pose = [pose_head[i](x[i]) for i in range(self.nl)]
    kpts = [self.dequant(kpts_head[i](pose[i])) for i in range(self.nl)]
    
    angles = []
    for head_idx in range(self.n_angle_heads):
        angles.append([
            self.dequant(angle_head[head_idx][i](x[i])) 
            for i in range(self.nl)
        ])

    if self.export:
        x0_s0 = torch.cat((boxes[0][0], scores[0][0], kpts[0], angles[0][0]), dim=1)
        x0_s1 = torch.cat((boxes[0][1], scores[0][1], kpts[0], angles[0][1]), dim=1)
        x0_s2 = torch.cat((boxes[0][2], scores[0][2], kpts[0], angles[0][2]), dim=1)
        x1_s0 = torch.cat((boxes[1][0], scores[1][0]), dim=1)
        x1_s1 = torch.cat((boxes[1][1], scores[1][1]), dim=1)
        x1_s2 = torch.cat((boxes[1][2], scores[1][2]), dim=1)
        return x0_s0, x0_s1, x0_s2, x1_s0, x1_s1, x1_s2

    bs = x[0].shape[0]
    for head_idx in range(self.n_heads):
        boxes[head_idx] = [
            torch.cat(boxes[i].view(bs, 4 * self.reg_max, -1), dim=-1) 
            for i in range(self.nl)
        ]
        scores = [
            torch.cat(scores[i].view(bs, self.nc_per_head[head_idx], -1), dim=-1) 
            for i in range(self.nl)
        ]

    kpts = torch.cat([
        kpts[i].view(bs, self.nk, -1)
        for i in range(self.nl)
    ], dim=2)

    for head_idx in range(self.n_heads):
        angles[head_idx] = [
            torch.cat(angles[i].view(bs, self.na, -1), dim=-1)
            for i in range(self.nl)
        ]

    return dict(
        feats=x,
        boxes=torch.cat(boxes, dim=1), 
        scores=torch.cat(scores, dim=1), 
        kpts=kpts,
        angles=torch.cat(angles, dim=1),
    )



def PoseAngle_quant_forward(self, x):
    cv2 = [self.dequant(self.cv2[i](x[i])) for i in range(self.nl)]
    cv3 = [self.dequant(self.cv3[i](x[i])) for i in range(self.nl)]
    kpt = [self.dequant(self.cv4[i](x[i])) for i in range(self.nl)]  # (bs, 17*3, h*w)       
    ang = [self.dequant(self.cv5[i](x[i])) for i in range(self.nl)]  # (bs, 3, h*w)

    x_s0 = torch.cat((cv2[0], cv3[0], kpt[0], ang[0]), 1)
    x_s1 = torch.cat((cv2[1], cv3[1], kpt[1], ang[1]), 1)
    x_s2 = torch.cat((cv2[2], cv3[2], kpt[2], ang[2]), 1)
    return x_s0, x_s1, x_s2

def _is_type(module: torch.nn.Module, type_name: str) -> bool:
    return module.__class__.__name__ == type_name

def apply_quantization_ops(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if _is_type(module, "Conv") and not _is_type(module.act, "ReLU"):
            print(f"    replace {name}.act to ReLU. {type(module.act)} maybe not supported by ONNX.")
            module.act = torch.nn.ReLU()
        # elif _is_type(module, "Concat"):
        #     module.op = FloatFunctional()
        #     module.forward = MethodType(Concat_quant_forward, module)
        # elif _is_type(module, "C2f"):
        #     module.dequant = quantization.DeQuantStub()
        #     module.quant0 = quantization.QuantStub()
        #     module.quant1 = quantization.QuantStub()
        #     module.op = FloatFunctional()
        #     module.forward = MethodType(C2f_quant_forward, module)
        # elif _is_type(module, "C3"):
        #     module.op = FloatFunctional()
        #     module.forward = MethodType(C3_quant_forward, module)
        # elif _is_type(module, "Bottleneck"):
        #     if module.add: 
        #         module.op = FloatFunctional()
        #         module.forward = MethodType(Bottleneck_quant_forward, module)
        # elif _is_type(module, "SPPF"):
        #     module.op = FloatFunctional()
        #     module.forward = MethodType(SPPF_quant_forward, module)
        # elif _is_type(module, "MultiHead"):
        #     module.op = FloatFunctional()
        #     # for i in range(len(module.cv2)):
        #     #     setattr(module, f"dequant2_{i}", quantization.DeQuantStub())
        #     # for i in range(len(module.cv3)):
        #     #     setattr(module, f"dequant3_{i}", quantization.DeQuantStub())       
        #     # if hasattr(module, "cv4"):
        #     #     for i in range(len(module.cv4)):
        #     #         setattr(module, f"dequant4_{i}", quantization.DeQuantStub())
        #     # if hasattr(module, "cv4_xy"):
        #     #     for i in range(len(module.cv4_xy)):
        #     #         setattr(module, f"dequant4_xy_{i}", quantization.DeQuantStub())
        #     # if hasattr(module, "cv4_s"):
        #     #     for i in range(len(module.cv4_s)):
        #     #         setattr(module, f"dequant4_s_{i}", quantization.DeQuantStub())
        #     # for i in range(len(module.cv5)):
        #     #     setattr(module, f"dequant5_{i}", quantization.DeQuantStub())
        #     module.forward_head = MethodType(MultiHead_qat_forward_head, module)
        # elif _is_type(module, "PoseAngle"):
        #     module.op = FloatFunctional()
        #     module.forward = MethodType(PoseAngle_quant_forward, module)

    # def Yolo_Model_forward(self, x):
    #     x = self.quant(x)
    #     return self.predict(x)

    # model.quant = quantization.QuantStub()
    # model.dequant = quantization.DeQuantStub()
    # model.model[-1].dequant = model.dequant
    # model.forward = MethodType(Yolo_Model_forward, model)
    return model

def fuse_ops(model: torch.nn.Module, inplace: bool = True) -> int:
    info = { "['conv', 'bn', 'act']": 0, "['conv', 'bn']": 0 }
    for _, module in model.named_modules():
        if _is_type(module, "Conv"):
            if _is_type(module.act, "ReLU"):
                quantization.fuse_modules_qat(module, ["conv", "bn", "act"], inplace)
                info["['conv', 'bn', 'act']"] += 1
            else:
                quantization.fuse_modules_qat(module, ["conv", "bn"], inplace)
                info["['conv', 'bn']"] += 1
        if _is_type(module, "ConvTranspose"):
            if _is_type(module.act, "ReLU"):
                quantization.fuse_modules_qat(module, ["bn", "act"], inplace)
    return info
