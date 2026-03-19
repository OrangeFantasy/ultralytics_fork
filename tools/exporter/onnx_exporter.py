from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass

import onnx
import onnxslim
import os
import torch

@dataclass
class ExportParams:
    model_path: str
    onnx_path: str
    task: str
    input_shape: Tuple
    input_names: List
    output_names: List[str]

    opset_version: int = 13
    dynamic_axes: Dict | None = None
    export_with_postprocess: bool = False
    export_function: Callable[[torch.nn.Module, List[torch.Tensor]], Tuple[torch.Tensor, ...]] | None = None

def export_student(self, x):
    cv2_0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
    cv3_0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
    ang_0 = [self.cv5[0][i](x[i]) for i in range(self.nl)]  # (bs, 3, h*w)

    cv2_1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
    cv3_1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]
    kpt_0 = [self.cv4[i](x[i]) for i in range(self.nl)]  # (bs, 17*3, h*w)
    kpt_0 = [self.cv4_kpts[i](kpt_0[i]) for i in range(self.nl)]  # (bs, 17*3, h*w)

    cv2_2 = [self.cv2[2][i](x[i]) for i in range(self.nl)]
    cv2_3 = [self.cv2[3][i](x[i]) for i in range(self.nl)]

    cv3_2 = [self.cv3[2][i](x[i]) for i in range(self.nl)]
    cv3_3 = [self.cv3[3][i](x[i]) for i in range(self.nl)]
    ang_1 = [self.cv5[1][i](x[i]) for i in range(self.nl)]  # (bs, 3, h*w)

    x0 = [torch.cat((cv2_0[i], cv3_0[i], ang_0[i]), 1) for i in range(self.nl)]
    x1 = [torch.cat((cv2_1[i], cv3_1[i], kpt_0[i]), 1) for i in range(self.nl)]
    x2 = [torch.cat((cv2_2[i], cv3_2[i]), 1) for i in range(self.nl)]
    x3 = [torch.cat((cv2_3[i], cv3_3[i], ang_1[i]), 1) for i in range(self.nl)]

    return (*x0, *x1, *x2, *x3)

student_config = ExportParams(
    model_path="runs/student/student_448x768_20250902/weights/best.pt",
    onnx_path="runs/_export/fp32_classroom_head_face/student_448x768_20250902.fp32.onnx",
    task="multi-detectors",
    input_shape=(1, 3, 448, 768),
    input_names=["input"],
    output_names=[
        "head_96", "head_48", "head_24",
        "body_96", "body_48", "body_24",
        "raise_hand_96", "raise_hand_48", "raise_hand_24",
        "stand_up_96", "stand_up_48", "stand_up_24",
        "face_96", "face_48", "face_24",
    ],
    export_function=export_student,
)

def export_teacher(self, x):
    cv2 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
    cv3 = [self.cv3[0][i](x[i]) for i in range(self.nl)]

    pose = [self.cv4[i](x[i]) for i in range(self.nl)]
    kpts = [self.cv4_kpts[i](pose[i]) for i in range(self.nl)]
    angs = [self.cv5[0][i](x[i]) for i in range(self.nl)]

    x0 = [torch.cat((cv2[i], cv3[i], kpts[i], angs[i]), 1) for i in range(self.nl)]
    return (*x0, )

teacher_config = ExportParams(
    model_path="runs/multi-head/classroom/20260313_133533_yolov8s-17kpts_Teacher/weights/best.pt",
    onnx_path="runs/_export/classroon/teacher_448x768_20260319.fp32.onnx",
    task="pose-angle",
    input_shape=(1, 3, 448, 768),
    input_names=["input"],
    output_names=[
        "box_cls_kpt_angle_96", "box_cls_kpt_angle_48", "box_cls_kpt_angle_24"
    ],
    export_function=export_teacher,
)

def export_student_qrcode(self, x):
    cv2_0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
    cv3_0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
    ang_0 = [self.cv5[0][i](x[i]) for i in range(self.nl)]  # (bs, 3, h*w)

    cv2_1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
    cv3_1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]
    kpt_0 = [self.cv4[i](x[i]) for i in range(self.nl)]  # (bs, 17*3, h*w)
    kpt_0 = [self.cv4_kpts[i](kpt_0[i]) for i in range(self.nl)]  # (bs, 17*3, h*w)

    cv2_2 = [self.cv2[2][i](x[i]) for i in range(self.nl)]
    cv2_3 = [self.cv2[3][i](x[i]) for i in range(self.nl)]

    cv3_2 = [self.cv3[2][i](x[i]) for i in range(self.nl)]
    cv3_3 = [self.cv3[3][i](x[i]) for i in range(self.nl)]
    ang_1 = [self.cv5[1][i](x[i]) for i in range(self.nl)]  # (bs, 3, h*w)

    cv2_4 = [self.cv2[4][i](x[i]) for i in range(self.nl)]
    cv3_4 = [self.cv3[4][i](x[i]) for i in range(self.nl)]

    cv2_5 = [self.cv2[5][i](x[i]) for i in range(self.nl)]
    cv3_5 = [self.cv3[5][i](x[i]) for i in range(self.nl)]

    # cv2_6 = [self.cv2[6][i](x[i]) for i in range(self.nl)]
    # cv3_6 = [self.cv3[6][i](x[i]) for i in range(self.nl)]
    
    x0 = [torch.cat((cv2_0[i], cv3_0[i], ang_0[i]), 1) for i in range(self.nl)]
    x1 = [torch.cat((cv2_1[i], cv3_1[i], kpt_0[i]), 1) for i in range(self.nl)]
    x2 = [torch.cat((cv2_2[i], cv3_2[i]), 1) for i in range(self.nl)]
    x3 = [torch.cat((cv2_3[i], cv3_3[i], ang_1[i]), 1) for i in range(self.nl)]
    x4 = [torch.cat((cv2_4[i], cv3_4[i]), 1) for i in range(self.nl)]
    x5 = [torch.cat((cv2_5[i], cv3_5[i]), 1) for i in range(self.nl)]
    # x6 = [torch.cat((cv2_6[i], cv3_6[i]), 1) for i in range(self.nl)]

    return (*x0, *x1, *x2, *x3, *x4, *x5)

student_qrcode_config = ExportParams(
    # model_path="runs/mdetectors_qrcode/20260126_add_match/weights/best.pt",
    # model_path="runs/multi-head/train/20260207_231833_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt",
    # model_path="runs/multi-head/qrcode/20260209_192315_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt",
    model_path="runs/multi-head/qrcode/20260310_205410_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt",
    onnx_path="runs/_export/qrcode/student_448x768_qrcode_20260310.fp32.onnx",
    task="multi-detectors",
    input_shape=(1, 3, 448, 768),
    input_names=["input"],
    output_names=[
        "head_96", "head_48", "head_24",
        "body_96", "body_48", "body_24",
        # "raise_hand_96", "raise_hand_48", "raise_hand_24",
        # "stand_up_96", "stand_up_48", "stand_up_24",
        "actions_96", "actions_48", "actions_24",
        "face_96", "face_48", "face_24",
        "qrcode_96", "qrcode_48", "qrcode_24",
        "match_96", "match_48", "match_24",
    ],
    export_function=export_student_qrcode
)

def export_balls_2H(self, x):
    cv2_0 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
    cv3_0 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
    cv2_1 = [self.cv2[1][i](x[i]) for i in range(self.nl)]
    cv3_1 = [self.cv3[1][i](x[i]) for i in range(self.nl)]

    pose = [self.cv4[i](x[i]) for i in range(self.nl)]
    kpts = [self.cv4_kpts[i](pose[i]) for i in range(self.nl)]
    angles = [self.cv5[0][i](x[i]) for i in range(self.nl)]

    x0 = [torch.cat((cv2_0[i], cv3_0[i], kpts[i], angles[i]), 1) for i in range(self.nl)]
    x1 = [torch.cat((cv2_1[i], cv3_1[i]), 1) for i in range(self.nl)]

    return (*x0, *x1)

ball_448x768_2H_config = ExportParams(
    # model_path="runs/multi-head/ball_sports/20260203_183401_Sparse/weights/best.pt",
    # model_path="runs/multi-head/train/20260206_142318_yolo11s_25kpts_2H/weights/best.pt",
    # model_path="runs/multi-head/train/20260206_172434_yolo11s_25kpts_2H_v1/weights/best.pt",
    # model_path="runs/multi-head/train/20260228_190347_best/weights/best.pt",
    # model_path="runs/multi-head/train/20260308_060807_best/weights/best.pt",
    model_path="runs/multi-head/train/20260311_143109_best/weights/best.pt",
    onnx_path="runs/_export/ball_sports/yolov8s_ball_fbvs_448x768_2H_fp32_v260311.onnx",
    task="multi-detectors",
    input_shape=(1, 3, 448, 768),
    input_names=["input"],
    output_names=[
        "head_body_56x96", "head_body_28x48", "head_body_14x24",
        "ball_56x96", "ball_28x48", "ball_14x24",
    ],
    export_function=export_balls_2H
)

def export_running(self, x):
    cv2 = [self.cv2[0][i](x[i]) for i in range(self.nl)]
    cv3 = [self.cv3[0][i](x[i]) for i in range(self.nl)]
    pose = [self.cv4[i](x[i]) for i in range(self.nl)]
    kpts = [self.cv4_kpts[i](pose[i]) for i in range(self.nl)]
    angles = [self.cv5[0][i](x[i]) for i in range(self.nl)]
    x = [torch.cat((cv2[i], cv3[i], kpts[i], angles[i]), 1) for i in range(self.nl)]
    return *x, 

running_416x640_config = ExportParams(
    model_path="/data4/yuanchengzhi/projects/ultralytics_fork/runs/multi-head/.experiments/20260205_145914_yolov8s-25-poseAngles-m/weights/best.pt",
    onnx_path="runs/_export/running/yolov8s_running_416x640_fp32_v260204.onnx",
    task="multi-head",
    input_shape=(1, 3, 416, 640),
    input_names=["input"],
    output_names=[
        "head_body_52x80", "head_body_26x40", "head_body_13x20",
    ],
    export_function=export_running
)

def export_onnx(params: ExportParams):
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    from ultralytics import YOLO
    sys.path.pop()

    print(f"==> Loading model from: {params.model_path}")
    model = YOLO(params.model_path, task=params.task).float()
    model = model.model
    model.fuse()

    m = model.model[-1]
    m.export = True
    m.format = "onnx"
    m.dynamic = False
    if params.export_function is not None:
        m.export_function = params.export_function

    onnx_path = params.onnx_path
    export_dir = os.path.dirname(onnx_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    print("==> Exporting model ...")
    torch.onnx.export(
        model, 
        (torch.randn(*params.input_shape, dtype=torch.float32),),
        onnx_path,
        verbose=False,
        opset_version=params.opset_version,
        do_constant_folding=True,
        input_names=params.input_names,
        output_names=params.output_names
    )

    model_onnx = onnx.load(onnx_path)  # load onnx model
    model_onnx = onnxslim.slim(model_onnx)
    os.remove(onnx_path)
    onnx.save(model_onnx, onnx_path)
    print(f"==> Convert done, model saved to: {onnx_path}")

if __name__ == "__main__":
    export_onnx(teacher_config)
