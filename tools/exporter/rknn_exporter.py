from typing import List, Optional
from dataclasses import dataclass, field

import os
from tqdm import tqdm

from rknn.api import RKNN

def get_rknn_version():
    import importlib.metadata
    return importlib.metadata.version("rknn-toolkit2")

def check_rknn(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret != 0:
            raise RuntimeError(f"{func.__name__} failed, ret={ret}")
        return ret
    return wrapper

def export_header(rknn_model_path: str, header_name: Optional[str] = None):
    # Check file path
    if header_name is None:
        header_name = os.path.basename(rknn_model_path).rsplit(".rknn")[0] + ".h"
    
    for invalid_char in [" ", "/", "\\", ":", "*", "?", "<", ">", "|"]:
        if invalid_char in header_name:
            print("==> Find invalid character: ", invalid_char)
            header_name = header_name.replace(invalid_char, "_")
    
    # Read model file
    with open(rknn_model_path, "rb") as f:
        f.seek(0, 2)
        model_size = f.tell()
        f.seek(0, 0)
        model_bytes = f.read(model_size)
    
    if len(model_bytes) != model_size:
        print("Read file failed.")
        return None
    
    # Export header
    header_path = os.path.join(os.path.dirname(rknn_model_path), header_name)
    macro = header_name.replace(".", "_").upper()
    variable_name = header_name.rsplit(".", 1)[0].replace(".", "_")

    uchar_data = list(model_bytes)
    uchar_data_str = f"unsigned char {variable_name}_model[] = " + "{\n"
    for i, v in tqdm(enumerate(uchar_data), total=model_size, ncols=100, desc="Exporting header:"):
        uchar_data_str += f"{v:3d},"
        if (i + 1) % 32 == 0 and (i + 1) != model_size:
            uchar_data_str += "\n"
    uchar_data_str = uchar_data_str[:-1] + "};\n"

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(f"#ifndef {macro}\n")
        f.write(f"#define {macro}\n")
        f.write(f"\n")
        f.write(f"unsigned long {variable_name}_model_size = {model_size}UL;\n")
        f.write(f"{uchar_data_str}")
        f.write(f"\n")
        f.write(f"#endif // {macro}\n")

    print(f"==> Export header: {header_path}.")
    return header_path

def export_rknn(
    model_path: str, 
    rknn_path: str, 
    mean_values: List[float] = [[0, 0, 0]],
    std_values: List[float] = [[255, 255, 255]],
    target_platform: str = "rk3588",
    optimization_level: int = 3,
    do_quantization: bool = True,
    dataset_path: str | None = None,
    proposal: bool = True,
    custom_hybrid: List[List[str]] | None = None,
    rknn_batch_size: int = 1,
    sparse_infer: bool = False,
    verbose: bool = True
):
    if not do_quantization:
        assert dataset_path, "dataset_path is required when do_quantization is False"
    if not sparse_infer:
        version = get_rknn_version()
        expected_version = "2.3.0"
        print(f"==> [Warning] RKNN version {version} is less than {expected_version}, some features may not work properly. Please upgrade to {expected_version} or later.")

    assert proposal or len(custom_hybrid) > 0, "proposal=True or custom_hybrid is required"
    if proposal:
        if custom_hybrid and len(custom_hybrid) > 0:
            print("==> [Warning] proposal=True, custom_hybrid will be ignored.")

    export_dir = os.path.dirname(rknn_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Initialize RKNN
    rknn = RKNN(verbose=verbose, verbose_file=os.path.join(export_dir, "verbose.log"))
    check_rknn(
        rknn.config(
            mean_values,
            std_values,
            target_platform=target_platform,
            optimization_level=optimization_level,
            sparse_infer=sparse_infer,
        )
    )

    # Load model
    print(f"==> [Info] Load model: {model_path}")
    check_rknn(
        rknn.load_onnx(model_path,)
    )

    # Build and export RKNN model
    if do_quantization:
        check_rknn(
            rknn.hybrid_quantization_step1(
                dataset=dataset_path,
                proposal=proposal,
                custom_hybrid=custom_hybrid
            )
        )
        model_name = os.path.basename(model_path).rsplit(".", 1)[0]
        check_rknn(
            rknn.hybrid_quantization_step2(
                model_input = model_name + ".model",
                data_input= model_name + ".data",
                model_quantization_cfg=model_name + ".quantization.cfg",
            )
        )
    else:
        check_rknn(
            rknn.build(
                do_quantization=do_quantization, 
                rknn_batch_size=rknn_batch_size
            )
        )

    check_rknn(
        rknn.export_rknn(rknn_path)
    )
    print(f"==> [Info] Export RKNN model: {rknn_path}.")

    # Release resources
    rknn.release()

@dataclass
class ExportParams:
    model_path: str
    rknn_path: str
    mean_values: List[float] = field(default_factory=lambda: [[0, 0, 0]])
    std_values: List[float] = field(default_factory=lambda: [[255, 255, 255]])
    target_platform: str = "rk3588"
    optimization_level: int = 3
    do_quantization: bool = True
    dataset_path: str | None = None
    proposal: bool = True
    custom_hybrid: List[List[str]] | None = None
    rknn_batch_size: int = 1
    sparse_infer: bool = False
    verbose: bool = True

ball_sports_config = ExportParams(
    # model_path="runs/_export/ball_sports/yolov8s_ball_fbvs_448x768_2H_fp32_sparse42_v260203_simplified.onnx",
    # model_path="/data4/yuanchengzhi/projects/ultralytics_fork/runs/multi-head/.experiments/20260205_190521_yolov8s_25kpts_2H/weights/TorchAO/model_torchao_qat.onnx",
    model_path="runs/_export/ball_sports/yolov11s_ball_fbvs_448x768_2H_fp32_v260206.onnx",
    rknn_path="runs/_export/rknn/ball_sports/hybrid/yolov11s_ball_fbvs_448x768_2H_rk3588_hybrid_v260206.rknn",
    dataset_path="/data4/yuanchengzhi/projects/ultralytics_fork/tools/exporter/rknn_hybrid_quant/dataset.txt",
    target_platform="rk3588",
    sparse_infer=False,
    # do_quantization=False,
    proposal=False,
    custom_hybrid=[
        ['/model.21/cv4.0/cv4.0.0/act/Clip_output_0', 'head_body_56x96'],
        ['/model.21/cv4.1/cv4.1.0/act/Clip_output_0', 'head_body_28x48'],
        ['/model.21/cv4.2/cv4.2.0/act/Clip_output_0', 'head_body_14x24'],
        ['/model.21/cv3.1.0/cv3.1.0.0/act/Clip_output_0', 'ball_56x96'],
        ['/model.21/cv3.1.1/cv3.1.1.0/act/Clip_output_0', 'ball_28x48'],
        ['/model.21/cv3.1.2/cv3.1.2.0/act/Clip_output_0', 'ball_14x24'],
    ], 
)

running_config = ExportParams(
    model_path="runs/_export/running/yolov8s_running_416x640_fp32_v260204.onnx",
    rknn_path="runs/_export/rknn/running/hybrid/yolov8s_running_416x640_rk3576_hybrid_v260204.rknn",
    dataset_path="/data4/yuanchengzhi/projects/ultralytics_fork/tools/exporter/rknn_hybrid_quant/dataset.txt",
    target_platform="rk3576",
    custom_hybrid=[
        ['/model.21/cv4.0/cv4.0.0/act/Clip_output_0', 'head_body_52x80'],
        ['/model.21/cv4.1/cv4.1.0/act/Clip_output_0', 'head_body_26x40'],
        ['/model.21/cv4.2/cv4.2.0/act/Clip_output_0', 'head_body_13x20'],
    ], 
    sparse_infer=False
)

if __name__ == "__main__":
    config = ball_sports_config
    export_rknn(**config.__dict__)
    export_header(config.rknn_path)
