import sys,os
from rknn.api import RKNN

DATASET_PATH = os.path.dirname(__file__) + '/dataset.txt' #'../../../datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = "runs/_export/rknn/ball_sports/hybrid/yolov8s_ball_fbvs_448x768_2H_rk3576_hybrid_sparse42_v260203.rknn" #'../model/yolov8_pose.rknn'
# DEFAULT_RKNN_PATH = "runs/_export/rknn/ball_sports/hybrid/yolov8s_ball_fbvs_448x768_2H_rk3576_hybrid_v260131.rknn" #'../model/yolov8_pose.rknn'
DEFAULT_QUANT = True

if not os.path.exists(os.path.dirname(DEFAULT_RKNN_PATH)):
    os.makedirs(os.path.dirname(DEFAULT_RKNN_PATH))

def parse_arg():
    model_path = "runs/_export/ball_sports/yolov8s_ball_fbvs_448x768_2H_fp32_sparse42_v260203_simplified.onnx"
    # model_path = "runs/_export/ball_sports/yolov8s_ball_fbv_448x768_2H_fp32_v260131_simplified.onnx"
    platform = "rk3576"
    do_quant = DEFAULT_QUANT
    return model_path, platform, do_quant, DEFAULT_RKNN_PATH

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')

    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform=platform,
        sparse_infer=True,
    )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    if platform in ["rv1109","rv1126","rk1808"] :
        ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH, auto_hybrid_quant=True)
    else:
        if do_quant:
            rknn.hybrid_quantization_step1(
                dataset=DATASET_PATH,
                proposal=False,
                custom_hybrid=[
                    ['/model.21/cv4.0/cv4.0.0/act/Clip_output_0', 'head_body_56x96'],
                    ['/model.21/cv4.1/cv4.1.0/act/Clip_output_0', 'head_body_28x48'],
                    ['/model.21/cv4.2/cv4.2.0/act/Clip_output_0', 'head_body_14x24'],
                    ['/model.21/cv3.1.0/cv3.1.0.0/act/Clip_output_0', 'ball_56x96'],
                    ['/model.21/cv3.1.1/cv3.1.1.0/act/Clip_output_0', 'ball_28x48'],
                    ['/model.21/cv3.1.2/cv3.1.2.0/act/Clip_output_0', 'ball_14x24'],
                ]
            )

            model_name=os.path.basename(model_path).replace('.onnx','')
            rknn.hybrid_quantization_step2(
                model_input = model_name+".model",          # 表示第一步生成的模型文件
                data_input= model_name+".data",             # 表示第一步生成的配置文件
                model_quantization_cfg=model_name+".quantization.cfg"  # 表示第一步生成的量化配置文件
            )
        else:
            ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    image_path = "/data5/yuanchengzhi/Datasets/SmartSports/Balls_3_NoPole/images/train/captures-1/vlc_record_2025_12_15_15h36m13s_rtsp___192_168_1_64_554_Streaming_channels_1__009375.jpg"
    ret = rknn.accuracy_analysis(inputs=[image_path], output_dir=None)
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print("output_path:",output_path)
    print('done')
    # Release
    rknn.release()
