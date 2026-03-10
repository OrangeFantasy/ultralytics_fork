# python main.py \
#     --model cfg/qrcode/models/yolov8s_19kpts_6H.yaml \
#     --pretrained /data4/yuanchengzhi/projects/yolo/MultiTaskDetector/runs/mdetectors_qrcode/qrcode_0112/weights/best.fp16.state_dict.pt \
#     --data cfg/qrcode/datasets/19kpts_6H.yaml --mode train --nkpts 19 \
#     --device 0 --epochs 100 --batch 256 --workers 8 --imgsz 448 768 \
#     --override_hyp "{ 'scale': 0.6, 'albumentations': 0.0, 'close_mosaic': 10, 'fliplr': 0.0 }"

# with dfl, no rle
python main.py \
    --model cfg/ball_sports/models/yolov8s_25kpts_2H.yaml \
    --pretrained /data4/yuanchengzhi/projects/ultralytics_fork/runs/yolo26_arch_best.fp16.state_dict.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 0 --epochs 100 --batch 256 --workers 16 --imgsz 448 768 \
    --override_hyp "{ 'optimizer': 'SGD', 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20 }"

python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 2 --epochs 100 --batch 256 --imgsz 448 768 --act relu \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 0.75, 'kobj': 0.25, 'rle': 0.75 }" \
    --workers 8 --logging tensorboard

python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 0 --epochs 100 --batch 256 --imgsz 448 768 --act relu \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard

python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 0 --epochs 100 --batch 256 --imgsz 448 768 --act relu \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 0.75 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard


python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 0 --epochs 100 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard

python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 1 --epochs 100 --batch 256 --imgsz 448 768 --act relu \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 12.5, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard

python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml --mode train --nkpts 25 \
    --device 2 --epochs 100 --batch 256 --imgsz 448 768 --act relu \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard # add mosaic data


python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml --mode train --nkpts 25 \
    --device 2 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard # add mosaic data  -  final

# python main.py \
#     --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
#     --data cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml --mode train --nkpts 25 \
#     --device 2 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
#     --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
#     --workers 8 --logging tensorboard


python main.py \
    --model runs/multi-head/train/20260127_154721_yolov8s_25kpts_2H/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 2 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard # 0306

python main.py \
    --model runs/multi-head/train/20260306_171439_best/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 2 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard # 0308

python main.py \
    --model runs/multi-head/train/20260308_060807_best/weights/best.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid.yaml --mode train --nkpts 25 \
    --device 0 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard # 0308

# python main.py \
#     --model runs/multi-head/train/20260131_165216_best/weights/best.pt \
#     --data cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml --mode train --nkpts 25 \
#     --device 1 --epochs 100 --batch 256 --imgsz 448 768 --act relu6 \
#     --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
#     --workers 8 --logging tensorboard --sparse --sparse_mode 0 --project ball_sports --name Sparse



python main.py \
    --model cfg/ball_sports/models/yolo11s_25kpts_2H.yaml \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml --mode train --nkpts 25 \
    --device 1 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 30 }" \
    --workers 8 --logging tensorboard

python main.py \
    --model cfg/ball_sports/models/yolo11s_25kpts_2H.yaml \
    --pretrained runs/multi-head/train/20260131_165216_best/weights/best.fp16.state_dict.pt \
    --data cfg/ball_sports/datasets/25kpts_2H_Solid_Mosaic.yaml --mode train --nkpts 25 \
    --device 1 --epochs 300 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'albumentations': 1.0, 'close_mosaic': 20, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5 }" \
    --workers 8 --logging tensorboard # add mosaic data




python main.py \
    --model cfg/qrcode/models/yolov8s_19kpts_MergeRaiseHandAndStandUp.yaml \
    --pretrained runs/multi-head/qrcode/20260202_201603_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.fp16.state_dict.pt \
    --data cfg/qrcode/datasets/19kpts_MergeRaiseHandAndStandUp.yaml --mode train --nkpts 19 \
    --device 2 --epochs 100 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }" \
    --workers 8 --logging tensorboard --project qrcode

python main.py \
    --model cfg/qrcode/models/yolov8s_19kpts.yaml \
    --pretrained runs/multi-head/qrcode/20260202_201603_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.fp16.state_dict.pt \
    --data cfg/qrcode/datasets/19kpts.yaml --mode train --nkpts 19 \
    --device 2 --epochs 100 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }" \
    --workers 8 --logging tensorboard --project qrcode






python main.py \
    --model runs/multi-head/qrcode/20260205_154727_best/weights/last.pt \
    --data cfg/qrcode/datasets/19kpts_MergeRaiseHandAndStandUp.yaml --mode train --nkpts 19 \
    --device 3 --epochs 100 --batch 256 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'cls': 1.0 ,'kobj': 0.25, 'rle': 0.5, 'resume': 1 }" \
    --workers 8 --logging tensorboard --project qrcode











python main_qat.py \
    --model "../cfg/qrcode/models/yolov8s_19kpts_MergeRaiseHandAndStandUp.yaml" \
    --pretrained "../runs/multi-head/train/20260207_231833_yolov8s_19kpts_MergeRaiseHandAndStandUp/weights/best.pt" \
    --data "../cfg/qrcode/datasets/19kpts_MergeRaiseHandAndStandUp.yaml" --mode train --nkpts 19 \
    --device 0 --epoch 10 --batch 128 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }" \
    --workers 8 --logging tensorboard

python main_qat.py \
    --model "../cfg/qrcode/models/yolov8s_19kpts.yaml" \
    --pretrained "../runs/multi-head/qrcode/20260209_192404_yolov8s_19kpts/weights/best.pt" \
    --data "../cfg/qrcode/datasets/19kpts.yaml" --mode train --nkpts 19 \
    --device 2 --epoch 10 --batch 128 --imgsz 448 768 --act relu6 \
    --override_hyp "{ 'scale': 0.6, 'fliplr': 0.0, 'albumentations': 0.0, 'mosaic': 0.0, 'box': 10.0, 'dfl': 2.0, 'rle': 0.75 }" \
    --workers 0 --logging tensorboard
