docker load -i sophgo-tpuc_dev-v3.2_191a433358ad.tar.gz
docker run --privileged --name sophgo_tpuc_dev -v $PWD:/workspace -it sophgo/tpuc_dev:v3.2

$ source ./envsetup.sh

rm -r workspace
mkdir workspace && cd workspace

model_transform.py \
    --model_name v8s_teacher \
    --model_def ../v8s_teacher_deploy_model.onnx \
    --input_shapes [[1,3,768,768]] \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --mlir v8s_teacher.mlir \

model_deploy.py \
  --mlir v8s_teacher.mlir --model_version 1.0 \
  --quantize INT8 \
  --processor bm1688 \
  --calibration_table ../v8s_teacher_cali_table_from_sophgo_mq_sophgo_tpu \
  --quantize_table ../v8s_teacher_q_table_from_sophgo_mq_sophgo_tpu \
  --model v8s_teacher_bm1688_int8.bmodel \
