#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
Copyright (c) CompanyNameMagicTag

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

AMCT_PYTORCH sample of resnet50 model

"""
import time
import os
from io import BytesIO
import onnx
import onnxruntime
import numpy as np
import torch

from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model


def has_dequant_node(layer_name, node_names):
    for node_name in node_names:
        if node_name.find('{}.dequant.fakedequant.mul_dequant_layer'.format(
            layer_name)) != -1:
            return True
    return False


def dump_out(model_path, dump_model, dump_result_dir):
    dump_result_dir = os.path.realpath(dump_result_dir)
    if not os.path.exists(dump_result_dir):
        os.makedirs(dump_result_dir)
    else:
        import shutil
        shutil.rmtree(dump_result_dir)
        os.makedirs(dump_result_dir)

    #修改模型，增加输出节点
    model_onnx = onnx.load(model_path)
    output = []
    for out in enumerate_model_node_outputs(model_onnx):
        output.append(out)

    num_onnx = select_model_inputs_outputs(model_onnx, outputs=output)
    save_onnx_model(num_onnx, dump_model)

    #推理得到输出，本示例中采用随机数作为输入
    dummy_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
    dummy_input.tofile("test_data.bin")
    sess = onnxruntime.InferenceSession(dump_model)
    input_name = sess.get_inputs()[0].name
    output_name = [node.name for node in sess.get_outputs()]
    res = sess.run(output_name, {input_name: dummy_input})

    #获得输出名称，确保每个算子节点有对应名称
    node_name = []
    name_node_dict = {}
    for node in model_onnx.graph.node:
        name_node_dict[node.name] = node
        if node.output:
            for _ in range(len(node.output)):
                node_name.append(node.name)

    #保存数据，针对多个输出的节点要适配output_index的循环
    node_name_set = []
    output_index = 0
    for idx, data in enumerate(res):
        if node_name[idx] not in node_name_set:
            node_name_set.append(node_name[idx])
            output_index = 0
        if has_dequant_node(node_name[idx], node_name):
            continue
        layer_name = node_name[idx].replace("/", "_").replace(".", "_").\
                    replace(" ", "_").replace("\\", "_")
        file_name = '{}.{}.{}.npy'.format(layer_name, str(output_index), \
            str(round(time.time() * 1000000)))
        output_dump_path = os.path.join(dump_result_dir, file_name)
        np.save(output_dump_path, data.astype(np.float32))
        if node_name[idx] in name_node_dict.keys() and \
                name_node_dict.get(node_name[idx]).op_type == 'Mul' and \
            node_name[idx].find('.dequant.fakedequant') != -1:
            dequant_index = node_name[idx].find('.dequant.fakedequant')
            layer_name = node_name[idx][:dequant_index]
            layer_name = layer_name.replace("/", "_").replace(".", "_").\
                replace(" ", "_").replace("\\", "_")
            file_name = '{}.{}.{}.npy'.format(layer_name, str(output_index),
                str(round(time.time() * 1000000)))
            output_dump_path = os.path.join(dump_result_dir, file_name)
            np.save(output_dump_path, data.astype(np.float32))
            print('layer_name: {} and {} are the same'.format(layer_name, \
                node_name[idx]))
        output_index = output_index + 1


def main():
    float_model_path = "./tmp/calibration_float.onnx"
    float_dump_model = "float_dump.onnx"
    float_result_dir = "./float_dump_result/"

    fakequant_model_path = \
        "./results/calibration_results/ResNet50_fake_quant_model.onnx"
    fakequant_dump_model = "ResNet50_dump.onnx"
    fakequant_result_dir = "./fake_quant_dump_result/"

    dump_out(float_model_path, float_dump_model, float_result_dir)
    dump_out(fakequant_model_path, fakequant_dump_model, fakequant_result_dir)


if __name__ == '__main__':
    main()

