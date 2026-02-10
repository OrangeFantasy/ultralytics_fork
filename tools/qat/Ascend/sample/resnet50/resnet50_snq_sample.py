#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (c) CompanyNameMagicTag

This program is free software; you can redistribute it and/or modify it
under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache
License for more details at
    http://www.apache.org/licenses/LICENSE-2.0

This is a sample script that guides users how to apply AMCT toolkit to
classification network.

"""
import os
import ssl
import math
import torch # pylint: disable=E0401
from PIL import Image # pylint: disable=E0401
from torchvision import transforms, models # pylint: disable=E0401
import onnxruntime as ort # pylint: disable=E0401

import hotwheels.amct_pytorch as amct # pylint: disable=E0401


IMG_DIR = 'images'
LABLE_FILE = os.path.join(IMG_DIR, 'image_label.txt')
PATH = os.path.split(os.path.realpath(__file__))[0]
TMP = os.path.join(PATH, 'tmp')
SIZE = 224
RESULT = os.path.join(PATH, 'results/calibration_results')
# Ignore verification of website certificate when downloading resnet50 from the Internet
ssl._create_default_https_context = ssl._create_unverified_context


def get_labels_from_txt(label_file):
    """Read all images' name and label from label_file"""
    images = []
    labels = []
    with open(label_file, 'r') as file_open:
        lines = file_open.readlines()
        for line in lines:
            images.append(line.split(' ')[0])
            labels.append(int(line.split(' ')[1]))
    return images, labels


def prepare_image_input(images):
    """Read all images"""
    input_tensor = torch.zeros(len(images), # pylint: disable=E1101
                               3, SIZE, SIZE)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    for index, image in enumerate(images):
        input_image = Image.open(image).convert('RGB')
        input_tensor[index, ...] = preprocess(input_image)
    return input_tensor


def img_postprocess(probs, labels):
    """Do image post-process"""
    # calculate top1 and top5 accuracy
    top1_get = 0
    top5_get = 0
    prob_size = probs.shape[1]
    for index, label in enumerate(labels):
        top5_record = (probs[index, :].argsort())[prob_size - 5:prob_size]
        if label == top5_record[-1]:
            top1_get += 1
            top5_get += 1
        elif label in top5_record:
            top5_get += 1
    return float(top1_get) / len(labels), float(top5_get) / len(labels)


def model_forward(model, batch_size, iterations):
    """Do pytorch model forward"""
    ori_images, ori_labels = get_labels_from_txt(LABLE_FILE)
    repeat_count = math.ceil(batch_size * iterations / len(ori_images))
    images = ori_images * repeat_count
    labels = ori_labels * repeat_count

    images = [os.path.join(IMG_DIR, image) for image in images]
    top1_total = 0
    top5_total = 0
    for i in range(iterations):
        input_batch = prepare_image_input(
            images[i * batch_size: (i + 1) * batch_size])
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        top1, top5 = img_postprocess(
            output, labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5
        print('****************iteration:{}*****************'.format(i))
        print('top1_acc:{}'.format(top1))
        print('top5_acc:{}'.format(top5))
    return top1_total / iterations, top5_total / iterations


def onnx_forward(onnx_model, batch_size, iterations):
    """Do onnx model forward"""
    ort_session = ort.InferenceSession(onnx_model)

    images, labels = get_labels_from_txt(LABLE_FILE)
    images = [os.path.join(IMG_DIR, image) for image in images]
    top1_total = 0
    top5_total = 0
    for i in range(iterations):
        input_batch = prepare_image_input(
            images[i * batch_size: (i + 1) * batch_size])
        output = ort_session.run(None, {'input': input_batch.numpy()})
        top1, top5 = img_postprocess(
            output[0], labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5
        print('****************iteration:{}*****************'.format(i))
        print('top1_acc:{}'.format(top1))
        print('top5_acc:{}'.format(top5))
    print('******final top1:{}'.format(top1_total / iterations))
    print('******final top5:{}'.format(top5_total / iterations))
    return top1_total / iterations, top5_total / iterations


def main():
    """Sample main function"""
    use_online = True
    if use_online:
        torch_model = models.resnet50(pretrained=True)
    else:
        # use local modified resnet50 pytorch model
        from resnet50_model.resnet import resnet50 # pylint: disable=E0401, C0415
        torch_model = resnet50(pretrained=True)

    torch_model.eval()
    ori_top1, ori_top5 = model_forward(torch_model, batch_size=32, iterations=5)
    print('******original top1:{}'.format(ori_top1))
    print('******original top5:{}'.format(ori_top5))

    # Quantize configurations
    args_shape = [(1, 3, SIZE, SIZE)]
    dummy_input = tuple([torch.randn(arg_shape) # pylint: disable=E1101
                        for arg_shape in args_shape])
    if torch.cuda.is_available():
        dummy_input = tuple([data.to('cuda') for data in dummy_input])
        torch_model.to('cuda')
    config_json_file = os.path.join(TMP, 'config.json')
    skip_layers = []
    batch_num = 2
    config_defination = './cali_conf/calibration_snq.cfg'
    update_bn_iter = 30

    amct.create_quant_config(config_json_file,
                             torch_model,
                             dummy_input,
                             skip_layers,
                             batch_num,
                             activation_offset=True,
                             config_defination=config_defination)
    # Phase1: do conv+bn fusion, weights calibration and generate
    #         calibration pytorch model
    scale_offset_record_file = os.path.join(TMP, 'scale_offset_record.txt')
    modified_model = os.path.join(TMP, 'modified_model.onnx')
    calibrated_torch_model = amct.quantize_model(
        config_json_file,
        modified_model,
        scale_offset_record_file,
        torch_model,
        dummy_input,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # Phase2: update bn status to True
    amct.update_bn_status(calibrated_torch_model, True)
    bn_update_top1, bn_update_top5 = model_forward(calibrated_torch_model,
        batch_size=32, iterations=update_bn_iter)
    print('******after bn update, top1:{}'.format(bn_update_top1))
    print('******after bn update, top5:{}'.format(bn_update_top5))

    # Phase2: update bn status to False and do activation calibration
    amct.update_bn_status(calibrated_torch_model, False)
    atcs_cali_top1, atcs_cali_top5 = model_forward(calibrated_torch_model,
        batch_size=32, iterations=batch_num)
    print('******after activation calibration, top1:{}'.format(atcs_cali_top1))
    print('******after activation calibration, top5:{}'.format(atcs_cali_top1))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Phase3: save final model, one for onnx do fake quant test, one
    #         deploy model for ATC
    result_path = os.path.join(RESULT, 'ResNet50')
    amct.save_model(modified_model, scale_offset_record_file, result_path,
        calibrated_torch_model)

    # Phase4: run fake_quant model test
    quant_top1, quant_top5 = onnx_forward(
        '%s_%s' % (result_path, 'fake_quant_model.onnx'),
        batch_size=32,
        iterations=5)
    print('[INFO] ResNet50 before quantize top1:{:>10} top5:{:>10}'.format(
        ori_top1, ori_top5))
    print('[INFO] ResNet50 after quantize top1:{:>10} top5:{:>10}'.format(
        quant_top1, quant_top5))


if __name__ == '__main__':
    main()
