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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import ssl
import torch # pylint: disable=E0401
import torch.nn as nn # pylint: disable=E0401
import torch.backends.cudnn as cudnn # pylint: disable=E0401
import torch.distributed as dist # pylint: disable=E0401
import torch.optim # pylint: disable=E0401
import torch.multiprocessing as mp # pylint: disable=E0401
import torch.utils.data # pylint: disable=E0401
import torchvision.transforms as transforms # pylint: disable=E0401
import torchvision.datasets as datasets # pylint: disable=E0401
import torchvision.models as models # pylint: disable=E0401
import onnxruntime as ort # pylint: disable=E0401

import hotwheels.amct_pytorch as amct # pylint: disable=E0401

PATH, _ = os.path.split(os.path.realpath(__file__))
TMP = os.path.join(PATH, 'tmp')
SIZE = 224
RESULTS = os.path.join(PATH, 'results/retrain_results')
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
# Ignore verification of website certificate when downloading resnet50 from the Internet
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config_defination', dest='config_defination',
                        default=None, type=str,
                        help='The simple configure define file.')
    parser.add_argument('--batch_num', dest='batch_num', default=2, type=int,
                        help='number of total batch to run')
    parser.add_argument('--train_set', dest='train_set',
                        default=None, type=str,
                        help='The path of ILSVRC-2012-CLS image classification'
                        ' dataset for training.')
    parser.add_argument('--eval_set', dest='eval_set', default=None, type=str,
                        help='The path of ILSVRC-2012-CLS image classification'
                        ' dataset for evaluation.')
    parser.add_argument('--num_parallel_reads', dest='num_parallel_reads',
                        default=4, type=int,
                        help='The number of files to read in parallel.')
    parser.add_argument('--batch_size', dest='batch_size',
                        default=25, type=int,
                        help='batch size (default: 25)')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        default=1e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--train_iter', dest='train_iter',
                        default=2000, type=int,
                        help='number of total iterations to run')
    parser.add_argument('--print_freq', dest='print_freq',
                        default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--dist_url', dest='dist_url',
                        default='tcp://127.0.0.1:50011', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--distributed',
                        dest='distributed', action='store_true',
                        help='Use multi-processing distributed training')
    return parser.parse_args()


def args_check(args):
    """Verify the validity of input parameters"""
    if args.train_set is None:
        raise RuntimeError('Must specify a training dataset path!')
    args.train_set = os.path.realpath(args.train_set)
    if not os.access(args.train_set, os.F_OK):
        raise RuntimeError('Must specify a valid training dataset path!')

    if args.eval_set is None:
        raise RuntimeError('Must specify a evaluation dataset path!')
    args.eval_set = os.path.realpath(args.eval_set)
    if not os.access(args.eval_set, os.F_OK):
        raise RuntimeError('Must specify a valid evaluation dataset path!')


class AverageCounter:
    """Compute and store the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset_param()
        self.value = 0
        self.average = 0

    def __str__(self):
        fmtstr = ''.join(
            ['{name} {value', self.fmt, '} ({average', self.fmt, '})'])
        return fmtstr.format(**self.__dict__)

    def reset_param(self):
        """reset param"""
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count_num = 0

    def update_param(self, value, size=1):
        """Update param"""
        self.value = value
        self.sum += value * size
        self.count_num += size
        self.average = self.sum / self.count_num


class ProgressCounter:
    """Manage and show the information of training and validation"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self.get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    @staticmethod
    def get_batch_fmtstr(num_batches):
        """get batch fmt string"""
        num_digits = len(str(num_batches // 1))
        fmt = ''.join(['{:', str(num_digits), 'd}'])
        return ''.join(['[', fmt, '/', fmt.format(num_batches), ']'])

    def display(self, batch):
        """display current batch info"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


def get_dummy_input(shape_list, model):
    """Get input data to generate onnx graph for amct_pytorch tools"""
    device = next(model.parameters()).device
    dummy_input = tuple([torch.randn(shape).to(device) # pylint: disable=E1101
                        for shape in shape_list])
    return dummy_input


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Set the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def accuracy(output, target, top_k=(1,)):
    """Compute the accuracy over the k top predictions"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, prediction = output.topk(max_k, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))

        result = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def train(train_loader, model, optimizer, # pylint: disable=R0913, R0914
          iteration, gpu_index, print_freq):
    """Train the model"""
    batch_time = AverageCounter('Time', ':6.3f')
    data_time = AverageCounter('Data', ':6.3f')
    losses = AverageCounter('Loss', ':.4e')
    top1 = AverageCounter('Acc@1', ':6.2f')
    top5 = AverageCounter('Acc@5', ':6.2f')
    progress = ProgressCounter(
        iteration,
        [batch_time, data_time, losses, top1, top5],
        prefix='Train: ')

    # switch to train mode.
    model.train()

    criterion = nn.CrossEntropyLoss()
    if gpu_index >= 0:
        criterion = nn.CrossEntropyLoss().cuda(gpu_index)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time.
        data_time.update_param(time.time() - end)

        if gpu_index >= 0:
            images = images.cuda(gpu_index, non_blocking=True)
            target = target.cuda(gpu_index, non_blocking=True)

        # compute output.
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss.
        acc_result = accuracy(output, target, top_k=(1, 5))
        acc1, acc5 = acc_result[0], acc_result[1]
        losses.update_param(loss.item(), images.size(0))
        top1.update_param(acc1[0], images.size(0))
        top5.update_param(acc5[0], images.size(0))

        # compute gradient and do SGD step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time.
        batch_time.update_param(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            progress.display(i + 1)
        if (i + 1) >= iteration:
            break


def validate(val_loader, model, iteration, # pylint: disable=R0913, R0914
             gpu_index, print_freq):
    """Validate the model"""
    batch_time = AverageCounter('Time', ':6.3f')
    losses = AverageCounter('Loss', ':.4e')
    top1 = AverageCounter('Acc@1', ':6.2f')
    top5 = AverageCounter('Acc@5', ':6.2f')
    progress = ProgressCounter(
        iteration,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode.
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if gpu_index >= 0:
        criterion = nn.CrossEntropyLoss().cuda(gpu_index)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu_index >= 0:
                images = images.cuda(gpu_index, non_blocking=True)
                target = target.cuda(gpu_index, non_blocking=True)

            # compute output.
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss.
            acc_result = accuracy(output, target, top_k=(1, 5))
            acc1, acc5 = acc_result[0], acc_result[1]
            losses.update_param(loss.item(), images.size(0))
            top1.update_param(acc1[0], images.size(0))
            top5.update_param(acc5[0], images.size(0))

            # measure elapsed time.
            batch_time.update_param(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                progress.display(i + 1)
            if (i + 1) >= iteration:
                break

    return top1.average, top5.average


def validate_onnx(val_loader, model, print_freq): # pylint: disable=R0914
    """Validate the onnx model"""
    batch_time = AverageCounter('Time', ':6.3f')
    losses = AverageCounter('Loss', ':.4e')
    top1 = AverageCounter('Acc@1', ':6.2f')
    top5 = AverageCounter('Acc@5', ':6.2f')
    progress = ProgressCounter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode.
    ort_session = ort.InferenceSession(model)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output.
            output = ort_session.run(None, {'input': images.numpy()})
            output = torch.from_numpy(output[0]) # pylint: disable=E1101
            loss = criterion(output, target)

            # measure accuracy and record loss.
            acc_result = accuracy(output, target, top_k=(1, 5))
            acc1, acc5 = acc_result[0], acc_result[1]
            losses.update_param(loss.item(), images.size(0))
            top1.update_param(acc1[0], images.size(0))
            top5.update_param(acc5[0], images.size(0))

            # measure elapsed time.
            batch_time.update_param(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                progress.display(i + 1)

    return top1.average, top5.average


def create_data_loader(train_set_dir, test_set_dir, args):
    """Generate training dataset loader."""
    traindir = os.path.realpath(train_set_dir)
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORM,
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_parallel_reads, pin_memory=True,
        sampler=train_sampler)
    # Generate validation dataset loader.
    valdir = os.path.realpath(test_set_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(SIZE),
            transforms.ToTensor(),
            NORM,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_parallel_reads, pin_memory=True)

    return train_loader, train_sampler, val_loader


def cal_original_model_accuracy(model, gpu_index, val_loader, args):
    """Infer the accuracy of the original model."""
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)
        model = model.cuda(gpu_index)

    # Validation origin model.
    print("=> Validate pre-trained model 'resnet50'")
    ori_top1, ori_top5 = validate(
        val_loader, model, len(val_loader), gpu_index, args.print_freq)
    print('The origin model top 1 accuracy = {:.2f}%.'.format(ori_top1))
    print('The origin model top 5 accuracy = {:.2f}%.'.format(ori_top5))

    return ori_top1, ori_top5


def train_and_val(model, gpu_index, train_loader, # pylint: disable=R0913
                  train_sampler, val_loader, args):
    """train and validation"""
    # Allocating a model to a specified device.
    if gpu_index >= 0:
        if args.distributed:
            torch.cuda.set_device(gpu_index)
            model.cuda(gpu_index)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu_index], find_unused_parameters=True)
        else:
            torch.cuda.set_device(gpu_index)
            model = model.cuda(gpu_index)

    # Define optimizer.
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4)

    # Retrain the model.
    for epoch in range(0, 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.learning_rate)

        # train for train_iter.
        print("=> training quantized model")
        train(train_loader, model, optimizer, args.train_iter, gpu_index,
              args.print_freq)

        # evaluate on validation set.
        validate(val_loader, model, args.batch_num, gpu_index,
                 args.print_freq)


def cal_quant_model_accuracy(torch_model, val_loader, args,
                             config_file, record_file):
    """ calculate quant torch model accuracy"""
    # Save the quantized torch model and infer the accuracy of the quantized model.
    torch.save({'state_dict': torch_model.state_dict()},
               os.path.join(TMP, 'model_best.pth.tar'))
    print('==> AMCT step3: save_quant_retrain_model..')
    quantized_pb_path = os.path.join(RESULTS, 'ResNet50')
    amct.save_quant_retrain_model(
        config_file,
        torch_model,
        record_file,
        quantized_pb_path,
        get_dummy_input([(1, 3, SIZE, SIZE)], torch_model),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print("=> validating fake quant model")
    quant_top1, quant_top5 = validate_onnx(
        val_loader, ''.join([quantized_pb_path, '_fake_quant_model.onnx']),
        args.print_freq)
    return quant_top1, quant_top5


def main():
    """main function"""
    args = parse_args()
    args_check(args)

    cudnn.benchmark = True
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
    else:
        gpu_num = 0

    if args.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function.
        if gpu_num == 0:
            raise RuntimeError('Must has at least one available GPU in '
                               'distributed training mode!')
        print('Using multi GPUs: DistributedDataParallel mode.')
        mp.spawn(main_worker, nprocs=gpu_num, args=(gpu_num, args))
    else:
        # Simply call main_worker function.
        if gpu_num > 0:
            gpu_index = 0
            print('Using single GPU.')
        else:
            gpu_index = -1
            print('Using CPU, this will be slow')
        main_worker(gpu_index, gpu_num, args)


def main_worker(gpu_index, gpu_num, args):
    """
    Phase initialization.
    If multi-card distributed training is used, initialize the training
    process.
    """
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=gpu_num, rank=gpu_index)

    # Generate training dataset and validation dataset loader.
    train_loader, train_sampler, val_loader = \
        create_data_loader(args.train_set, args.eval_set, args)

    # Phase origin torch model accuracy.
    # Step 1: Create torch model.
    print("=> Create pre-trained model 'resnet50'")
    # Choose whether to use the torch model downloaded online or
    # the torch model modified locally.
    # 1. use the resnet50 torch model downloaded online.
    use_online = True
    if use_online:
        torch_model = models.resnet50(pretrained=True)
    else:
        # 2. use the resnet50 torch model modified locally.
        from resnet50_model.resnet import resnet50 # pylint: disable=E0401, C0415
        torch_model = resnet50(pretrained=True)

    # Step 2: Calculate origin model's accuracy.
    ori_top1, ori_top5 = cal_original_model_accuracy(
        torch_model, gpu_index, val_loader, args)

    # Phase retrain the model.
    # Step 1: Create the retraining configuration file.
    print('==> AMCT step1: create_quant_retrain_config..')
    config_file = os.path.join(TMP, 'config.json')
    record_file = os.path.join(TMP, 'record.txt')
    # switch to train mode.
    torch_model.train()
    amct.create_quant_retrain_config(
        config_file,
        torch_model,
        get_dummy_input([(1, 3, SIZE, SIZE)], torch_model),
        args.config_defination)
    # Step 2: Generate the retraining model in default graph and create the
    # quantization factor record_file.
    print('==> AMCT step2: create_quant_retrain_model..')
    torch_model = amct.create_quant_retrain_model(
        config_file,
        torch_model,
        record_file,
        get_dummy_input([(1, 3, SIZE, SIZE)], torch_model))

    # Step 3: Retraining quantitative model and inferencing.
    train_and_val(torch_model, gpu_index, train_loader, train_sampler, val_loader,
                  args)

    # Step 4: Save the quantized model and infer the accuracy of the
    # quantized model.
    if not args.distributed or (args.distributed and gpu_index == 0):
        quant_top1, quant_top5 = \
            cal_quant_model_accuracy(torch_model, val_loader, args,
                                     config_file, record_file)

        print('[INFO] ResNet50 before retrain top1:{:.2f}% '
              'top5:{:.2f}%'.format(ori_top1, ori_top5))
        print('[INFO] ResNet50 after retrain top1:{:.2f}% '
              'top5:{:.2f}%'.format(quant_top1, quant_top5))


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=3 python resnet50_retrain_sample.py --train_set ./data/train --eval_set ./data/val --config_defination ./retrain_conf/retrain.cfg 