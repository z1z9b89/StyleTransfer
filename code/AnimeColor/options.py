#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import random
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='用GAN着色')
        parser.add_argument('--seed', type=int, default=0, metavar='S', help='随机种子（默认值：0）')
        parser.add_argument('--name', type=str, default='CGAN', help='任意型号名称（默认值：CGAN）')
        parser.add_argument('--mode', default=0, help='运行模式[0：训练，1：测试，2：turing-test]（默认值：0）')
        parser.add_argument('--dataset', type=str, default='places365', help='数据集的名称[places365，cifar10]（默认值：places365）')
        parser.add_argument('--dataset-path', type=str, default='./dataset', help='数据集路径（默认值：./ dataset）')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='模型保存在此处（默认值：./ checkpoints）')
        parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='输入培训的批次大小（默认值：16）')
        parser.add_argument('--color-space', type=str, default='lab', help='模型色彩空间[lab，rgb]（默认值：lab）')
        parser.add_argument('--epochs', type=int, default=30, metavar='N', help='要训练的次数（默认值：30）')
        parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='学习率（默认值：3e-4）')
        parser.add_argument('--lr-decay', type=str2bool, default=True, help='是否进行学习率衰减（默认值：True）')
        parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='学习率呈指数衰减率（默认值：0.1）')
        parser.add_argument('--lr-decay-steps', type=float, default=5e5, help='学习率呈指数衰减步长（默认值：1e5）')
        parser.add_argument('--beta1', type=float, default=0, help='Adam优化器的动量项（默认值：0）')
        parser.add_argument("--l1-weight", type=float, default=100.0, help="生成器梯度在L1项上的权重（默认值：100.0）")
        parser.add_argument('--augment', type=str2bool, default=True, help='是否进行数据增强 （默认值：True）')
        parser.add_argument('--label-smoothing', type=str2bool, default=False, help='是否进行单面标签平滑处理（默认值：False）')
        parser.add_argument('--acc-thresh', type=float, default=2.0, help="准确度阈值（默认值：2.0）")
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ID：例如 0 0,1,2，0,2。 CPU使用-1')
        
        parser.add_argument('--save', type=str2bool, default=True, help='是否保存,（默认值：True）')
        parser.add_argument('--save-interval', type=int, default=1000, help='保存模型之前要等待多少批次（默认值：1000）')
        parser.add_argument('--sample', type=str2bool, default=True, help='是否进行采样处理')
        parser.add_argument('--sample-size', type=int, default=8, help='要采样的图像数（默认值：8）')
        parser.add_argument('--sample-interval', type=int, default=1000, help='采样前要等待多少批次（默认值：1000）')
        parser.add_argument('--validate', type=str2bool, default=True, help='是否进行校验（默认值：True）')
        parser.add_argument('--validate-interval', type=int, default=0, help='校验之前要等待多少批次（默认值：0）')
        parser.add_argument('--log', type=str2bool, default=False, help='是否进行记录')
        parser.add_argument('--log-interval', type=int, default=10, help='记录训练状态之前要等待多少次迭代（默认值：10）')
        parser.add_argument('--visualize', type=str2bool, default=False, help='是否进行进度的可视化 (默认值False)')
        parser.add_argument('--visualize-window', type=int, default=100, help='指数移动平均窗口宽度（默认值：100）')
        
        parser.add_argument('--test-input', type=str, default='', help='灰度图像目录或灰度文件的路径')
        parser.add_argument('--test-output', type=str, default='', help='测试模型输出目录')
        parser.add_argument('--turing-test-size', type=int, default=100, metavar='N', help='图灵测试次数（默认值：100）')
        parser.add_argument('--turing-test-delay', type=int, default=0, metavar='N', help='图灵测试时等待的秒数，0表示无限制（默认值：0）')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        opt.color_space = opt.color_space.upper()
        opt.training = opt.mode == 1

        if opt.seed == 0:
            opt.seed = random.randint(0, 2**31 - 1)

        if opt.dataset_path == './dataset':
            opt.dataset_path += ('/' + opt.dataset)

        if opt.checkpoints_path == './checkpoints':
            opt.checkpoints_path += ('/' + opt.dataset)

        return opt
