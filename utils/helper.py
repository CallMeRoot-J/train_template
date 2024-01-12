import os
import numpy as np


def create_directory(directory_path):
    """ 检查文件夹是否存在，如果不存在，则创建文件夹
    """
    #
    if not os.path.exists(directory_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(directory_path)

    return directory_path


def print_model_parameters(model):
    """ 打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")
