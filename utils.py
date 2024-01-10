import math
import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F


def weights_init(init_type):
    '''Generate a init function given a init type'''
    def init_fun(m):
        classname = m.__class__.__name__
        # First we check if the layer has custom init method.
        # If so, we just call it without our uniform initialization.
        if hasattr(m, 'custom_init'):
            m.custom_init()
        # Call our unifrom initialization methods for all Conv and Linear layers
        elif (classname.startswith('Conv') or classname.startswith('Linear')) \
                and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'default':
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        if hasattr(m, 'post_init'):
            m.post_init()

    return init_fun


def build_optimizer(optim_type, parameters, lr, weight_decay=0.0, **kwargs):
    if optim_type == 'adamw':
        opt = optim.AdamW(parameters,
                          lr=lr,
                          betas=(kwargs.get('beta1', 0.9),
                                 kwargs.get('beta2', 0.999)),
                          eps=1e-8,
                          weight_decay=weight_decay)
    elif optim_type == 'sgd':
        opt = optim.SGD(parameters, lr=lr, momentum=0,
                        dampening=0, weight_decay=weight_decay)
    elif optim_type == 'sgd_momentum':
        opt = optim.SGD(parameters,
                        lr=lr,
                        momentum=kwargs.get('momentum', 0.9),
                        dampening=kwargs.get('dampening', 0.1),
                        weight_decay=weight_decay)
    else:
        assert 0, f"Unsupported optimizer: {optim_type}"

    return opt


def build_lr_scheduler(optimizer, lr_schedule_type='constant', last_it=-1, **kwargs):
    if lr_schedule_type == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer,
                                                  factor=1.0,
                                                  total_iters=0,
                                                  last_epoch=last_it)
    elif lr_schedule_type == 'step':
        step_size = kwargs.get('step_size', 50000)
        step_gamma = kwargs.get('step_gamma', 0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=step_size,
                                              gamma=step_gamma,
                                              last_epoch=last_it)
    else:
        assert 0, f"Unsupported lr scheduler: {lr_schedule_type}"

    return scheduler


def weight_clipping(named_parameters, clip_parameters):
    named_parameters = dict(named_parameters)
    for group in clip_parameters:
        min_weight = group['min_weight']
        max_weight = group['max_weight']
        for i, param_name in enumerate(group['params']):
            p = named_parameters[param_name]
            p_data_fp32 = p.data
            if 'virtual_params' in group:
                virtual_param_name = group['virtual_params'][i]
                virtual_param = named_parameters[virtual_param_name]
                virtual_param = virtual_param.repeat(*[
                    p_data_fp32.shape[i] // virtual_param.shape[i]
                    for i in range(virtual_param.ndim)
                ])
                min_weight_t = p_data_fp32.new_full(
                    p_data_fp32.shape, min_weight) - virtual_param
                p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                max_weight_t = p_data_fp32.new_full(
                    p_data_fp32.shape, max_weight) - virtual_param
                p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
            else:
                p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)


def cross_entropy_with_softlabel(input, target, reduction='mean', adjust=False, weight=None):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input,
        each item must be a valid distribution: target[i, :].sum() == 1.
    :param adjust: subtract soft-label bias from the loss
    :param weight: (batch, *) same shape as input, 
        if not none, a weight is specified for each loss item
    """
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)
    if weight is not None:
        weight = weight.view(weight.shape[0], -1)

    logprobs = F.log_softmax(input, dim=1)
    if weight is not None:
        logprobs = logprobs * weight
    batchloss = -torch.sum(target * logprobs, dim=1)

    if adjust:
        eps = 1e-8
        bias = target * torch.log(target + eps)
        if weight is not None:
            bias = bias * weight
        bias = torch.sum(bias, dim=1)
        batchloss += bias

    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        assert 0, f'Unsupported reduction mode {reduction}.'


def fake_quant(x: torch.Tensor, scale=128, zero_point=0, num_bits=8, signed=True):
    """Fake quantization while keep float gradient."""
    if num_bits is not None:
        if signed:
            qmin = -(2**(num_bits - 1))
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        x = torch.clamp(x, qmin / scale, qmax / scale)
    x_quant = (x.detach() * scale + zero_point).round()
    x_dequant = (x_quant - zero_point) / scale
    x = x - x.detach() + x_dequant  # stop gradient
    return x


def print_model_parameters(model):
    """打印模型各个层参数"""
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")
