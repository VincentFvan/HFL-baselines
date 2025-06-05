# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import random
import torch.nn.functional as F
import torch.nn.functional as func
import collections
from sklearn.model_selection import train_test_split
from collections import Counter
from utils.language_utils import word_to_indices, letter_to_vec # Assuming this path is correct relative to where the script runs
from utils.ShakeSpeare_reduce import ShakeSpeare # Assuming this path is correct

import math
import os # For creating directories and path joining
import json # For saving communication overhead data
import datetime # For timestamping output files

from models.lstm import CharLSTM # Assuming models.lstm resolves to lstm.py
from models.vgg import VGG16    # Assuming models.vgg resolves to vgg.py

# %% [markdown]
# v 19.1_250604_flops_non_invasive
# - Integrated FLOPs calculation into the training process without altering training logic.
# - Added FLOPs counter functions (adapted from main_flops_counter.py context).
# - `update_weights` and `update_weights_correction` now return calculated FLOPs.
# - All federated algorithms now track cumulative FLOPs.
# - `run_once` now plots Test Accuracy vs. Cumulative FLOPs.
# - FLOPs vs. Accuracy data is saved to a JSON file.
# - Ensured original parameters and training outcomes are preserved.

# %%
# import os # Already imported
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# %%
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %%
# Helper class to pass parameters to FLOPs counting functions
class ModelParamsForFlops:
    def __init__(self, dataset_name, batch_size_val, device_val, origin_model_name_val, num_classes_val,
                 # For LSTM:
                 model_vocab_size_val=None, model_seq_len_val=80, # Default seq_len for Shakespeare
                 # FLOPs counter specific flags (defaults for full model FLOPs)
                 learnable_val=False, 
                 mask_val=False 
                 ):
        self.dataset = dataset_name
        self.local_bs = batch_size_val 
        self.device = device_val
        self.origin_model = origin_model_name_val
        self.num_classes = num_classes_val

        self.learnable = learnable_val # Corresponds to args.learnable in original counter
        self.mask = mask_val # If False, 'full' becomes True in count_model_param_flops_adapted

        # For LSTM model (e.g. CharLSTM for Shakespeare)
        self.nvocab = model_vocab_size_val # Corresponds to model.args.nvocab
        self.seq_len = model_seq_len_val # Expected sequence length for LSTM dummy input


# %% FLOPs Calculation Utilities (Adapted from main_flops_counter.py context)
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

_list_conv_flops, _list_linear_flops, _list_bn_flops, _list_relu_flops = [], [], [], []
_list_pooling_flops, _list_upsample_flops, _list_lstm_flops, _list_embedding_flops = [], [], [], []

def _conv_hook(module, input, output, model_params_val, full=True, multiply_adds=True):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
    bias_ops = 1 if module.bias is not None else 0
    if not full: 
        num_weight_params = (module.weight.data != 0).float().sum()
    else: 
        num_weight_params = torch.numel(module.weight.data)
    flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size
    if model_params_val.learnable: 
        flops += module.weight.data.size(0)  
    _list_conv_flops.append(flops)

def _lstm_hook(module, input, output, model_params_val, full=True, multiply_adds=True):
    # input is a tuple, input[0] is the actual input tensor (seq_len, batch, input_size)
    # input[1] is hx (hidden state) if provided
    seq_len, batch_size, _ = input[0].shape 
    
    # Assuming single layer for hook simplicity, PyTorch nn.LSTM handles multi-layer internally.
    # Weights are weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    if not full:
        weight_hh_ops = (module.weight_hh_l0.data != 0).float().sum() * (2 if multiply_adds else 1)
        weight_ih_ops = (module.weight_ih_l0.data != 0).float().sum() * (2 if multiply_adds else 1)
        bias_hh_ops = (module.bias_hh_l0.data != 0).float().sum() if module.bias_hh_l0 is not None else 0
        bias_ih_ops = (module.bias_ih_l0.data != 0).float().sum() if module.bias_ih_l0 is not None else 0
    else:
        weight_hh_ops = torch.numel(module.weight_hh_l0.data) * (2 if multiply_adds else 1)
        weight_ih_ops = torch.numel(module.weight_ih_l0.data) * (2 if multiply_adds else 1)
        bias_hh_ops = torch.numel(module.bias_hh_l0.data) if module.bias_hh_l0 is not None else 0
        bias_ih_ops = torch.numel(module.bias_ih_l0.data) if module.bias_ih_l0 is not None else 0
    
    # FLOPs per (batch_element, sequence_element) for one layer:
    # Roughly 4 * (hidden_size * (input_size + hidden_size) + hidden_size_bias_term)
    # The context's original hook was: batch_size * (sum of weight_ops + sum of bias_ops)
    # This seems to be total ops for one time step across batch, needs * seq_len
    total_weight_matrix_ops = weight_ih_ops + weight_hh_ops
    total_bias_ops = bias_ih_ops + bias_hh_ops
    flops = (total_weight_matrix_ops + total_bias_ops) * batch_size * seq_len

    if model_params_val.learnable:
        flops = flops + (module.weight_hh_l0.data.size(0) + module.weight_ih_l0.data.size(0)) * seq_len
    _list_lstm_flops.append(flops)

def _embedding_hook(module, input, output, model_params_val, full=True, multiply_adds=True):
    # Embedding is a lookup, typically 0 FLOPs.
    # Context's calculation seems to be more about parameter access cost.
    flops = 0.0 
    if model_params_val.learnable:
        flops = flops + module.weight.data.size(0) 
    _list_embedding_flops.append(flops)

def _linear_hook(module, input, output, model_params_val, full=True, multiply_adds=True):
    batch_size = input[0].size(0)
    # For (seq, batch, feature) inputs, batch_size needs to be seq*batch
    if input[0].dim() > 2:
        batch_size = input[0].size(0) * input[0].size(1)

    # FLOPs = batch_size * (in_features * out_features * 2 + out_features_bias)
    flops = batch_size * (module.in_features * module.out_features * (2 if multiply_adds else 1) + \
             (module.out_features if module.bias is not None else 0))
    
    if model_params_val.learnable:
        flops = flops + module.weight.data.size(0) 
    _list_linear_flops.append(flops)

def _bn_hook(module, input, output):
    _list_bn_flops.append(input[0].nelement() * 2)

def _relu_hook(module, input, output):
    _list_relu_flops.append(input[0].nelement())

def _pooling_hook(module, input, output):
    batch_size, _, H_in, W_in = input[0].size()
    _, _, H_out, W_out = output[0].size()
    
    if isinstance(module.kernel_size, int):
        k_h, k_w = module.kernel_size, module.kernel_size
    else:
        k_h, k_w = module.kernel_size
        
    # Each output element requires k_h * k_w operations (comparisons for MaxPool, adds/div for AvgPool)
    ops_per_output_element = k_h * k_w
    flops = batch_size * output[0].size(1) * H_out * W_out * ops_per_output_element
    _list_pooling_flops.append(flops)
    
def _adaptive_pooling_hook(module, input, output):
    # For AdaptiveAvgPool2d, output size is fixed (e.g., 1x1).
    # FLOPs are roughly input_elements_pooled_per_output_element.
    # If output is (B, C, 1, 1), then input (B, C, H, W) means H*W ops per C per B.
    batch_size, channels, H_in, W_in = input[0].size()
    flops = batch_size * channels * H_in * W_in # Each input element is involved once for avg.
    _list_pooling_flops.append(flops)


def _upsample_hook(module, input, output):
    # Bilinear upsampling: context used 12 ops per output element.
    batch_size, channels, H_out, W_out = output[0].size()
    flops = batch_size * channels * H_out * W_out * 12 
    _list_upsample_flops.append(flops)

def _foo_register_hooks(handles, net, model_params_val):
    childrens = list(net.children())
    if not childrens: # Leaf module
        if isinstance(net, nn.Conv2d):
            handles.append(net.register_forward_hook(lambda m, i, o: _conv_hook(m, i, o, model_params_val, full=not model_params_val.mask)))
        elif isinstance(net, nn.Linear):
            handles.append(net.register_forward_hook(lambda m, i, o: _linear_hook(m, i, o, model_params_val, full=not model_params_val.mask)))
        elif isinstance(net, nn.LSTM):
            handles.append(net.register_forward_hook(lambda m, i, o: _lstm_hook(m, i, o, model_params_val, full=not model_params_val.mask)))
        elif isinstance(net, nn.Embedding):
            handles.append(net.register_forward_hook(lambda m, i, o: _embedding_hook(m, i, o, model_params_val, full=not model_params_val.mask)))
        elif isinstance(net, nn.BatchNorm2d):
            handles.append(net.register_forward_hook(_bn_hook))
        elif isinstance(net, (nn.ReLU, nn.ReLU6)):
            handles.append(net.register_forward_hook(_relu_hook))
        elif isinstance(net, (nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(net.register_forward_hook(_pooling_hook))
        elif isinstance(net, nn.AdaptiveAvgPool2d):
            handles.append(net.register_forward_hook(_adaptive_pooling_hook))
        elif isinstance(net, nn.Upsample): # Note: nn.Upsample is a class, mode='bilinear' makes it bilinear.
            handles.append(net.register_forward_hook(_upsample_hook))
        return
    for c in childrens:
        _foo_register_hooks(handles, c, model_params_val)

def count_model_param_flops_adapted(model_obj, model_params_val, multiply_adds=True):
    global _list_conv_flops, _list_linear_flops, _list_bn_flops, _list_relu_flops
    global _list_pooling_flops, _list_upsample_flops, _list_lstm_flops, _list_embedding_flops
    _list_conv_flops, _list_linear_flops, _list_bn_flops, _list_relu_flops = [], [], [], []
    _list_pooling_flops, _list_upsample_flops, _list_lstm_flops, _list_embedding_flops = [], [], [], []

    handles = []
    _foo_register_hooks(handles, model_obj, model_params_val)
    
    model_obj.eval() 
    current_model_device = next(model_obj.parameters()).device
    
    # Create dummy input based on dataset and model_params_val
    if model_params_val.dataset in ["cifar10", "cifar100"]:
        if model_params_val.origin_model in ['cnn', 'resnet', 'vgg', 'mobilenet']:
             input_tensor = torch.randn(model_params_val.local_bs, 3, 32, 32, device=current_model_device)
             _ = model_obj(input_tensor)
        else:
            raise ValueError(f"FLOPs: Unsupported model {model_params_val.origin_model} for {model_params_val.dataset}")
    elif model_params_val.dataset == "shake":
        if model_params_val.origin_model == 'lstm':
            # For CharLSTM, input is (seq_len, batch_size) of Long type (indices)
            seq_len = model_params_val.seq_len
            vocab_size = model_params_val.nvocab # Should be set from CharLSTM.vocab_size
            if vocab_size is None: vocab_size = 80 # Default if not available
            input_tensor = torch.randint(high=vocab_size, 
                                         size=(seq_len, model_params_val.local_bs), 
                                         device=current_model_device, dtype=torch.long)
            # LSTM model might initialize hidden state if not provided
            _ = model_obj(input_tensor) 
        else:
            raise ValueError(f"FLOPs: Unsupported model {model_params_val.origin_model} for {model_params_val.dataset}")
    else:
        raise ValueError(f"FLOPs calculation not implemented for dataset: {model_params_val.dataset}")

    total_flops = (sum(_list_conv_flops) + sum(_list_linear_flops) + sum(_list_bn_flops) + 
                   sum(_list_relu_flops) + sum(_list_pooling_flops) + sum(_list_upsample_flops) + 
                   sum(_list_lstm_flops) + sum(_list_embedding_flops))
    
    for handle in handles:
        handle.remove()
    return total_flops

def count_training_flops_adapted(model_obj, model_params_val):
    forward_flops = count_model_param_flops_adapted(model_obj, model_params_val, multiply_adds=True)
    return 3 * forward_flops # Standard heuristic: 1x fwd, 2x bwd

# %% End FLOPs Calculation Utilities


def get_object_size_in_bytes(obj_dict):
    """Calculates the total size of a dictionary of tensors in bytes."""
    if not isinstance(obj_dict, dict):
        return 0
    total_size = 0
    for key, value in obj_dict.items():
        if torch.is_tensor(value):
            total_size += value.nelement() * value.element_size()
    return total_size

# %%

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100): 
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual

class MobileNetV2(nn.Module):
    def __init__(self, num_classes_arg=20): 
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1), # Original MobileNetV2 uses kernel_size=3, stride=2 for first conv
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6) # Stride was 2 in some versions
        self.stage6 = self._make_stage(3, 96, 160, 1, 6) # Stride was 2, then 1
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)
        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Conv2d(1280, num_classes_arg, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x_avg_pool = F.adaptive_avg_pool2d(x, 1)
        x_conv2 = self.conv2(x_avg_pool)
        output = x_conv2.view(x_conv2.size(0), -1)
        return {'output': output, 'representation': x_avg_pool.view(x_avg_pool.size(0),-1)}

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)

def mobilenetv2(num_classes_arg=20): 
    return MobileNetV2(num_classes_arg=num_classes_arg)


class CNNCifar(nn.Module):
    def __init__(self, num_classes_arg): 
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes_arg) 

    def forward(self, x, start_layer_idx=0, logit=False): # start_layer_idx and logit not used by main script
        act1 = self.pool(F.relu(self.conv1(x)))
        act2 = self.pool(F.relu(self.conv2(act1)))
        result = {'activation1' : act1, 'activation2': act2}
        x_flat = act2.view(-1, 16 * 5 * 5)
        result['hint'] = x_flat 
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc2 = F.relu(self.fc2(x_fc1))
        result['representation'] = x_fc2 
        output = self.fc3(x_fc2)
        result['output'] = output
        return result
    
def cnncifar(num_classes_arg): 
    return CNNCifar(num_classes_arg=num_classes_arg)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width); self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation); self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion); self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class ResNetCifar10(nn.Module):
    def __init__(self, block, layers, num_classes_arg, zero_init_residual=False, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer; self.inplanes = 64; self.dilation = 1
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups; self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes); self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes_arg) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck): nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock): nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer; downsample = None; previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); result = {}
        x = self.layer1(x); result['activation1'] = x
        x = self.layer2(x); result['activation2'] = x
        x = self.layer3(x); result['activation3'] = x
        x = self.layer4(x); result['activation4'] = x
        x_avgpool = self.avgpool(x); x_flatten = torch.flatten(x_avgpool, 1)
        result['representation'] = x_flatten; output = self.fc(x_flatten)
        result['output'] = output
        return result

    def forward(self, x, start_layer_idx=0, logit=False): 
        return self._forward_impl(x)

def ResNet18_cifar10(num_classes_arg, **kwargs): 
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], num_classes_arg=num_classes_arg, **kwargs)

def ResNet50_cifar10(num_classes_arg, **kwargs): 
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], num_classes_arg=num_classes_arg, **kwargs)

# %%
def test_inference(net_glob, dataset_test_val): 
    net_glob = net_glob.to(device)
    acc_test, loss_test = test_img(net_glob, dataset_test_val)
    return acc_test.item()

def test_img(net_g, datatest):
    net_g.eval()
    test_loss = 0; correct = 0
    
    if not isinstance(datatest, Dataset):
        if isinstance(datatest[0], np.ndarray):
             images_tensor = torch.Tensor(datatest[0])
             labels_tensor = torch.Tensor(datatest[1]).long()
             datatest = TensorDataset(images_tensor, labels_tensor)
        elif torch.is_tensor(datatest[0]): 
             labels_tensor = datatest[1].long() if torch.is_tensor(datatest[1]) else torch.Tensor(datatest[1]).long()
             datatest = TensorDataset(datatest[0], labels_tensor)

    data_loader = DataLoader(datatest, batch_size=test_bc_size, shuffle=False)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            model_output = net_g(data)
            log_probs = model_output['output'] if isinstance(model_output, dict) and 'output' in model_output else model_output
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset) if len(data_loader.dataset) > 0 else 1
    accuracy = 100.00 * correct / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0.0
    if verbose: print(f'\nTest set: Average loss: {test_loss:.4f} \nAccuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy, test_loss

# %%
def sparse2coarse(targets):
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
    targets = np.array(targets).astype(int)
    return coarse_labels[targets]

# %%
def CIFAR100():
    trans_cifar100 = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    data_root = os.path.join(os.getcwd(), 'data', 'CIFAR-100'); os.makedirs(data_root, exist_ok=True)
    train_dataset = torchvision.datasets.CIFAR100(root=data_root, train=True, transform=trans_cifar100, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=data_root, train=False, transform=trans_cifar100, download=True)
    
    total_img,total_label = [],[]
    for imgs,labels in train_dataset: total_img.append(imgs.numpy()); total_label.append(labels)
    total_img = np.array(total_img); total_label = np.array(sparse2coarse(total_label))
    cifar_data_pool = (total_img, total_label) # Renamed for clarity
    
    if hasattr(test_dataset, 'targets'): test_dataset.targets = sparse2coarse(test_dataset.targets); test_dataset.targets = np.array(test_dataset.targets).astype(int)
    elif hasattr(test_dataset, 'labels'): test_dataset.labels = sparse2coarse(test_dataset.labels); test_dataset.labels = np.array(test_dataset.labels).astype(int)
    return cifar_data_pool, test_dataset


# %%
def get_prob(non_iid_strength, client_num_val, class_num_val=20, iid_mode=False): 
    if data_random_fix: np.random.seed(seed_num)
    if iid_mode: return np.ones((client_num_val, class_num_val)) / class_num_val
    else: alpha = max(0.01, non_iid_strength); return np.random.dirichlet(np.repeat(alpha, class_num_val), client_num_val)

# %%
def create_data_all_train(prob, size_per_client_val, dataset_pool_val, N_classes=20): 
    total_each_class = size_per_client_val * np.sum(prob, 0)
    data, label = dataset_pool_val # Use dataset_pool_val
    if data_random_fix: np.random.seed(seed_num); random.seed(seed_num)

    all_class_set = []
    for i in range(N_classes):
        size = total_each_class[i]; sub_data = data[label == i]; sub_label = label[label == i]
        num_samples = int(size)
        if num_samples > len(sub_data): num_samples = len(sub_data)
        if num_samples == 0: rand_indx = []
        elif len(sub_data) == 0: rand_indx = []
        else: rand_indx = np.random.choice(len(sub_data), size=num_samples, replace=False).astype(int)
        all_class_set.append((sub_data[rand_indx], sub_label[rand_indx]))

    index = [0] * N_classes; clients = []
    for m in range(prob.shape[0]):
        labels_list, images_list = [], []
        for n in range(N_classes):
            start, end = index[n], index[n] + int(prob[m][n] * size_per_client_val)
            image_samples, label_samples = all_class_set[n][0][start:end], all_class_set[n][1][start:end]
            index[n] += int(prob[m][n] * size_per_client_val)
            labels_list.extend(label_samples); images_list.extend(image_samples)
        clients.append((np.array(images_list), np.array(labels_list)))
    return clients

# %%
def select_server_subset(data_pool_val, percentage=0.1, mode='iid', dirichlet_alpha=1.0):
    images, labels = data_pool_val; unique_classes_arr = np.unique(labels); total_num = len(labels)
    server_total = int(total_num * percentage); selected_indices = []
    
    if mode == 'iid':
        for cls_val in unique_classes_arr:
            cls_indices = np.where(labels == cls_val)[0]
            num_cls = int(len(cls_indices) * percentage) if percentage < 1.0 else len(cls_indices)
            if num_cls > len(cls_indices): num_cls = len(cls_indices)
            if num_cls == 0 and len(cls_indices) > 0 and server_total > 0 : num_cls = 1 
            sampled = np.random.choice(cls_indices, size=num_cls, replace=False) if len(cls_indices) > 0 and num_cls > 0 else []
            selected_indices.extend(sampled)
    elif mode == 'non-iid':
        classes_len = len(unique_classes_arr); alpha_val = max(0.01, dirichlet_alpha)
        prob_dist = np.random.dirichlet(np.repeat(alpha_val, classes_len)) if classes_len > 0 else np.array([])
        cls_sample_numbers = {}; total_assigned = 0
        for i, cls_val in enumerate(unique_classes_arr):
            n_cls = int(prob_dist[i] * server_total) if len(prob_dist) > i else 0
            cls_sample_numbers[cls_val] = n_cls; total_assigned += n_cls
        diff = server_total - total_assigned
        if diff > 0 and len(unique_classes_arr) > 0:
            for cls_val_choice in np.random.choice(unique_classes_arr, size=diff, replace=True): cls_sample_numbers[cls_val_choice] += 1
        for cls_val in unique_classes_arr:
            cls_indices = np.where(labels == cls_val)[0]; n_sample = cls_sample_numbers.get(cls_val, 0)
            if n_sample > len(cls_indices): n_sample = len(cls_indices)
            sampled = np.random.choice(cls_indices, size=n_sample, replace=False) if len(cls_indices) > 0 and n_sample > 0 else []
            selected_indices.extend(sampled)
    else: raise ValueError("mode 参数必须为 'iid' 或 'non-iid'")
    
    selected_indices = list(set(selected_indices)) 
    if server_fill and len(selected_indices) < server_total :
        shortfall = server_total - len(selected_indices)
        if shortfall > 0:
            remaining_pool = np.setdiff1d(np.arange(total_num), selected_indices, assume_unique=True)
            if shortfall > len(remaining_pool): shortfall = len(remaining_pool) 
            extra = np.random.choice(remaining_pool, shortfall, replace=False) if len(remaining_pool) > 0 and shortfall > 0 else []
            selected_indices = np.concatenate([selected_indices, extra]) if len(extra) > 0 else np.array(selected_indices)
            
    selected_indices = np.array(selected_indices, dtype=int) 
    if len(selected_indices) > 0: np.random.shuffle(selected_indices) 
    if len(selected_indices) > server_total: selected_indices = selected_indices[:server_total]
    return images[selected_indices] if len(selected_indices) > 0 else np.array([]), labels[selected_indices] if len(selected_indices) > 0 else np.array([])

# %%
def update_weights(model_weight_input, dataset_val, learning_rate, local_epoch): 
    # Instantiate model for training
    if origin_model == 'resnet': model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm": model = CharLSTM().to(device) # Assumes CharLSTM has vocab_size, etc.
    elif origin_model == "cnn": model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
    
    model.load_state_dict(model_weight_input) # Load weights for training
    model.train() # Set to train mode
    
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    first_iter_gradient = {} 

    if len(dataset_val[0]) == 0: # No data
        return model.state_dict(), 0.0, first_iter_gradient, 0.0 # Return 0 FLOPs

    # Prepare DataLoader for training
    if origin_model in ['resnet', 'cnn', 'vgg', 'mobilenet']:
        Tensor_set = TensorDataset(torch.Tensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device))
    elif origin_model == 'lstm':
        # Process sequences for LSTM: list of numpy arrays -> stacked LongTensor
        processed_data = [torch.LongTensor(s_item) for s_item in dataset_val[0]] # s_item is a sequence
        stacked_data = torch.stack(processed_data) # Shape: (num_samples, seq_len)
        Tensor_set = TensorDataset(stacked_data.to(device), torch.Tensor(dataset_val[1]).long().to(device))
    data_loader = DataLoader(Tensor_set, batch_size=bc_size, shuffle=True)

    # --- FLOPs Calculation (Non-invasive) ---
    total_flops_this_update = 0.0
    if len(data_loader) > 0:
        # Create a temporary model instance for FLOPs calculation to avoid state interference
        temp_model_for_flops = None
        if origin_model == 'resnet': temp_model_for_flops = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
        elif origin_model == "lstm": temp_model_for_flops = CharLSTM().to(device)
        elif origin_model == "cnn": temp_model_for_flops = cnncifar(num_classes_arg=num_classes).to(device)
        elif origin_model == 'vgg': temp_model_for_flops = VGG16(num_classes, 3).to(device)
        elif origin_model == 'mobilenet': temp_model_for_flops = mobilenetv2(num_classes_arg=num_classes).to(device)
        
        if temp_model_for_flops:
            temp_model_for_flops.load_state_dict(model_weight_input) # Use weights at start of update
            
            model_params_flops = ModelParamsForFlops(
                dataset_name=dataset, batch_size_val=bc_size, device_val=device,
                origin_model_name_val=origin_model, num_classes_val=num_classes
            )
            if origin_model == "lstm": # Specific params for LSTM
                model_params_flops.nvocab = temp_model_for_flops.vocab_size if hasattr(temp_model_for_flops, 'vocab_size') else 80
                model_params_flops.seq_len = 80 # Default for Shakespeare from CharLSTM
            
            flops_per_batch_train = count_training_flops_adapted(temp_model_for_flops, model_params_flops)
            total_flops_this_update = local_epoch * len(data_loader) * flops_per_batch_train
            del temp_model_for_flops # Clean up
    # --- End FLOPs Calculation ---

    # --- Actual Training Loop (Original Logic) ---
    for iter_val in range(local_epoch): 
        batch_loss = []
        if not data_loader or len(data_loader.dataset) == 0 : 
            epoch_loss.append(0.0); continue
        for batch_idx, (images, labels) in enumerate(data_loader):
            if origin_model == 'lstm' and images.dim() == 2 and images.size(1) == (model_params_flops.seq_len if origin_model == 'lstm' else 0) : # (batch, seq_len)
                images = images.permute(1,0) # LSTM expects (seq_len, batch)

            model.zero_grad()
            outputs_dict = model(images)
            loss = criterion(outputs_dict['output'], labels) 
            loss.backward()
            
            if iter_val == 0 and batch_idx == 0: # Collect first iter gradient (original logic)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        first_iter_gradient[name] = param.grad.clone().cpu() 
                for name, module_val in model.named_modules(): 
                    if isinstance(module_val, nn.BatchNorm2d):
                        if hasattr(module_val, 'running_mean') and module_val.running_mean is not None:
                             first_iter_gradient[name + '.running_mean'] = module_val.running_mean.clone().cpu()
                        if hasattr(module_val, 'running_var') and module_val.running_var is not None:
                             first_iter_gradient[name + '.running_var'] = module_val.running_var.clone().cpu()
            optimizer.step()
            batch_loss.append(loss.item()/images.shape[0] if images.shape[0] > 0 else 0.0)
        epoch_loss.append(sum(batch_loss)/len(batch_loss) if len(batch_loss) > 0 else 0.0)
    # --- End Actual Training Loop ---
    
    final_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
    return model.state_dict(), final_loss, first_iter_gradient, total_flops_this_update


# %%
def weight_differences(n_w, p_w, lr_val): 
    w_diff = copy.deepcopy(n_w)
    target_device = n_w[list(n_w.keys())[0]].device if n_w else torch.device("cpu") # Get target device

    for key in w_diff.keys():
        if 'num_batches_tracked' in key: continue
        if key in p_w: 
             w_diff[key] = (p_w[key].to(target_device) - n_w[key].to(target_device)) * lr_val
    return w_diff

# %%
def update_weights_correction(model_weight_input, dataset_val, learning_rate, local_epoch, c_i, c_s): 
    if origin_model == 'resnet': model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm": model = CharLSTM().to(device)
    elif origin_model == "cnn": model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    model.load_state_dict(model_weight_input)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if len(dataset_val[0]) == 0:
        return model.state_dict(), 0.0, None, 0.0 # Return None for gradient placeholder, 0 FLOPs

    if origin_model in ['resnet', 'cnn', 'vgg', 'mobilenet']:
        Tensor_set = TensorDataset(torch.Tensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device))
    elif origin_model == 'lstm':
        processed_data = [torch.LongTensor(s_item) for s_item in dataset_val[0]]
        stacked_data = torch.stack(processed_data)
        Tensor_set = TensorDataset(stacked_data.to(device), torch.Tensor(dataset_val[1]).long().to(device))
    data_loader = DataLoader(Tensor_set, batch_size=bc_size, shuffle=True)

    # --- FLOPs Calculation (Non-invasive) ---
    total_flops_this_update = 0.0
    if len(data_loader) > 0:
        temp_model_for_flops = None
        if origin_model == 'resnet': temp_model_for_flops = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
        elif origin_model == "lstm": temp_model_for_flops = CharLSTM().to(device)
        elif origin_model == "cnn": temp_model_for_flops = cnncifar(num_classes_arg=num_classes).to(device)
        elif origin_model == 'vgg': temp_model_for_flops = VGG16(num_classes, 3).to(device)
        elif origin_model == 'mobilenet': temp_model_for_flops = mobilenetv2(num_classes_arg=num_classes).to(device)

        if temp_model_for_flops:
            temp_model_for_flops.load_state_dict(model_weight_input)
            model_params_flops = ModelParamsForFlops(
                dataset_name=dataset, batch_size_val=bc_size, device_val=device,
                origin_model_name_val=origin_model, num_classes_val=num_classes
            )
            if origin_model == "lstm":
                model_params_flops.nvocab = temp_model_for_flops.vocab_size if hasattr(temp_model_for_flops, 'vocab_size') else 80
                model_params_flops.seq_len = 80
            flops_per_batch_train = count_training_flops_adapted(temp_model_for_flops, model_params_flops)
            total_flops_this_update = local_epoch * len(data_loader) * flops_per_batch_train
            del temp_model_for_flops
    # --- End FLOPs Calculation ---

    # --- Actual Training Loop (Original Logic) ---
    for iter_val in range(local_epoch): 
        batch_loss = []
        if not data_loader or len(data_loader.dataset) == 0: epoch_loss.append(0.0); continue
        for batch_idx, (images, labels) in enumerate(data_loader):
            if origin_model == 'lstm' and images.dim() == 2 and images.size(1) == (model_params_flops.seq_len if origin_model == 'lstm' else 0):
                images = images.permute(1,0)
            model.zero_grad()
            outputs_dict = model(images)
            loss = criterion(outputs_dict['output'], labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/images.shape[0] if images.shape[0] > 0 else 0.0)
        epoch_loss.append(sum(batch_loss)/len(batch_loss) if len(batch_loss) > 0 else 0.0)
        
        # Correction logic (original)
        if c_i and c_s and len(c_i)>0 and len(c_s)>0: 
            c_i_dev = {k: v.to(device) for k, v in c_i.items() if torch.is_tensor(v)}
            c_s_dev = {k: v.to(device) for k, v in c_s.items() if torch.is_tensor(v)}
            if c_i_dev and c_s_dev:
                corrected_gradient_term = weight_differences(c_i_dev, c_s_dev, learning_rate) 
                current_model_state_dict = model.state_dict()
                corrected_model_weight = weight_differences(corrected_gradient_term, current_model_state_dict, 1)  
                model.load_state_dict(corrected_model_weight)
        elif not c_i and not c_s: pass
    # --- End Actual Training Loop ---

    final_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
    return model.state_dict(), final_loss, None, total_flops_this_update


# %%
def average_weights(w_list): 
    if not w_list: return {}
    valid_w_list = [w for w in w_list if w and isinstance(w, dict) and len(w) > 0]
    if not valid_w_list: return {}

    w_avg = copy.deepcopy(valid_w_list[0])
    target_device = w_avg[list(w_avg.keys())[0]].device # Get device from first tensor

    for key in w_avg.keys():
        if 'num_batches_tracked' in key: 
            if len(valid_w_list) > 0 and key in valid_w_list[0]: w_avg[key] = valid_w_list[0][key].clone()
            continue
        current_sum = valid_w_list[0][key].clone().to(target_device).float()
        for i in range(1, len(valid_w_list)):
            if key in valid_w_list[i]: current_sum += valid_w_list[i][key].to(target_device).float()
        w_avg[key] = torch.div(current_sum, len(valid_w_list)).type(valid_w_list[0][key].type())
    return w_avg

# %%
def server_only(initial_w, global_round_val, gamma_val, E_val): 
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0

    for round_idx in tqdm(range(global_round_val), desc="Server-Only"): 
        update_server_w, round_loss_val, _, round_flops = update_weights(train_w, server_data, gamma_val, E_val) 
        train_w = update_server_w; test_model.load_state_dict(train_w)
        train_loss_list.append(round_loss_val)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        
        cumulative_overhead += 0 # No communication
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

# %%
def fedavg(initial_w, global_round_val, eta_val, K_val, M_val): 
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w); model_size_bytes = get_object_size_in_bytes(train_w) 
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    
    for round_idx in tqdm(range(global_round_val), desc="FedAvg"):
        local_weights, local_loss_vals = [], []
        sampled_client_indices = random.sample(range(client_num), M_val) 
        active_clients_this_round, round_total_flops = 0, 0.0

        for client_idx in sampled_client_indices: 
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _, client_flops = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w); local_loss_vals.append(client_round_loss)
            round_total_flops += client_flops

        loss_avg = train_loss_list[-1] if not local_weights and train_loss_list else (sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else 0.0)
        if local_weights: train_w = average_weights(local_weights)
        
        train_loss_list.append(loss_avg); test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)

        cumulative_overhead += active_clients_this_round * (model_size_bytes + model_size_bytes) 
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
            
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

# %%
def hybridFL(initial_w, global_round_val, eta_val, K_val, E_val, M_val):
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations as in fedavg) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w); model_size_bytes = get_object_size_in_bytes(train_w)
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    
    for round_idx in tqdm(range(global_round_val), desc="HybridFL"):
        local_weights, local_loss_vals = [], []
        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round, round_total_flops = 0, 0.0

        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _, client_flops = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w); local_loss_vals.append(client_round_loss)
            round_total_flops += client_flops

        server_flops = 0.0
        if len(server_data[0]) > 0:
            update_server_w, server_round_loss, _, server_flops_val = update_weights(train_w, server_data, eta_val, E_val) 
            local_weights.append(update_server_w); local_loss_vals.append(server_round_loss)
            server_flops = server_flops_val
        round_total_flops += server_flops
        
        loss_avg = train_loss_list[-1] if not local_weights and train_loss_list else (sum(local_loss_vals) / len(local_loss_vals) if local_loss_vals else 0.0)
        if local_weights: train_w = average_weights(local_weights)

        train_loss_list.append(loss_avg); test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
    
        cumulative_overhead += active_clients_this_round * (model_size_bytes + model_size_bytes)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

def Data_Sharing(initial_w, global_round_val, eta_val, K_val, M_val, share_ratio=1):
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise ValueError("Unknown origin_model")

    local_datasets = client_data_mixed 
    w_global = copy.deepcopy(initial_w); model_size_bytes = get_object_size_in_bytes(w_global)
    all_test_acc_list, all_train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0

    for r_idx in tqdm(range(global_round_val), desc="DataSharing"): 
        selected_indices = np.random.choice(range(client_num), M_val, replace=False) 
        local_ws, local_ls = [], []
        active_clients_this_round, round_total_flops = 0, 0.0

        for cid in selected_indices:
            if len(local_datasets[cid][0]) == 0: continue
            active_clients_this_round +=1
            w_updated, loss_val, _, client_flops = update_weights(w_global, local_datasets[cid], eta_val, K_val) 
            local_ws.append(w_updated); local_ls.append(loss_val)
            round_total_flops += client_flops

        avg_loss = all_train_loss_list[-1] if not local_ws and all_train_loss_list else (sum(local_ls) / len(local_ls) if local_ls else 0.0)
        if local_ws: w_global = average_weights(local_ws)
        
        all_train_loss_list.append(avg_loss); test_model.load_state_dict(w_global)
        current_test_acc = test_inference(test_model, test_dataset); all_test_acc_list.append(current_test_acc)

        cumulative_overhead += active_clients_this_round * (model_size_bytes + model_size_bytes)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})

    return all_test_acc_list, all_train_loss_list, comm_vs_acc_list, flops_vs_acc_list


def build_mixed_client_data(client_data_val, server_data_val, share_ratio=1.0, seed_val=None): 
    if seed_val is not None: np.random.seed(seed_val)
    s_imgs, s_lbls = server_data_val
    s_imgs_arr = np.array(s_imgs) if len(s_imgs) > 0 else np.array([])
    s_lbls_arr = np.array(s_lbls) if len(s_lbls) > 0 else np.array([])

    if share_ratio < 1.0 and len(s_imgs_arr) > 0 :
        sel_idx = np.random.choice(len(s_imgs_arr), size=int(len(s_imgs_arr) * share_ratio), replace=False).astype(int)
        s_imgs_arr, s_lbls_arr = s_imgs_arr[sel_idx], s_lbls_arr[sel_idx]

    mixed_clients = []
    for imgs_c, lbls_c in client_data_val: 
        imgs_c_arr = np.array(imgs_c) if len(imgs_c) > 0 else np.array([])
        lbls_c_arr = np.array(lbls_c) if len(lbls_c) > 0 else np.array([])
        if len(s_imgs_arr) > 0: 
            new_imgs = np.concatenate([imgs_c_arr, s_imgs_arr], axis=0) if len(imgs_c_arr) > 0 else s_imgs_arr
            new_lbls = np.concatenate([lbls_c_arr, s_lbls_arr], axis=0) if len(lbls_c_arr) > 0 else s_lbls_arr
        else: new_imgs, new_lbls = imgs_c_arr, lbls_c_arr
        mixed_clients.append((new_imgs, new_lbls))
    return mixed_clients

# %%
def CLG_SGD(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    
    for round_idx in tqdm(range(global_round_val), desc="CLG-SGD"):
        local_weights, local_loss_vals = [], []
        model_size_for_client_download = get_object_size_in_bytes(train_w) 
        round_total_flops, active_clients_this_round = 0.0, 0

        sampled_client_indices = random.sample(range(client_num), M_val)
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _, client_flops = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w); local_loss_vals.append(client_round_loss)
            round_total_flops += client_flops
        
        cumulative_overhead += active_clients_this_round * (model_size_for_client_download + model_size_for_client_download)
        if local_weights: train_w = average_weights(local_weights)
        
        server_flops = 0.0
        if len(server_data[0]) > 0: 
            update_server_w, server_loss, _, server_flops_val = update_weights(train_w, server_data, gamma_val, E_val) 
            train_w = update_server_w; local_loss_vals.append(server_loss) 
            server_flops = server_flops_val
        round_total_flops += server_flops
        
        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg); test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

def Fed_C(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0

    for round_idx in tqdm(range(global_round_val), desc="Fed-C"):
        local_weights, local_loss_vals, g_i_list = [], [], []
        model_size_bytes = get_object_size_in_bytes(train_w); gs_size_bytes, g_s = 0, {}
        round_total_flops, server_grad_flops = 0.0, 0.0

        if len(server_data[0]) > 0:
             _, _, g_s, server_grad_flops_val = update_weights(train_w, server_data, gamma_val, 1) 
             gs_size_bytes = get_object_size_in_bytes(g_s); server_grad_flops = server_grad_flops_val
        round_total_flops += server_grad_flops

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round, client_grad_flops_total = 0, 0.0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: g_i_list.append({}); continue
            active_clients_this_round +=1
            _, _, g_i, client_grad_flops = update_weights(train_w, client_data[client_idx], eta_val, 1) 
            g_i_list.append(g_i if g_i else {}); client_grad_flops_total += client_grad_flops
        round_total_flops += client_grad_flops_total

        client_iter, client_update_flops_total = 0, 0.0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: client_iter +=1; continue
            current_g_i = g_i_list[client_iter] if client_iter < len(g_i_list) and g_i_list else {}
            client_iter +=1
            update_client_w, client_round_loss, _, client_update_flops = update_weights_correction(
                train_w, client_data[client_idx], eta_val, K_val, current_g_i, g_s)
            local_weights.append(update_client_w); local_loss_vals.append(client_round_loss)
            client_update_flops_total += client_update_flops
        round_total_flops += client_update_flops_total
        
        cumulative_overhead += active_clients_this_round * (model_size_bytes + gs_size_bytes + model_size_bytes)
        if local_weights: train_w = average_weights(local_weights)
        
        server_train_flops = 0.0
        if len(server_data[0]) > 0: 
            update_server_w, server_loss, _, server_train_flops_val = update_weights(train_w, server_data, gamma_val, E_val)
            train_w = update_server_w; local_loss_vals.append(server_loss)
            server_train_flops = server_train_flops_val
        round_total_flops += server_train_flops

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg); test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list


def Fed_S(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
    
    train_w = copy.deepcopy(initial_w)
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0

    for round_idx in tqdm(range(global_round_val), desc="Fed-S"):
        local_weights, local_loss_vals, g_i_list_for_s_correction = [], [], []
        model_size_bytes = get_object_size_in_bytes(train_w); gi_size_bytes = 0 
        round_total_flops, server_grad_flops = 0.0, 0.0
        g_s = {}
        if len(server_data[0]) > 0:
            _, _, g_s, server_grad_flops_val = update_weights(train_w, server_data, gamma_val, 1)
            server_grad_flops = server_grad_flops_val
        round_total_flops += server_grad_flops

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round, client_update_flops_total, client_grad_flops_total = 0, 0.0, 0.0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _, client_update_flops = update_weights(train_w, client_data[client_idx], eta_val, K_val) 
            local_weights.append(update_client_w); local_loss_vals.append(client_round_loss)
            client_update_flops_total += client_update_flops
            _, _, g_i_for_correction, client_grad_flops = update_weights(train_w, client_data[client_idx], eta_val, 1) 
            if g_i_for_correction: 
                 g_i_list_for_s_correction.append(g_i_for_correction)
                 if gi_size_bytes == 0: gi_size_bytes = get_object_size_in_bytes(g_i_for_correction)
            client_grad_flops_total += client_grad_flops
        round_total_flops += client_update_flops_total + client_grad_flops_total
        
        cumulative_overhead += active_clients_this_round * (model_size_bytes + model_size_bytes + gi_size_bytes)
        train_w_aggregated = average_weights(local_weights) if local_weights else copy.deepcopy(train_w) 

        if g_i_list_for_s_correction and g_s and len(g_s)>0: 
            g_i_list_dev = [{k: v.to(device) for k,v in g.items() if torch.is_tensor(v)} for g in g_i_list_for_s_correction if g]
            g_s_dev = {k: v.to(device) for k,v in g_s.items() if torch.is_tensor(v)}
            if g_i_list_dev and g_s_dev:
                g_i_average = average_weights([g for g in g_i_list_dev if g]) 
                if g_i_average and len(g_i_average)>0: 
                    correction_g_term = weight_differences(g_i_average, g_s_dev, K_val * eta_val) 
                    train_w = weight_differences(correction_g_term, copy.deepcopy(train_w_aggregated), 1) 
                else: train_w = train_w_aggregated
            else: train_w = train_w_aggregated
        else: train_w = train_w_aggregated

        server_train_flops = 0.0
        if len(server_data[0]) > 0: 
            update_server_w, server_loss, _, server_train_flops_val = update_weights(train_w, server_data, gamma_val, E_val)
            train_w = update_server_w; local_loss_vals.append(server_loss)
            server_train_flops = server_train_flops_val
        round_total_flops += server_train_flops

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg); test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

# %%
def KL_divergence(p1, p2): # p1, p2 are numpy arrays (distributions)
    p1_arr = np.array(p1); p2_arr = np.array(p2)
    valid_mask = (p1_arr > 1e-9) & (p2_arr > 1e-9) # Both positive
    if not np.any(valid_mask): return 0.0
    ratio = p1_arr[valid_mask] / p2_arr[valid_mask]
    return np.sum(p1_arr[valid_mask] * np.log2(ratio))

def calculate_js_divergence(p1, p2):
    p1 = np.array(p1); p2 = np.array(p2)
    if not np.isclose(np.sum(p1), 1.0) and np.sum(p1) > 0: p1 = p1 / np.sum(p1)
    if not np.isclose(np.sum(p2), 1.0) and np.sum(p2) > 0: p2 = p2 / np.sum(p2)
    m = 0.5 * (p1 + p2)
    return 0.5 * (KL_divergence(p1, m) + KL_divergence(p2, m))

def ratio_combine(w1, w2, ratio=0):
    if not w1: return copy.deepcopy(w2) 
    if not w2: return copy.deepcopy(w1) 
    dev = w1[list(w1.keys())[0]].device
    w = copy.deepcopy(w1)
    for key in w.keys():
        if 'num_batches_tracked' in key: continue
        if key in w2: w[key] = (w2[key].to(dev) - w1[key].to(dev)) * ratio + w1[key].to(dev)
    return w

def FedDU_modify(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w); test_model.load_state_dict(train_w)
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    
    all_client_labels_list = [lbl for i in range(client_num) for lbl in client_data[i][1]]
    all_client_labels_arr = np.array(all_client_labels_list) 
    P_dist = np.zeros(num_classes)
    if len(all_client_labels_arr) > 0:
        unique_cls, counts = np.unique(all_client_labels_arr, return_counts=True)
        for cls_val, count_val in zip(unique_cls, counts):
            if 0 <= cls_val < num_classes: P_dist[cls_val] = count_val / len(all_client_labels_arr)
    
    server_labels_arr = np.array(server_data[1]); n_0 = len(server_labels_arr)
    P_0_dist = np.zeros(num_classes)
    if n_0 > 0:
        unique_server_cls, server_cls_counts = np.unique(server_labels_arr, return_counts=True)
        for cls_val, count_val in zip(unique_server_cls, server_cls_counts):
             if 0 <= cls_val < num_classes: P_0_dist[cls_val] = count_val / n_0
    D_P_0 = calculate_js_divergence(P_0_dist, P_dist)
    
    for round_idx in tqdm(range(global_round_val), desc="FedDU"):
        local_weights, local_loss_vals_iter = [], [] 
        round_total_flops, model_size_bytes_download = 0.0, get_object_size_in_bytes(train_w) 
        sampled_clients_indices = random.sample(range(client_num), M_val) 
        num_current_samples, active_clients_this_round, client_update_flops_total = 0, 0, 0.0

        for client_idx in sampled_clients_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1; num_current_samples += len(client_data[client_idx][0])
            update_client_w, client_round_loss, _, client_flops = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w); local_loss_vals_iter.append(client_round_loss)
            client_update_flops_total += client_flops
        round_total_flops += client_update_flops_total
        cumulative_overhead += active_clients_this_round * (model_size_bytes_download + model_size_bytes_download)
        
        w_t_half = average_weights(local_weights) if local_weights else copy.deepcopy(train_w) 
        
        selected_client_labels_list = [lbl for idx in sampled_clients_indices for lbl in client_data[idx][1]]
        selected_client_labels_arr = np.array(selected_client_labels_list) 
        P_t_prime_dist = np.zeros(num_classes)
        if len(selected_client_labels_arr) > 0:
            unique_selected_cls, selected_cls_counts = np.unique(selected_client_labels_arr, return_counts=True)
            for cls_val, count_val in zip(unique_selected_cls, selected_cls_counts):
                if 0 <= cls_val < num_classes: P_t_prime_dist[cls_val] = count_val / len(selected_client_labels_arr)
        D_P_t_prime = calculate_js_divergence(P_t_prime_dist, P_dist)
        
        test_model.load_state_dict(w_t_half); acc_t = test_inference(test_model, test_dataset) / 100.0
        epsilon = 1e-10
        denominator_alpha = (n_0 * D_P_t_prime + num_current_samples * D_P_0 + epsilon)
        alpha_dyn = (1 - acc_t) * (n_0 * D_P_t_prime) / denominator_alpha if denominator_alpha != 0 else 0
        alpha_dyn = alpha_dyn * (decay_rate ** round_idx) * du_C; alpha_dyn = max(0, min(1, alpha_dyn))
        
        server_update_flops, final_train_w = 0.0, w_t_half
        if alpha_dyn > 0.001 and n_0 > 0: 
            update_server_w, server_loss, _, server_flops_val = update_weights(copy.deepcopy(w_t_half), server_data, gamma_val, E_val) 
            local_loss_vals_iter.append(server_loss); server_update_flops = server_flops_val
            final_train_w = ratio_combine(w_t_half, update_server_w, alpha_dyn) 
        elif n_0 > 0: # Server data exists but alpha too small
            _, server_loss, _, server_flops_val = update_weights(copy.deepcopy(w_t_half), server_data, gamma_val, E_val) 
            local_loss_vals_iter.append(server_loss); server_update_flops = server_flops_val
        round_total_flops += server_update_flops; train_w = final_train_w

        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss_vals_iter) / len(local_loss_vals_iter) if local_loss_vals_iter else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

# %%
def Aggregation(w_list, lens_list): # lens_list is for weighted averaging, None for unweighted
    if not w_list: return {}
    valid_w_list = [w for w in w_list if w and isinstance(w, dict) and len(w) > 0]
    if not valid_w_list: return {}
    
    target_device = valid_w_list[0][list(valid_w_list[0].keys())[0]].device
    w_avg = OrderedDict() # Use OrderedDict to maintain key order

    if lens_list is None: # Unweighted average
        num_valid_models = float(len(valid_w_list))
        if num_valid_models == 0: return {}
        for k_key in valid_w_list[0].keys():
            if 'num_batches_tracked' in k_key: w_avg[k_key] = valid_w_list[0][k_key].clone(); continue
            sum_tensor = torch.zeros_like(valid_w_list[0][k_key], dtype=torch.float, device=target_device)
            for w_model in valid_w_list:
                if k_key in w_model: sum_tensor += w_model[k_key].to(target_device).float()
            w_avg[k_key] = (sum_tensor / num_valid_models).type(valid_w_list[0][k_key].type())
    else: # Weighted average
        valid_indices = [i for i, w in enumerate(w_list) if w and isinstance(w, dict) and len(w) > 0]
        lens_list_actual = [lens_list[i] for i in valid_indices]
        total_weight = float(sum(lens_list_actual))
        if total_weight == 0: return copy.deepcopy(valid_w_list[0]) if valid_w_list else {}
        for k_key in valid_w_list[0].keys():
            if 'num_batches_tracked' in k_key: w_avg[k_key] = valid_w_list[0][k_key].clone(); continue
            sum_tensor = torch.zeros_like(valid_w_list[0][k_key], dtype=torch.float, device=target_device)
            for i, w_model_idx in enumerate(valid_indices):
                w_model = valid_w_list[i] # w_model is valid_w_list[idx_in_valid_w_list]
                if k_key in w_model: sum_tensor += w_model[k_key].to(target_device).float() * (lens_list_actual[i] / total_weight)
            w_avg[k_key] = sum_tensor.type(valid_w_list[0][k_key].type())
    return w_avg


def FedSub(w_curr, w_prev, weight_val): 
    if not w_curr or not w_prev: return {} 
    dev = w_curr[list(w_curr.keys())[0]].device; w_sub = OrderedDict()
    for k_key in w_curr.keys():
        if 'num_batches_tracked' in k_key: w_sub[k_key] = w_curr[k_key].clone(); continue
        if k_key in w_prev: w_sub[k_key] = (w_curr[k_key].to(dev) - w_prev[k_key].to(dev)) * weight_val
    return w_sub

def delta_rank(delta_dict):
    if not delta_dict : return 0.0
    dict_a_list = [delta_dict[p_key].view(-1).float() for p_key in delta_dict.keys() if 'num_batches_tracked' not in p_key and torch.is_tensor(delta_dict[p_key])]
    if not dict_a_list: return 0.0
    return torch.norm(torch.cat(dict_a_list, dim=0), p=2, dim=0).item()

def mutation_spread(iter_val, w_glob_val, m_clients, w_delta_val, alpha_mut): 
    if not w_glob_val or not w_delta_val or not isinstance(w_delta_val, dict) or len(w_delta_val) == 0:
        return [copy.deepcopy(w_glob_val) for _ in range(m_clients)] if w_glob_val else [OrderedDict() for _ in range(m_clients)]
    dev = w_glob_val[list(w_glob_val.keys())[0]].device
    ctrl_rate_val = mut_acc_rate * (1.0 - min(iter_val * 1.0 / mut_bound if mut_bound > 0 else 1.0 , 1.0)) 
    ctrl_cmd_list_outer = []
    for k_key in w_glob_val.keys():
        if 'num_batches_tracked' in k_key : continue 
        ctrl_list_inner = [] 
        for _ in range(0, int(m_clients / 2)): 
            ctrl_rand = random.random() 
            ctrl_list_inner.extend([1.0, 1.0 * (-1.0 + ctrl_rate_val)] if ctrl_rand > 0.5 else [1.0 * (-1.0 + ctrl_rate_val), 1.0])
        if m_clients % 2 == 1: ctrl_list_inner.append(0.0) 
        random.shuffle(ctrl_list_inner); ctrl_cmd_list_outer.append(ctrl_list_inner)
    
    w_locals_new_list = [] 
    for j_client in range(m_clients): 
        w_sub_mutated = copy.deepcopy(w_glob_val) 
        if not (j_client == m_clients - 1 and m_clients % 2 == 1):
            for param_idx, k_key in enumerate(d_k for d_k in w_sub_mutated.keys() if 'num_batches_tracked' not in d_k):
                if param_idx < len(ctrl_cmd_list_outer) and j_client < len(ctrl_cmd_list_outer[param_idx]) and k_key in w_delta_val:
                     w_sub_mutated[k_key] = w_sub_mutated[k_key].to(dev) + w_delta_val[k_key].to(dev) * ctrl_cmd_list_outer[param_idx][j_client] * alpha_mut
        w_locals_new_list.append(w_sub_mutated)
    return w_locals_new_list

def FedMut(net_glob_model, global_round_val, eta_val, K_val, M_val): 
    net_glob_model.train()
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    w_locals_for_selected_clients = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)] 
    model_size_bytes = get_object_size_in_bytes(net_glob_model.state_dict()) 
    
    for round_idx in tqdm(range(global_round_val), desc="FedMut"):
        w_old_global_for_delta = copy.deepcopy(net_glob_model.state_dict()) 
        local_loss_vals, round_total_flops = [], 0.0
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False) 
        active_clients_this_round, temp_w_locals_for_agg = 0, []

        for i_local_idx, client_actual_idx in enumerate(idxs_users_sampled): 
            if len(client_data[client_actual_idx][0]) == 0: 
                temp_w_locals_for_agg.append(w_locals_for_selected_clients[i_local_idx]); continue 
            active_clients_this_round +=1
            current_client_model_state = w_locals_for_selected_clients[i_local_idx]
            updated_client_w, client_round_loss, _, client_flops = update_weights(
                current_client_model_state, client_data[client_actual_idx], eta_val, K_val)
            w_locals_for_selected_clients[i_local_idx] = copy.deepcopy(updated_client_w) 
            temp_w_locals_for_agg.append(updated_client_w)
            local_loss_vals.append(client_round_loss); round_total_flops += client_flops

        w_aggregated = Aggregation(temp_w_locals_for_agg, None) if temp_w_locals_for_agg else copy.deepcopy(w_old_global_for_delta)
        if not w_aggregated or len(w_aggregated) == 0: w_aggregated = copy.deepcopy(w_old_global_for_delta) 
        net_glob_model.load_state_dict(w_aggregated) 
        
        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg); test_model.load_state_dict(w_aggregated)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)

        cumulative_overhead += active_clients_this_round * (model_size_bytes + model_size_bytes) 
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})

        w_delta_mutation = FedSub(w_aggregated, w_old_global_for_delta, 1.0) 
        w_locals_for_selected_clients = mutation_spread(round_idx, w_aggregated, M_val, w_delta_mutation, radius)
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list   

# %%
def CLG_Mut_2(net_glob_model, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    net_glob_model.train()
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    w_locals_for_selected_clients = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)]
    model_size_bytes = get_object_size_in_bytes(net_glob_model.state_dict()) 

    for round_idx in tqdm(range(global_round_val), desc="CLG-Mut-2"):
        w_old_global_round_for_delta = copy.deepcopy(net_glob_model.state_dict()) 
        local_loss_vals, round_total_flops = [], 0.0
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False)
        active_clients_this_round, temp_w_locals_for_agg, client_update_flops_total = 0, [], 0.0

        for i_local_idx, client_actual_idx in enumerate(idxs_users_sampled):
            if len(client_data[client_actual_idx][0]) == 0: 
                temp_w_locals_for_agg.append(w_locals_for_selected_clients[i_local_idx]); continue
            active_clients_this_round +=1
            current_client_model_state = w_locals_for_selected_clients[i_local_idx] 
            updated_client_w, client_round_loss, _, client_flops = update_weights(
                current_client_model_state, client_data[client_actual_idx], eta_val, K_val)
            w_locals_for_selected_clients[i_local_idx] = copy.deepcopy(updated_client_w)
            temp_w_locals_for_agg.append(updated_client_w)
            local_loss_vals.append(client_round_loss); client_update_flops_total += client_flops
        round_total_flops += client_update_flops_total
        cumulative_overhead += active_clients_this_round * (model_size_bytes + model_size_bytes)

        w_aggregated_clients = Aggregation(temp_w_locals_for_agg, None) if temp_w_locals_for_agg else copy.deepcopy(w_old_global_round_for_delta)
        w_after_server_train, server_train_flops = w_aggregated_clients, 0.0
        if len(server_data[0]) > 0: 
            w_after_server_train, server_loss, _, server_flops_val = update_weights(w_aggregated_clients, server_data, gamma_val, E_val)
            local_loss_vals.append(server_loss); server_train_flops = server_flops_val
        round_total_flops += server_train_flops
        net_glob_model.load_state_dict(w_after_server_train) 

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg); test_model.load_state_dict(w_after_server_train)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})

        w_delta_mutation = FedSub(w_after_server_train, w_old_global_round_for_delta, 1.0)
        w_locals_for_selected_clients = mutation_spread(round_idx, w_after_server_train, M_val, w_delta_mutation, radius)
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

# %%
def FedATMV(net_glob_model, global_round_val, eta_val, gamma_val, K_val, E_val, M_val, lambda_val_fedatmv=1): 
    net_glob_model.train()
    if origin_model == 'resnet': test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    # ... (other model instantiations) ...
    elif origin_model == "lstm": test_model = CharLSTM().to(device)
    elif origin_model == "cnn": test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': test_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
    
    test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list = [], [], [], []
    cumulative_overhead, cumulative_flops = 0, 0.0
    w_locals_for_selected_clients = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)]
    personalized_model_size_bytes = get_object_size_in_bytes(net_glob_model.state_dict())

    all_client_labels_list = [lbl for i in range(client_num) for lbl in client_data[i][1]]
    all_client_labels_arr = np.array(all_client_labels_list)
    P_dist_fedatmv = np.zeros(num_classes)
    if len(all_client_labels_arr) > 0:
        unique_cls, counts = np.unique(all_client_labels_arr, return_counts=True)
        for cls_val, count_val in zip(unique_cls, counts):
            if 0 <= cls_val < num_classes: P_dist_fedatmv[cls_val] = count_val / len(all_client_labels_arr)
    
    server_labels_arr_fedatmv = np.array(server_data[1]); n_0_fedatmv = len(server_labels_arr_fedatmv)
    P_0_dist_fedatmv = np.zeros(num_classes)
    if n_0_fedatmv > 0:
        unique_server_cls, server_cls_counts = np.unique(server_labels_arr_fedatmv, return_counts=True)
        for cls_val, count_val in zip(unique_server_cls, server_cls_counts):
            if 0 <= cls_val < num_classes: P_0_dist_fedatmv[cls_val] = count_val / n_0_fedatmv
    D_P_0_fedatmv = calculate_js_divergence(P_0_dist_fedatmv, P_dist_fedatmv)
    acc_prev_fedatmv = 0.0 
    
    for round_idx in tqdm(range(global_round_val), desc="FedATMV"):
        w_old_global_round_for_delta = copy.deepcopy(net_glob_model.state_dict())
        local_loss_vals, round_total_flops = [], 0.0
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False)
        selected_client_labels_list_fedatmv, num_current_samples_fedatmv = [], 0
        active_clients_this_round, temp_w_locals_for_agg, client_update_flops_total = 0, [], 0.0

        for i_local_idx, client_actual_idx in enumerate(idxs_users_sampled):
            if len(client_data[client_actual_idx][0]) == 0: 
                temp_w_locals_for_agg.append(w_locals_for_selected_clients[i_local_idx]); continue
            active_clients_this_round +=1
            current_client_model_state = w_locals_for_selected_clients[i_local_idx]
            updated_client_w, client_round_loss, _, client_flops = update_weights(
                current_client_model_state, client_data[client_actual_idx], eta_val, K_val)
            w_locals_for_selected_clients[i_local_idx] = copy.deepcopy(updated_client_w)
            temp_w_locals_for_agg.append(updated_client_w); local_loss_vals.append(client_round_loss)
            selected_client_labels_list_fedatmv.extend(client_data[client_actual_idx][1])
            num_current_samples_fedatmv += len(client_data[client_actual_idx][0])
            client_update_flops_total += client_flops
        round_total_flops += client_update_flops_total
        cumulative_overhead += active_clients_this_round * (personalized_model_size_bytes + personalized_model_size_bytes)

        w_aggregated_clients = Aggregation(temp_w_locals_for_agg, None) if temp_w_locals_for_agg else copy.deepcopy(w_old_global_round_for_delta)
        P_t_prime_dist_fedatmv = np.zeros(num_classes)
        if num_current_samples_fedatmv > 0:
            sel_arr = np.array(selected_client_labels_list_fedatmv)
            if len(sel_arr)>0:
                unique_sel_cls, sel_counts = np.unique(sel_arr, return_counts=True)
                for cls_val, count_val in zip(unique_sel_cls, sel_counts):
                     if 0 <= cls_val < num_classes: P_t_prime_dist_fedatmv[cls_val] = count_val / len(sel_arr)
        D_P_t_prime_fedatmv = calculate_js_divergence(P_t_prime_dist_fedatmv, P_dist_fedatmv)
        
        test_model.load_state_dict(w_aggregated_clients); acc_t_fedatmv = test_inference(test_model, test_dataset) / 100.0
        epsilon_fedatmv = 1e-10 
        den_r_data = (n_0_fedatmv + num_current_samples_fedatmv + epsilon_fedatmv)
        r_data_fedatmv = n_0_fedatmv / den_r_data if den_r_data !=0 else 0
        den_r_noniid = (D_P_t_prime_fedatmv + D_P_0_fedatmv + epsilon_fedatmv)
        r_noniid_fedatmv = D_P_t_prime_fedatmv / den_r_noniid if den_r_noniid !=0 else 0
        improvement_fedatmv = max(0.0, acc_prev_fedatmv - acc_t_fedatmv) / (acc_prev_fedatmv + epsilon_fedatmv) if round_idx > 0 and (acc_prev_fedatmv + epsilon_fedatmv) !=0 else 0
        alpha_new_fedatmv = du_C * (1 - acc_t_fedatmv) * r_data_fedatmv * r_noniid_fedatmv + lambda_val_fedatmv * improvement_fedatmv
        alpha_new_fedatmv = max(0.001, min(1.0, alpha_new_fedatmv)); acc_prev_fedatmv = acc_t_fedatmv 

        final_model_state_after_server, server_train_flops = w_aggregated_clients, 0.0
        if alpha_new_fedatmv > 0.001 and n_0_fedatmv > 0:
            update_server_w, server_loss, _, server_flops_val = update_weights(copy.deepcopy(w_aggregated_clients), server_data, gamma_val, E_val)
            local_loss_vals.append(server_loss); server_train_flops = server_flops_val
            final_model_state_after_server = ratio_combine(w_aggregated_clients, update_server_w, alpha_new_fedatmv)
        elif n_0_fedatmv > 0: 
             _, server_loss, _, server_flops_val = update_weights(copy.deepcopy(w_aggregated_clients), server_data, gamma_val, E_val)
             local_loss_vals.append(server_loss); server_train_flops = server_flops_val
        round_total_flops += server_train_flops; net_glob_model.load_state_dict(final_model_state_after_server) 
        
        loss_avg = sum(local_loss_vals) / len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg); test_model.load_state_dict(final_model_state_after_server)
        current_test_acc = test_inference(test_model, test_dataset); test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        cumulative_flops += round_total_flops
        flops_vs_acc_list.append({'flops': cumulative_flops, 'accuracy': current_test_acc})
        
        w_delta_mutation = FedSub(final_model_state_after_server, w_old_global_round_for_delta, 1.0)
        tmp_radius_fedatmv = radius * (1 + scal_ratio * alpha_new_fedatmv) 
        w_locals_for_selected_clients = mutation_spread(round_idx, final_model_state_after_server, M_val, w_delta_mutation, tmp_radius_fedatmv)
    return test_acc_list, train_loss_list, comm_vs_acc_list, flops_vs_acc_list

# %%
# Global parameters (Original, Unchanged)
data_random_fix = False 
seed_num = 42
random_fix = True
seed = 2
GPU = 0 # User's original setting
verbose = False
client_num = 100
size_per_client = 400
is_iid = False
non_iid = 0.1
server_iid = False
server_dir = 0.1
server_percentage = 0.1
server_fill = True
origin_model = 'resnet' 
dataset = 'cifar10' 
momentum = 0.5
weight_decay = 0
bc_size = 50
test_bc_size = 128
num_classes = 10 
global_round = 2
eta = 0.01  
gamma = 0.01  
K = 5  
E = 1  
M = 10  
du_C = 5
radius = 4.0  
scal_ratio=0.3
mut_acc_rate = 0.5  
mut_bound = 50  
decay_rate = 0.99
lambda_val_fedatmv=1 # Added for FedATMV, user can adjust if needed, kept original script's implicit value or a common default

# %%
def set_random_seed(seed_val): 
    random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
        # For full reproducibility, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False 

device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu')
if random_fix: set_random_seed(seed)

cifar_data_pool, test_dataset_obj, client_data, server_data = None, None, [], (np.array([]), np.array([]))
init_model, initial_w, client_data_mixed = None, None, []
data_base_path = os.path.join(os.getcwd(), 'data'); os.makedirs(data_base_path, exist_ok=True)

if dataset == 'cifar100':
    num_classes = 20 
    cifar_data_pool, test_dataset_obj = CIFAR100() 
    test_dataset = test_dataset_obj 
    prob_dist = get_prob(non_iid, client_num, class_num_val=num_classes, iid_mode=is_iid) 
    client_data = create_data_all_train(prob_dist, size_per_client, cifar_data_pool, N_classes=num_classes)
    server_images, server_labels = select_server_subset(cifar_data_pool, percentage=server_percentage, 
                                                        mode='iid' if server_iid else 'non-iid', dirichlet_alpha=server_dir)
    server_data = (server_images, server_labels)
    if origin_model == 'vgg': init_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'resnet': init_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == 'mobilenet': init_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise ValueError(f"Unsupported model {origin_model} for CIFAR100")

elif dataset =='shake':
    num_classes = 80 
    shake_data_path = os.path.join(data_base_path, 'shakespeare')
    train_dataset_obj = ShakeSpeare(train=True, data_root_path=shake_data_path) 
    test_dataset = ShakeSpeare(train=False, data_root_path=shake_data_path) 
    total_shake_samples_x, total_shake_samples_y = [], []
    for i in range(len(train_dataset_obj)):
        sample_x, sample_y = train_dataset_obj[i]
        total_shake_samples_x.append(sample_x.numpy())
        total_shake_samples_y.append(sample_y.item() if torch.is_tensor(sample_y) else sample_y)
    total_shake_samples_x = np.array(total_shake_samples_x, dtype=object) # dtype=object for ragged sequences if any
    total_shake_samples_y = np.array(total_shake_samples_y)
    dict_users_indices = train_dataset_obj.get_client_dic() 
    user_ids_available = sorted(list(dict_users_indices.keys()))
    selected_user_ids_for_fl = user_ids_available[:client_num]
    if len(selected_user_ids_for_fl) < client_num: client_num = len(selected_user_ids_for_fl); M = min(M, client_num)
    
    server_data_pool_images_list, server_data_pool_labels_list = [], []
    for user_id in selected_user_ids_for_fl:
        indices_for_user = np.array(dict_users_indices[user_id]).astype(int)
        indices_for_user = indices_for_user[indices_for_user < len(total_shake_samples_x)]
        client_x_data = total_shake_samples_x[indices_for_user]
        client_y_data = total_shake_samples_y[indices_for_user]
        client_data.append((client_x_data, client_y_data))
        server_data_pool_images_list.extend(client_x_data) 
        server_data_pool_labels_list.extend(client_y_data)
    if server_data_pool_images_list:
        server_pool_tuple = (np.array(server_data_pool_images_list, dtype=object), np.array(server_data_pool_labels_list))
        server_images, server_labels = select_server_subset(server_pool_tuple, percentage=server_percentage,
                                                          mode='iid' if server_iid else 'non-iid', dirichlet_alpha=server_dir)
        server_data = (server_images, server_labels)
    if origin_model == 'lstm': init_model = CharLSTM().to(device) 
    else: raise ValueError(f"Unsupported model {origin_model} for Shakespeare")

elif dataset == "cifar10":
    num_classes = 10
    cifar10_data_path = os.path.join(data_base_path, 'cifar10'); os.makedirs(cifar10_data_path, exist_ok=True)
    # Using simpler transform for pool, augmentations usually in training DataLoader
    pool_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset_for_pool = torchvision.datasets.CIFAR10(root=cifar10_data_path, train=True, download=True, transform=pool_transform)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_dataset = torchvision.datasets.CIFAR10(root=cifar10_data_path, train=False, download=True, transform=test_transform)
    
    total_img_list, total_label_list = [np.array(img_i) for img_i, _ in train_dataset_for_pool], [label_i for _, label_i in train_dataset_for_pool]
    cifar_data_pool = (np.array(total_img_list), np.array(total_label_list))
    prob_dist = get_prob(non_iid, client_num, class_num_val=num_classes, iid_mode=is_iid)
    client_data = create_data_all_train(prob_dist, size_per_client, cifar_data_pool, N_classes=num_classes)
    server_images, server_labels = select_server_subset(cifar_data_pool, percentage=server_percentage, 
                                                        mode="iid" if server_iid else "non-iid", dirichlet_alpha=server_dir)
    server_data = (server_images, server_labels)
    if origin_model == 'cnn': init_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'resnet': init_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg': init_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'mobilenet': init_model = mobilenetv2(num_classes_arg=num_classes).to(device)
    else: raise ValueError(f"Unsupported model {origin_model} for CIFAR10")
else: raise ValueError(f"Unknown dataset: {dataset}")

if init_model: initial_w = copy.deepcopy(init_model.state_dict())
else: print("Error: init_model was not created.")

if client_data and (server_data[0].size > 0 or server_data[1].size > 0 or server_percentage == 0):
    client_data_mixed = build_mixed_client_data(client_data, server_data, share_ratio=1.0, seed_val=seed if random_fix else None)
else: print("Warning: client_data or server_data is empty/invalid before build_mixed_client_data.")

print(f"Device: {device}"); print(f"Dataset: {dataset}, Model: {origin_model}, Num clients: {client_num}, Num classes: {num_classes}")
print(f"Server data size: {len(server_data[0]) if server_data and len(server_data[0])>0 else 0}")
if client_data:
    client_data_sizes = [len(cd[0]) if cd and len(cd[0])>0 else 0 for cd in client_data]
    print(f"Client data sizes (first 5): {client_data_sizes[:5]}, Total client samples: {sum(client_data_sizes)}")
    if sum(client_data_sizes) == 0 and client_num > 0: print("Warning: All clients have 0 samples.")
else: print("Client data not initialized.")


# %%
def run_once():
    results_test_acc, results_train_loss, results_comm_vs_acc, results_flops_vs_acc = {}, {}, {}, {}
    if init_model is None or initial_w is None: raise ValueError("init_model or initial_w not initialized.")

    # test_acc_so, train_loss_so, comm_so, flops_so = server_only(initial_w, global_round, gamma, E)
    # results_test_acc['Server-Only'] = test_acc_so; results_train_loss['Server-Only'] = train_loss_so
    # results_comm_vs_acc['Server-Only'] = comm_so; results_flops_vs_acc['Server-Only'] = flops_so

    test_acc_fa, train_loss_fa, comm_fa, flops_fa = fedavg(initial_w, global_round, eta, K, M)
    results_test_acc['FedAvg'] = test_acc_fa; results_train_loss['FedAvg'] = train_loss_fa
    results_comm_vs_acc['FedAvg'] = comm_fa; results_flops_vs_acc['FedAvg'] = flops_fa
    
    test_acc_hfl, train_loss_hfl, comm_hfl, flops_hfl = hybridFL(initial_w, global_round, eta, K, E, M)
    results_test_acc['HybridFL'] = test_acc_hfl; results_train_loss['HybridFL'] = train_loss_hfl
    results_comm_vs_acc['HybridFL'] = comm_hfl; results_flops_vs_acc['HybridFL'] = flops_hfl

    # test_acc_ds, train_loss_ds, comm_ds, flops_ds = Data_Sharing(initial_w, global_round, eta, K, M)
    # results_test_acc['Data_Sharing'] = test_acc_ds; results_train_loss['Data_Sharing'] = train_loss_ds
    # results_comm_vs_acc['Data_Sharing'] = comm_ds; results_flops_vs_acc['Data_Sharing'] = flops_ds
    
    test_acc_clgsgd, train_loss_clgsgd, comm_clgsgd, flops_clgsgd = CLG_SGD(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['CLG-SGD'] = test_acc_clgsgd; results_train_loss['CLG-SGD'] = train_loss_clgsgd
    results_comm_vs_acc['CLG-SGD'] = comm_clgsgd; results_flops_vs_acc['CLG-SGD'] = flops_clgsgd

    test_acc_fc, train_loss_fc, comm_fc, flops_fc = Fed_C(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['Fed-C'] = test_acc_fc ; results_train_loss['Fed-C'] = train_loss_fc
    results_comm_vs_acc['Fed-C'] = comm_fc; results_flops_vs_acc['Fed-C'] = flops_fc

    test_acc_fs, train_loss_fs, comm_fs, flops_fs = Fed_S(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['Fed-S'] = test_acc_fs ; results_train_loss['Fed-S'] = train_loss_fs
    results_comm_vs_acc['Fed-S'] = comm_fs; results_flops_vs_acc['Fed-S'] = flops_fs
    
    test_acc_fdum, train_loss_fdum, comm_fdum, flops_fdum = FedDU_modify(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['FedDU'] = test_acc_fdum ; results_train_loss['FedDU'] = train_loss_fdum
    results_comm_vs_acc['FedDU'] = comm_fdum; results_flops_vs_acc['FedDU'] = flops_fdum

    fedmut_model_instance = copy.deepcopy(init_model).to(device) 
    test_acc_fm, train_loss_fm, comm_fm, flops_fm = FedMut(fedmut_model_instance, global_round, eta, K, M)
    results_test_acc['FedMut'] = test_acc_fm; results_train_loss['FedMut'] = train_loss_fm
    results_comm_vs_acc['FedMut'] = comm_fm; results_flops_vs_acc['FedMut'] = flops_fm; del fedmut_model_instance

    # clgmut2_model_instance = copy.deepcopy(init_model).to(device)
    # test_acc_clgm2, train_loss_clgm2, comm_clgm2, flops_clgm2 = CLG_Mut_2(clgmut2_model_instance, global_round, eta, gamma, K, E, M)
    # results_test_acc['CLG_Mut_2'] = test_acc_clgm2; results_train_loss['CLG_Mut_2'] = train_loss_clgm2
    # results_comm_vs_acc['CLG_Mut_2'] = comm_clgm2; results_flops_vs_acc['CLG_Mut_2'] = flops_clgm2; del clgmut2_model_instance
    
    fedatmv_model_instance = copy.deepcopy(init_model).to(device)
    test_acc_fatmv, train_loss_fatmv, comm_fatmv, flops_fatmv = FedATMV(fedatmv_model_instance, global_round, eta, gamma, K, E, M, lambda_val_fedatmv)
    results_test_acc['FedATMV'] = test_acc_fatmv; results_train_loss['FedATMV'] = train_loss_fatmv
    results_comm_vs_acc['FedATMV'] = comm_fatmv; results_flops_vs_acc['FedATMV'] = flops_fatmv; del fedatmv_model_instance
    
    print("\n--- Accuracy & Loss at specific rounds/final ---")
    for algo_name in results_test_acc: 
        if results_test_acc[algo_name] and len(results_test_acc[algo_name]) > 0:
            if len(results_test_acc[algo_name]) >= 20: print(f"{algo_name} - R20 Acc: {results_test_acc[algo_name][19]:.2f}%, R20 Loss: {results_train_loss[algo_name][19]:.4f}")
            print(f"{algo_name} - Final Acc: {results_test_acc[algo_name][-1]:.2f}%, Final Loss: {results_train_loss[algo_name][-1]:.4f}")
    
    output_dir = os.path.join(os.getcwd(), "output", "FLOPs"); os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

    plt.figure(figsize=(12, 8))
    for algo, data_list in results_comm_vs_acc.items(): 
        if data_list: plt.plot([item['overhead'] / (1024*1024) for item in data_list], [item['accuracy'] for item in data_list], label=algo, marker='o', markersize=2, linestyle='-')
    plt.xlabel('Cumulative Communication Overhead (MB)'); plt.ylabel('Test Accuracy (%)')
    plt.title(f'Accuracy vs. Comm. Overhead ({dataset.upper()}-{origin_model.upper()})'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'all_comm_vs_acc_{dataset}_{origin_model}_{ts}.png')); plt.close()
    with open(os.path.join(output_dir, f'all_comm_data_{dataset}_{origin_model}_{ts}.json'), 'w') as f: json.dump(results_comm_vs_acc, f, indent=2)
    print(f"\nComm vs. Acc plot & data saved in {output_dir}")

    plt.figure(figsize=(12, 8))
    for algo, data_list in results_flops_vs_acc.items():
        if data_list: plt.plot([item['flops'] / 1e9 for item in data_list], [item['accuracy'] for item in data_list], label=algo, marker='s', markersize=2, linestyle='--')
    plt.xlabel('Cumulative FLOPs (GFLOPs)'); plt.ylabel('Test Accuracy (%)')
    plt.title(f'Accuracy vs. Cumulative FLOPs ({dataset.upper()}-{origin_model.upper()})'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'all_flops_vs_acc_{dataset}_{origin_model}_{ts}.png')); plt.close()
    flops_data_to_save = {algo: [{'gflops': item['flops'] / 1e9, 'accuracy': item['accuracy']} for item in data_list] for algo, data_list in results_flops_vs_acc.items()}
    with open(os.path.join(output_dir, f'all_flops_data_{dataset}_{origin_model}_{ts}.json'), 'w') as f: json.dump(flops_data_to_save, f, indent=2)
    print(f"FLOPs vs. Acc plot & data saved in {output_dir}")

    plt.figure(figsize=(12, 6))
    for algo, acc in results_test_acc.items():
        if acc : plt.plot(range(1, len(acc) + 1), acc, label=algo) 
    plt.xlabel('Training Rounds'); plt.ylabel('Test Accuracy (%)'); plt.title(f'Accuracy Comparison ({dataset}-{origin_model})')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'test_accuracy_{origin_model}_{dataset}_{ts}.png')); plt.close()

    plt.figure(figsize=(12, 6))
    for algo, loss in results_train_loss.items(): 
        if loss: plt.plot(range(1, len(loss) + 1), loss, label=algo)
    plt.xlabel('Training Rounds'); plt.ylabel('Train Loss'); plt.title(f'Train Loss Comparison ({dataset}-{origin_model})')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'train_loss_{origin_model}_{dataset}_{ts}.png')); plt.close()
    return results_test_acc, results_train_loss 

if __name__ == '__main__':
    print(f"Starting run_once: dataset={dataset}, model={origin_model}, rounds={global_round}, M={M}, K={K}, E={E}, lr_client={eta}, lr_server={gamma}")
    if initial_w is None: print("Error: initial_w is None. Data loading or model init failed.")
    else:
        if init_model is not None: init_model = init_model.to(device)
        initial_w = {k: v.to(device) for k,v in initial_w.items()}
        returned_test_acc, returned_train_loss = run_once()
        print("\nrun_once execution complete.")