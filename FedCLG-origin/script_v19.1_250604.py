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
# v 18.0_comm_overhead_actual_size
# - Added communication overhead tracking based on actual byte sizes.
# - Introduced get_object_size_in_bytes helper function.
# - Algorithms now use byte sizes for overhead calculation.
# - run_once now plots and saves this data with byte-based overhead.

# %%
# import os # Already imported
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# %%
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %%
def get_object_size_in_bytes(obj_dict):
    """Calculates the total size of a dictionary of tensors in bytes."""
    if not isinstance(obj_dict, dict):
        # print(f"Warning: get_object_size_in_bytes expected a dict, got {type(obj_dict)}. Returning 0.")
        return 0
    total_size = 0
    for key, value in obj_dict.items():
        if torch.is_tensor(value):
            total_size += value.nelement() * value.element_size()
        # Could add handling for other types if necessary, but state_dict/gradients are tensors.
    return total_size

# %%

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100): # class_num seems unused here
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

    def __init__(self, num_classes_arg=20): # Changed class_num to num_classes_arg
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, num_classes_arg, 1) # Use num_classes_arg

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
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2(num_classes_arg=20): # Added num_classes_arg
    return MobileNetV2(num_classes_arg=num_classes_arg)


class CNNCifar(nn.Module):
    def __init__(self, num_classes_arg): # Modified to accept num_classes_arg
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes_arg) # Use num_classes_arg

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        result = {'activation' : x}
        x = x.view(-1, 16 * 5 * 5)
        result['hint'] = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result['representation'] = x
        x = self.fc3(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc3(z)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result
    
def cnncifar(num_classes_arg): # Modified factory
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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCifar10(nn.Module):
    def __init__(self, block, layers, num_classes_arg, zero_init_residual=False, # Added num_classes_arg
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes_arg) # Use num_classes_arg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        result = {}
        x = self.layer1(x)
        result['activation1'] = x
        x = self.layer2(x)
        result['activation2'] = x
        x = self.layer3(x)
        result['activation3'] = x
        x = self.layer4(x)
        result['activation4'] = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        result['representation'] = x
        x = self.fc(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        return self._forward_impl(x)

def ResNet18_cifar10(num_classes_arg, **kwargs): # Modified factory
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], num_classes_arg=num_classes_arg, **kwargs)

def ResNet50_cifar10(num_classes_arg, **kwargs): # Modified factory
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], num_classes_arg=num_classes_arg, **kwargs)

# %%
def test_inference(net_glob, dataset_test):
    acc_test, loss_test = test_img(net_glob, dataset_test)
    return acc_test.item()

def test_img(net_g, datatest):
    net_g.eval()
    test_loss = 0
    correct = 0
    # Ensure datatest is a PyTorch Dataset object
    if not isinstance(datatest, Dataset):
        # Assuming datatest is a tuple (images, labels)
        if isinstance(datatest[0], np.ndarray):
             datatest = TensorDataset(torch.Tensor(datatest[0]), torch.Tensor(datatest[1]).long())
        else: # Assuming it's already tensors
             datatest = TensorDataset(datatest[0], datatest[1].long())

    data_loader = DataLoader(datatest, batch_size=test_bc_size)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            log_probs = net_g(data)['output']
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset) if len(data_loader.dataset) > 0 else 1
    accuracy = 100.00 * correct / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0
    if verbose:
        print(f'\nTest set: Average loss: {test_loss:.4f} \nAccuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
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
    return coarse_labels[targets]

# %%
def CIFAR100():
    trans_cifar100 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(root='../data/CIFAR-100', train=True, transform=trans_cifar100, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='../data/CIFAR-100', train=False, transform=trans_cifar100, download=True)
    
    total_img,total_label = [],[]
    for imgs,labels in train_dataset:
        total_img.append(imgs.numpy())
        total_label.append(labels)
    total_img = np.array(total_img)
    total_label = np.array(sparse2coarse(total_label))
    cifar = [total_img, total_label]
    return cifar, test_dataset

# %%
def get_prob(non_iid_strength, client_num_val, class_num_val=20, iid_mode=False): # Renamed params to avoid global conflicts
    if data_random_fix:
        np.random.seed(seed_num)
    if iid_mode:
        return np.ones((client_num_val, class_num_val)) / class_num_val
    else:
        return np.random.dirichlet(np.repeat(non_iid_strength, class_num_val), client_num_val)

# %%
def create_data_all_train(prob, size_per_client_val, dataset_val, N_classes=20): # Renamed params
    total_each_class = size_per_client_val * np.sum(prob, 0)
    data, label = dataset_val
    if data_random_fix:
        np.random.seed(seed_num)
        random.seed(seed_num)

    all_class_set = []
    for i in range(N_classes):
        size = total_each_class[i]
        sub_data = data[label == i]
        sub_label = label[label == i]
        num_samples = int(size)
        if num_samples > len(sub_data):
            # print(f"Warning: Class {i} has insufficient samples. Requested {num_samples}, available {len(sub_data)}. Taking all available.")
            num_samples = len(sub_data)
        if num_samples == 0 and len(sub_data) > 0 : 
             rand_indx = [] 
        elif num_samples == 0 and len(sub_data) == 0:
             rand_indx = []
        else:
             rand_indx = np.random.choice(len(sub_data), size=num_samples, replace=False).astype(int)
        
        sub2_data, sub2_label = sub_data[rand_indx], sub_label[rand_indx]
        all_class_set.append((sub2_data, sub2_label))

    index = [0] * N_classes
    clients = []
    for m in range(prob.shape[0]):
        labels_list, images_list = [], []
        for n in range(N_classes):
            start, end = index[n], index[n] + int(prob[m][n] * size_per_client_val)
            image_samples, label_samples = all_class_set[n][0][start:end], all_class_set[n][1][start:end]
            index[n] += int(prob[m][n] * size_per_client_val)
            labels_list.extend(label_samples)
            images_list.extend(image_samples)
        clients.append((np.array(images_list), np.array(labels_list)))
    return clients

# %%
def select_server_subset(cifar_data, percentage=0.1, mode='iid', dirichlet_alpha=1.0):
    images, labels = cifar_data
    unique_classes_arr = np.unique(labels)
    total_num = len(labels)
    server_total = int(total_num * percentage)
    selected_indices = []
    
    if mode == 'iid':
        for cls_val in unique_classes_arr:
            cls_indices = np.where(labels == cls_val)[0]
            num_cls = int(len(cls_indices) * percentage) 
            if percentage == 1.0 : num_cls = len(cls_indices) 

            if num_cls > len(cls_indices): num_cls = len(cls_indices)
            if num_cls == 0 and len(cls_indices) > 0 and server_total > 0 : num_cls = 1 

            sampled = np.random.choice(cls_indices, size=num_cls, replace=False) if len(cls_indices) > 0 and num_cls > 0 else []
            selected_indices.extend(sampled)
    elif mode == 'non-iid':
        classes_len = len(unique_classes_arr)
        prob_dist = np.random.dirichlet(np.repeat(dirichlet_alpha, classes_len)) if classes_len > 0 else np.array([]) # Handle empty unique_classes_arr
        cls_sample_numbers = {}
        total_assigned = 0
        for i, cls_val in enumerate(unique_classes_arr):
            n_cls = int(prob_dist[i] * server_total) if len(prob_dist) > i else 0
            cls_sample_numbers[cls_val] = n_cls
            total_assigned += n_cls
        
        diff = server_total - total_assigned
        if diff > 0 and len(unique_classes_arr) > 0: # Ensure unique_classes_arr is not empty for choice
            for cls_val_choice in np.random.choice(unique_classes_arr, size=diff, replace=True):
                cls_sample_numbers[cls_val_choice] += 1
        
        for cls_val in unique_classes_arr:
            cls_indices = np.where(labels == cls_val)[0]
            n_sample = cls_sample_numbers.get(cls_val, 0)
            if n_sample > len(cls_indices): n_sample = len(cls_indices)

            sampled = np.random.choice(cls_indices, size=n_sample, replace=False) if len(cls_indices) > 0 and n_sample > 0 else []
            selected_indices.extend(sampled)
    else:
        raise ValueError("mode 参数必须为 'iid' 或 'non-iid'")
    
    selected_indices = list(set(selected_indices)) 

    if server_fill and len(selected_indices) < server_total :
        shortfall = server_total - len(selected_indices)
        if shortfall > 0:
            remaining_pool = np.setdiff1d(np.arange(total_num), selected_indices, assume_unique=True)
            if shortfall > len(remaining_pool): shortfall = len(remaining_pool) 
            extra = np.random.choice(remaining_pool, shortfall, replace=False) if len(remaining_pool) > 0 and shortfall > 0 else [] # Ensure shortfall > 0
            selected_indices = np.concatenate([selected_indices, extra]) if len(extra) > 0 else np.array(selected_indices)
            
    selected_indices = np.array(selected_indices, dtype=int) 
    if len(selected_indices) > 0: # Shuffle only if there are indices
        np.random.shuffle(selected_indices) 
    
    if len(selected_indices) > server_total:
        selected_indices = selected_indices[:server_total]

    subset_images = images[selected_indices] if len(selected_indices) > 0 else np.array([])
    subset_labels = labels[selected_indices] if len(selected_indices) > 0 else np.array([])
    return subset_images, subset_labels

# %%
def update_weights(model_weight, dataset_val, learning_rate, local_epoch): # dataset_val
    if origin_model == 'resnet':
        model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        model = CharLSTM().to(device)
    elif origin_model == "cnn":
        model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
    
    model.load_state_dict(model_weight)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if len(dataset_val[0]) == 0:
        return model.state_dict(), 0.0, {}

    if origin_model == 'resnet' or origin_model == 'cnn' or origin_model == 'vgg':
        Tensor_set = TensorDataset(torch.Tensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device))
    elif origin_model == 'lstm':
        Tensor_set = TensorDataset(torch.LongTensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device)) 
    
    data_loader = DataLoader(Tensor_set, batch_size=bc_size, shuffle=True)
    first_iter_gradient = {} # Initialize as empty dict

    for iter_val in range(local_epoch): 
        batch_loss = []
        if not data_loader or len(data_loader.dataset) == 0 : # Check if dataset in dataloader is empty
            epoch_loss.append(0.0) 
            continue

        for batch_idx, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['output'], labels) 
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/images.shape[0] if images.shape[0] > 0 else 0.0)

            if iter_val == 0 and batch_idx == 0:
                # first_iter_gradient = {} # Already initialized
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        first_iter_gradient[name] = param.grad.clone()
                for name, module_val in model.named_modules(): 
                    if isinstance(module_val, nn.BatchNorm2d):
                        first_iter_gradient[name + '.running_mean'] = module_val.running_mean.clone()
                        first_iter_gradient[name + '.running_var'] = module_val.running_var.clone()
        epoch_loss.append(sum(batch_loss)/len(batch_loss) if len(batch_loss) > 0 else 0.0)
    
    final_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
    return model.state_dict(), final_loss, first_iter_gradient

# %%
def weight_differences(n_w, p_w, lr_val): # lr_val
    w_diff = copy.deepcopy(n_w)
    for key in w_diff.keys():
        if 'num_batches_tracked' in key:
            continue
        if key in p_w: # Ensure key exists in p_w
             w_diff[key] = (p_w[key] - n_w[key]) * lr_val
        # else:
             # print(f"Warning: Key {key} not in p_w during weight_differences.")
    return w_diff

# %%
def update_weights_correction(model_weight, dataset_val, learning_rate, local_epoch, c_i, c_s): # dataset_val
    if origin_model == 'resnet':
        model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        model = CharLSTM().to(device)
    elif origin_model == "cnn":
        model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    model.load_state_dict(model_weight)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if len(dataset_val[0]) == 0:
        return model.state_dict(), 0.0
    
    if origin_model == 'resnet' or origin_model == 'cnn' or origin_model == 'vgg':
        Tensor_set = TensorDataset(torch.Tensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device))
    elif origin_model == 'lstm':
        Tensor_set = TensorDataset(torch.LongTensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device))
        
    data_loader = DataLoader(Tensor_set, batch_size=bc_size, shuffle=True)

    for iter_val in range(local_epoch): 
        batch_loss = []
        if not data_loader or len(data_loader.dataset) == 0:
            epoch_loss.append(0.0)
            continue
        for batch_idx, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['output'], labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.sum().item()/images.shape[0] if images.shape[0] > 0 else 0.0)
            
        epoch_loss.append(sum(batch_loss)/len(batch_loss) if len(batch_loss) > 0 else 0.0)
        
        if c_i and c_s: # Both control variates must exist
            corrected_graident = weight_differences(c_i, c_s, learning_rate) 
            current_model_state_dict = model.state_dict() # Get current state dict
            # Ensure all keys from corrected_gradient are in current_model_state_dict before subtraction
            # And all keys from current_model_state_dict are in corrected_gradient for the operation in weight_differences
            # The weight_differences(A, B, factor) calculates (B-A)*factor.
            # Here, we want model_weights - (c_s - c_i)*lr = model_weights - corrected_gradient
            # So, it should be weight_differences(corrected_gradient, current_model_state_dict, 1)
            # where corrected_gradient is (c_s-c_i)*lr
            
            # Original logic: corrected_model_weight = weight_differences(corrected_graident, orginal_model_weight, 1)
            # This means: corrected_model_weight[key] = (orginal_model_weight[key] - corrected_graident[key]) * 1
            # where corrected_graident[key] = (c_s[key] - c_i[key]) * learning_rate
            # So, new_weight = old_weight - (c_s - c_i) * learning_rate. This is correct for Scaffold-like update.
            corrected_model_weight = weight_differences(corrected_graident, current_model_state_dict, 1)  
            model.load_state_dict(corrected_model_weight)
        elif not c_i and not c_s: 
            pass
        # else: 
            # print("Warning: Mismatched control variates in update_weights_correction. Skipping correction.")

    final_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
    return model.state_dict(), final_loss

# %%
def average_weights(w_list): 
    if not w_list: return {}
    # Filter out empty dicts, which can happen if a client had no data and returned an empty state_dict
    valid_w_list = [w for w in w_list if w]
    if not valid_w_list: return {}

    w_avg = copy.deepcopy(valid_w_list[0])
    for key in w_avg.keys():
        if 'num_batches_tracked' in key:
            continue
        for i in range(1, len(valid_w_list)):
            w_avg[key] += valid_w_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(valid_w_list))
    return w_avg

# %%
def server_only(initial_w, global_round_val, gamma_val, E_val): 
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    test_acc_list = [] 
    train_loss_list = [] 
    comm_vs_acc_list = [] 
    cumulative_overhead = 0 # Bytes

    for round_idx in tqdm(range(global_round_val)): 
        update_server_w, round_loss_val, _ = update_weights(train_w, server_data, gamma_val, E_val) 
        train_w = update_server_w
        test_model.load_state_dict(train_w)
        train_loss_list.append(round_loss_val)
        
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        current_round_overhead = 0 # No communication for server-only
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def fedavg(initial_w, global_round_val, eta_val, K_val, M_val): 
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    model_size_bytes = get_object_size_in_bytes(train_w) # Calculate once, assuming model structure is fixed

    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], [] 
        sampled_client_indices = random.sample(range(client_num), M_val) 
        active_clients_this_round = 0
        for client_idx in sampled_client_indices: 
            if len(client_data[client_idx][0]) == 0: 
                continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)

        if not local_weights: 
            loss_avg = train_loss_list[-1] if train_loss_list else 0.0
        else:
            train_w = average_weights(local_weights)
            loss_avg = sum(local_loss_vals)/ len(local_loss_vals)
        
        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Overhead: download global model, upload local model
        current_round_overhead = active_clients_this_round * (model_size_bytes + model_size_bytes) 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
            
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def hybridFL(initial_w, global_round_val, eta_val, K_val, E_val, M_val):
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    model_size_bytes = get_object_size_in_bytes(train_w)

    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round = 0

        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)

        if len(server_data[0]) > 0:
            update_server_w, server_round_loss, _ = update_weights(train_w, server_data, eta_val, E_val) 
            local_weights.append(update_server_w)
            local_loss_vals.append(server_round_loss)
        
        if not local_weights:
            loss_avg = train_loss_list[-1] if train_loss_list else 0.0
        else:
            train_w = average_weights(local_weights)
            loss_avg = sum(local_loss_vals) / len(local_loss_vals)

        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
    
        current_round_overhead = active_clients_this_round * (model_size_bytes + model_size_bytes)
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

def Data_Sharing(initial_w, global_round_val, eta_val, K_val, M_val, share_ratio=1):
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == 'cnn':
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'lstm':
        test_model = CharLSTM().to(device)
    else: raise ValueError("Unknown origin_model")

    local_datasets = client_data_mixed 
    w_global = copy.deepcopy(initial_w)
    model_size_bytes = get_object_size_in_bytes(w_global)

    all_test_acc_list = [] 
    all_train_loss_list = [] 
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes

    for r_idx in tqdm(range(global_round_val)): 
        selected_indices = np.random.choice(range(client_num), M_val, replace=False) 
        local_ws, local_ls = [], []
        active_clients_this_round = 0

        for cid in selected_indices:
            if len(local_datasets[cid][0]) == 0: continue
            active_clients_this_round +=1
            w_updated, loss_val, _ = update_weights(w_global, local_datasets[cid], eta_val, K_val) 
            local_ws.append(w_updated)
            local_ls.append(loss_val)

        if not local_ws:
            avg_loss = all_train_loss_list[-1] if all_train_loss_list else 0.0
        else:
            w_global = average_weights(local_ws)
            avg_loss = sum(local_ls) / len(local_ls)
        
        all_train_loss_list.append(avg_loss)
        test_model.load_state_dict(w_global)
        current_test_acc = test_inference(test_model, test_dataset)
        all_test_acc_list.append(current_test_acc)

        current_round_overhead = active_clients_this_round * (model_size_bytes + model_size_bytes)
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

    return all_test_acc_list, all_train_loss_list, comm_vs_acc_list

def build_mixed_client_data(client_data_val, server_data_val, share_ratio=1.0, seed_val=None): 
    if seed_val is not None:
        np.random.seed(seed_val)

    s_imgs, s_lbls = server_data_val
    s_imgs_arr = np.array(s_imgs) 
    s_lbls_arr = np.array(s_lbls) 

    if share_ratio < 1.0 and len(s_imgs_arr) > 0 :
        sel_idx = np.random.choice(len(s_imgs_arr),
                                   size=int(len(s_imgs_arr) * share_ratio),
                                   replace=False).astype(int)
        s_imgs_arr = s_imgs_arr[sel_idx]
        s_lbls_arr = s_lbls_arr[sel_idx]

    mixed_clients = []
    for imgs_c, lbls_c in client_data_val: 
        imgs_c_arr = np.array(imgs_c)
        lbls_c_arr = np.array(lbls_c)

        if len(s_imgs_arr) > 0: 
            new_imgs = np.concatenate([imgs_c_arr, s_imgs_arr], axis=0) if len(imgs_c_arr) > 0 else s_imgs_arr
            new_lbls = np.concatenate([lbls_c_arr, s_lbls_arr], axis=0) if len(lbls_c_arr) > 0 else s_lbls_arr
        else: 
            new_imgs = imgs_c_arr
            new_lbls = lbls_c_arr
        mixed_clients.append((new_imgs, new_lbls))
    return mixed_clients

# %%
def CLG_SGD(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    # model_size_bytes will be calculated based on train_w before client comm
    
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        
        model_size_for_client_download = get_object_size_in_bytes(train_w) # Size of model clients download

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round = 0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)
        
        # Overhead for client phase: download global, upload local
        # Assuming uploaded model has same size as downloaded one for this round.
        current_round_overhead = active_clients_this_round * (model_size_for_client_download + model_size_for_client_download)
        cumulative_overhead += current_round_overhead

        if local_weights: 
             train_w = average_weights(local_weights)
        
        if len(server_data[0]) > 0: # Server training is local, no additional comm overhead for this step
            update_server_w, server_loss, _ = update_weights(train_w, server_data, gamma_val, E_val) 
            train_w = update_server_w
            local_loss_vals.append(server_loss) 
        
        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

def Fed_C(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes

    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        g_i_list = []
        
        model_size_bytes = get_object_size_in_bytes(train_w) # Global model size
        gs_size_bytes = 0
        g_s = {}

        if len(server_data[0]) > 0:
             _, _, g_s = update_weights(train_w, server_data, gamma_val, 1) 
             gs_size_bytes = get_object_size_in_bytes(g_s)

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round = 0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            _, _, g_i = update_weights(train_w, client_data[client_idx], eta_val, 1) 
            g_i_list.append(g_i if g_i else {}) 

        client_iter = 0 
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            current_g_i = g_i_list[client_iter] if client_iter < len(g_i_list) and g_i_list else {} # Ensure g_i_list is not empty
            client_iter +=1
            
            update_client_w, client_round_loss = update_weights_correction(train_w, client_data[client_idx], eta_val, K_val, current_g_i, g_s)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)
        
        # Overhead: M clients download model, M clients download g_s, M clients upload model
        current_round_overhead = active_clients_this_round * (model_size_bytes + gs_size_bytes + model_size_bytes)
        cumulative_overhead += current_round_overhead
        
        if local_weights:
            train_w = average_weights(local_weights)
        
        if len(server_data[0]) > 0: # Server training is local
            update_server_w, server_loss, _ = update_weights(train_w, server_data, gamma_val, E_val)
            train_w = update_server_w
            local_loss_vals.append(server_loss)

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

    return test_acc_list, train_loss_list, comm_vs_acc_list

def Fed_S(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
    
    train_w = copy.deepcopy(initial_w)
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes

    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        g_i_list_for_s_correction = [] # Store g_i for server correction
        
        model_size_bytes = get_object_size_in_bytes(train_w)
        gi_size_bytes = 0 # Initialize, will be updated if clients compute g_i

        g_s = {}
        if len(server_data[0]) > 0:
            _, _, g_s = update_weights(train_w, server_data, gamma_val, 1)

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round = 0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val) 
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)
            
            _, _, g_i_for_correction = update_weights(train_w, client_data[client_idx], eta_val, 1) 
            if g_i_for_correction: # Check if gradient was computed
                 g_i_list_for_s_correction.append(g_i_for_correction)
                 if gi_size_bytes == 0: # Get size from the first valid g_i
                     gi_size_bytes = get_object_size_in_bytes(g_i_for_correction)

        # Overhead: M clients download model, M clients upload model, M clients upload g_i
        current_round_overhead = active_clients_this_round * (model_size_bytes + model_size_bytes + gi_size_bytes)
        cumulative_overhead += current_round_overhead

        if local_weights:
            train_w_aggregated = average_weights(local_weights) 
        else: 
            train_w_aggregated = copy.deepcopy(train_w) 

        if g_i_list_for_s_correction and g_s : 
            g_i_average = average_weights([g for g in g_i_list_for_s_correction if g]) 
            if g_i_average: 
                correction_g = weight_differences(g_i_average, g_s, K_val * eta_val) 
                train_w = weight_differences(correction_g, copy.deepcopy(train_w_aggregated), 1) 
            else: 
                train_w = train_w_aggregated
        else: 
            train_w = train_w_aggregated

        if len(server_data[0]) > 0: # Server training is local
            update_server_w, server_loss, _ = update_weights(train_w, server_data, gamma_val, E_val)
            train_w = update_server_w
            local_loss_vals.append(server_loss)

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def KL_divergence(p1, p2):
    d = 0
    for i in range(len(p1)):
        if p2[i] == 0 or p1[i] == 0: continue
        # Check for NaN/inf before log
        val_to_log = p1[i]/p2[i]
        if val_to_log <= 0 or not math.isfinite(val_to_log): continue # Skip if ratio is non-positive or not finite
        d += p1[i] * math.log(val_to_log, 2)
    return d

def calculate_js_divergence(p1, p2):
    p3 = []
    # Ensure p1 and p2 have the same length
    if len(p1) != len(p2):
        # print(f"Warning: p1 and p2 have different lengths in JS divergence ({len(p1)} vs {len(p2)}). Using shorter length.")
        min_len = min(len(p1), len(p2))
        p1 = p1[:min_len]
        p2 = p2[:min_len]

    for i in range(len(p1)): p3.append((p1[i] + p2[i])/2)
    
    # Handle cases where p3 might be all zeros if p1 and p2 are all zeros.
    if not any(p3): # If p3 is all zeros, KL will be problematic.
        return 0.0 # Or handle as an error/special case. JS div is 0 if p1=p2. If both are zero vectors, it's undefined or 0.

    kl_p1_p3 = KL_divergence(p1, p3)
    kl_p2_p3 = KL_divergence(p2, p3)
    return (kl_p1_p3 + kl_p2_p3) / 2.0


def ratio_combine(w1, w2, ratio=0):
    if not w1: return copy.deepcopy(w2) # If w1 is empty, return w2
    if not w2: return copy.deepcopy(w1) # If w2 is empty, return w1
    w = copy.deepcopy(w1)
    for key in w.keys():
        if 'num_batches_tracked' in key: continue
        if key in w2: # Ensure key exists in w2
            w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w

def FedDU_modify(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")

    train_w = copy.deepcopy(initial_w)
    test_model.load_state_dict(train_w)
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    server_min_iter = 0 
    
    all_client_labels_list = [] 
    for i in range(client_num): all_client_labels_list.extend(client_data[i][1])
    all_client_labels_arr = np.array(all_client_labels_list) 
    
    unique_classes_arr, client_counts = np.unique(all_client_labels_arr, return_counts=True) 
    P_dist = [0.0] * num_classes 
    for cls_val, count_val in zip(unique_classes_arr, client_counts):
        if cls_val < num_classes: P_dist[cls_val] = count_val / len(all_client_labels_arr) if len(all_client_labels_arr) > 0 else 0.0
    
    server_labels_arr = np.array(server_data[1]) 
    n_0 = len(server_labels_arr)
    
    P_0_dist = [0.0] * num_classes 
    if n_0 > 0:
        unique_server_cls, server_cls_counts = np.unique(server_labels_arr, return_counts=True)
        for cls_val, count_val in zip(unique_server_cls, server_cls_counts):
             if cls_val < num_classes: P_0_dist[cls_val] = count_val / n_0
    
    D_P_0 = calculate_js_divergence(P_0_dist, P_dist)
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals_iter = [], [] 
        
        model_size_bytes_download = get_object_size_in_bytes(train_w) # Model downloaded by clients

        sampled_clients_indices = random.sample(range(client_num), M_val) 
        num_current_samples = 0 
        active_clients_this_round = 0
        for client_idx in sampled_clients_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            num_current_samples += len(client_data[client_idx][0])
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals_iter.append(client_round_loss)
        
        # Overhead for client phase (download global, upload local)
        # Assuming uploaded model has same size as downloaded
        current_round_client_overhead = active_clients_this_round * (model_size_bytes_download + model_size_bytes_download)
        cumulative_overhead += current_round_client_overhead
        
        if not local_weights: 
            w_t_half = copy.deepcopy(train_w) 
        else:
            w_t_half = average_weights(local_weights)
        
        selected_client_labels_list = [] 
        for client_idx in sampled_clients_indices: selected_client_labels_list.extend(client_data[client_idx][1])
        selected_client_labels_arr = np.array(selected_client_labels_list) 
        
        P_t_prime_dist = [0.0] * num_classes 
        if len(selected_client_labels_arr) > 0:
            unique_selected_cls, selected_cls_counts = np.unique(selected_client_labels_arr, return_counts=True)
            for cls_val, count_val in zip(unique_selected_cls, selected_cls_counts):
                if cls_val < num_classes: P_t_prime_dist[cls_val] = count_val / len(selected_client_labels_arr)
        
        D_P_t_prime = calculate_js_divergence(P_t_prime_dist, P_dist)
        
        test_model.load_state_dict(w_t_half)
        acc_t = test_inference(test_model, test_dataset) / 100.0
        
        avg_iter_val = (num_current_samples * K_val) / (M_val * bc_size) if M_val > 0 and bc_size > 0 else 0 
        epsilon = 1e-10
        alpha_dyn = (1 - acc_t) * (n_0 * D_P_t_prime) / (n_0 * D_P_t_prime + num_current_samples * D_P_0 + epsilon) if (n_0 * D_P_t_prime + num_current_samples * D_P_0 + epsilon) != 0 else 0
        alpha_dyn = alpha_dyn * (decay_rate ** round_idx) * du_C
        
        server_iter_count = max(server_min_iter, int(alpha_dyn * avg_iter_val)) 
        
        current_round_loss_server = 0.0 
        # Server update is local, no additional communication overhead for this step
        if alpha_dyn > 0.001 and n_0 > 0: 
            actual_server_iter = math.ceil(n_0 / bc_size) * E_val if bc_size > 0 else 0 
            effective_server_iter = min(actual_server_iter, server_iter_count) 
            
            if effective_server_iter > 0 : 
                update_server_w, current_round_loss_server, _ = update_weights(copy.deepcopy(w_t_half), server_data, gamma_val, E_val) 
                local_loss_vals_iter.append(current_round_loss_server)
                train_w = ratio_combine(w_t_half, update_server_w, alpha_dyn) 
            else: 
                train_w = copy.deepcopy(w_t_half)
        else:
            train_w = copy.deepcopy(w_t_half)
            if n_0 > 0:
                _, current_round_loss_server, _ = update_weights(copy.deepcopy(w_t_half), server_data, gamma_val, E_val) 
                local_loss_vals_iter.append(current_round_loss_server)

        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss_vals_iter) / len(local_loss_vals_iter) if local_loss_vals_iter else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def Aggregation(w_list, lens_list): 
    w_avg = None
    if not w_list: return {}
    
    valid_w_list = [w for w in w_list if w] # Filter out empty dicts
    if not valid_w_list: return {}
    
    if lens_list is None:
        total_count = len(valid_w_list)
        lens_list = [1.0] * len(valid_w_list) # Adjust lens_list to match valid_w_list
    else: # If lens_list is provided, it should correspond to the original w_list
          # We need to filter lens_list as well if w_list had empty dicts
        valid_lens_list = [lens_list[i] for i, w in enumerate(w_list) if w]
        total_count = sum(valid_lens_list)
        lens_list = valid_lens_list # Use filtered lens
        if total_count == 0: 
            return copy.deepcopy(valid_w_list[0]) if valid_w_list else {}

    w_avg = copy.deepcopy(valid_w_list[0])
    for k_key in w_avg.keys(): 
        w_avg[k_key] = valid_w_list[0][k_key] * lens_list[0] # Initialize with the first valid item

    for i in range(1, len(valid_w_list)):
        for k_key in w_avg.keys():
            if k_key in valid_w_list[i]: # Ensure key exists
                 w_avg[k_key] += valid_w_list[i][k_key] * lens_list[i]

    for k_key in w_avg.keys():
        if total_count > 0 : w_avg[k_key] = torch.div(w_avg[k_key], total_count)
    return w_avg

def FedSub(w_curr, w_prev, weight_val): 
    if not w_curr or not w_prev: return {} # Handle empty inputs
    w_sub = copy.deepcopy(w_curr)
    for k_key in w_sub.keys():
        if 'num_batches_tracked' in k_key: 
            w_sub[k_key] = w_curr[k_key] 
            continue
        if k_key in w_prev: # Ensure key exists in w_prev
            w_sub[k_key] = (w_curr[k_key] - w_prev[k_key]) * weight_val
        # else:
            # print(f"Warning: Key {k_key} not in w_prev during FedSub.")
    return w_sub

def delta_rank(delta_dict):
    if not delta_dict : return 0.0
    dict_a_list = [] 
    for p_key in delta_dict.keys(): 
        if 'num_batches_tracked' in p_key: continue 
        a_tensor = delta_dict[p_key] 
        if not torch.is_tensor(a_tensor): continue 
        dict_a_list.append(a_tensor.view(-1).float()) 
    if not dict_a_list: return 0.0
    
    dict_a_combined = torch.cat(dict_a_list, dim=0) 
    s_norm = torch.norm(dict_a_combined, p=2, dim=0) 
    return s_norm.item()


def mutation_spread(iter_val, w_glob_val, m_clients, w_delta_val, alpha_mut): 
    w_locals_new_list = [] 
    ctrl_cmd_list_outer = [] 
    
    if not w_glob_val or not w_delta_val : # Handle empty inputs
        return [copy.deepcopy(w_glob_val) for _ in range(m_clients)] if w_glob_val else [{} for _ in range(m_clients)]

    ctrl_rate_val = mut_acc_rate * (1.0 - min(iter_val * 1.0 / mut_bound if mut_bound > 0 else 1.0 , 1.0)) 

    for k_key in w_glob_val.keys():
        if 'num_batches_tracked' in k_key : continue 
        ctrl_list_inner = [] 
        for _ in range(0, int(m_clients / 2)): 
            ctrl_rand = random.random() 
            if ctrl_rand > 0.5:
                ctrl_list_inner.append(1.0)
                ctrl_list_inner.append(1.0 * (-1.0 + ctrl_rate_val))
            else:
                ctrl_list_inner.append(1.0 * (-1.0 + ctrl_rate_val))
                ctrl_list_inner.append(1.0)
        if m_clients % 2 == 1: 
             ctrl_list_inner.append(0.0) 
        random.shuffle(ctrl_list_inner)
        ctrl_cmd_list_outer.append(ctrl_list_inner)
    
    client_counter = 0 
    for j_client in range(m_clients): 
        w_sub_mutated = copy.deepcopy(w_glob_val) 
        if not (client_counter == m_clients - 1 and m_clients % 2 == 1):
            param_idx = 0 
            for k_key in w_sub_mutated.keys():
                if 'num_batches_tracked' in k_key : continue
                if param_idx < len(ctrl_cmd_list_outer) and \
                   j_client < len(ctrl_cmd_list_outer[param_idx]) and \
                   k_key in w_delta_val: # Check k_key in w_delta_val
                     w_sub_mutated[k_key] = w_sub_mutated[k_key] + w_delta_val[k_key] * ctrl_cmd_list_outer[param_idx][j_client] * alpha_mut
                param_idx += 1
        client_counter += 1
        w_locals_new_list.append(w_sub_mutated)
    return w_locals_new_list

def FedMut(net_glob_model, global_round_val, eta_val, K_val, M_val): 
    net_glob_model.train()
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    w_locals_list = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)] 
    model_size_bytes = get_object_size_in_bytes(net_glob_model.state_dict()) # Size of each personalized model

    max_rank_val = 0 
    
    for round_idx in tqdm(range(global_round_val)):
        w_old_global = copy.deepcopy(net_glob_model.state_dict()) 
        local_loss_vals = []
        
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False) 
        active_clients_this_round = 0
        
        temp_w_locals_for_agg = [] # Store models from clients that actually trained

        for i_local, client_actual_idx in enumerate(idxs_users_sampled): 
            if len(client_data[client_actual_idx][0]) == 0: 
                # If client has no data, add its current (possibly mutated) model to aggregation list
                # This ensures w_locals_list for mutation has M_val entries.
                # However, for aggregation, only trained models should ideally contribute.
                # Let's assume w_locals_list[i_local] is used if not trained.
                temp_w_locals_for_agg.append(w_locals_list[i_local])
                continue 
            active_clients_this_round +=1
            current_client_model_state = w_locals_list[i_local]
            updated_client_w, client_round_loss, _ = update_weights(current_client_model_state, 
                                                                    client_data[client_actual_idx], 
                                                                    eta_val, K_val)
            w_locals_list[i_local] = copy.deepcopy(updated_client_w) 
            temp_w_locals_for_agg.append(updated_client_w)
            local_loss_vals.append(client_round_loss)

        w_aggregated = Aggregation(temp_w_locals_for_agg, None) if temp_w_locals_for_agg else copy.deepcopy(w_old_global)
        
        if not w_aggregated: 
            w_aggregated = copy.deepcopy(w_old_global) 

        net_glob_model.load_state_dict(w_aggregated) 
        
        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(w_aggregated)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Overhead: M clients download (personalized), M clients upload (personalized)
        current_round_overhead = active_clients_this_round * (model_size_bytes + model_size_bytes) 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

        w_delta_mutation = FedSub(w_aggregated, w_old_global, 1.0) 
        rank_val = delta_rank(w_delta_mutation) 
        if rank_val > max_rank_val: max_rank_val = rank_val
        
        alpha_for_mutation = radius 
        w_locals_list = mutation_spread(round_idx, w_aggregated, M_val, w_delta_mutation, alpha_for_mutation)

    return test_acc_list, train_loss_list, comm_vs_acc_list   

# %%
def CLG_Mut_2(net_glob_model, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    net_glob_model.train()
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    w_locals_list = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)]
    model_size_bytes = get_object_size_in_bytes(net_glob_model.state_dict()) # Personalized model size

    max_rank_val = 0

    for round_idx in tqdm(range(global_round_val)):
        w_old_global_round = copy.deepcopy(net_glob_model.state_dict()) 
        local_loss_vals = []
        
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False)
        active_clients_this_round = 0
        temp_w_locals_for_agg = []


        for i_local, client_actual_idx in enumerate(idxs_users_sampled):
            if len(client_data[client_actual_idx][0]) == 0: 
                temp_w_locals_for_agg.append(w_locals_list[i_local]) # Add current state if not training
                continue
            active_clients_this_round +=1
            current_client_model_state = w_locals_list[i_local] 
            updated_client_w, client_round_loss, _ = update_weights(current_client_model_state, 
                                                                    client_data[client_actual_idx], 
                                                                    eta_val, K_val)
            w_locals_list[i_local] = copy.deepcopy(updated_client_w)
            temp_w_locals_for_agg.append(updated_client_w)
            local_loss_vals.append(client_round_loss)

        # Overhead for client phase
        current_round_client_overhead = active_clients_this_round * (model_size_bytes + model_size_bytes)
        cumulative_overhead += current_round_client_overhead

        w_aggregated_clients = Aggregation(temp_w_locals_for_agg, None) if temp_w_locals_for_agg else copy.deepcopy(w_old_global_round)
        
        w_after_server_train = w_aggregated_clients 
        if len(server_data[0]) > 0: # Server training is local
            w_after_server_train, server_loss, _ = update_weights(w_aggregated_clients, server_data, gamma_val, E_val)
            local_loss_vals.append(server_loss)
        
        net_glob_model.load_state_dict(w_after_server_train) 

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(w_after_server_train)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

        w_delta_mutation = FedSub(w_after_server_train, w_old_global_round, 1.0)
        rank_val = delta_rank(w_delta_mutation)
        if rank_val > max_rank_val: max_rank_val = rank_val
        
        alpha_for_mutation = radius 
        w_locals_list = mutation_spread(round_idx, w_after_server_train, M_val, w_delta_mutation, alpha_for_mutation)

    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def FedATMV(net_glob_model, global_round_val, eta_val, gamma_val, K_val, E_val, M_val, lambda_val_fedatmv=1): 
    net_glob_model.train()
    # ... (model instantiation as before) ...
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
    
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0 # Bytes
    
    w_locals_list = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)]
    personalized_model_size_bytes = get_object_size_in_bytes(net_glob_model.state_dict())

    max_rank_val = 0
    
    all_client_labels_list = []
    for i in range(client_num): all_client_labels_list.extend(client_data[i][1])
    all_client_labels_arr = np.array(all_client_labels_list)
    
    unique_classes_arr_fedatmv, client_counts_fedatmv = np.unique(all_client_labels_arr, return_counts=True) 
    P_dist_fedatmv = [0.0] * num_classes
    for cls_val, count_val in zip(unique_classes_arr_fedatmv, client_counts_fedatmv):
        if cls_val < num_classes: P_dist_fedatmv[cls_val] = count_val / len(all_client_labels_arr) if len(all_client_labels_arr) > 0 else 0.0
    
    server_labels_arr_fedatmv = np.array(server_data[1])
    n_0_fedatmv = len(server_labels_arr_fedatmv)
    
    P_0_dist_fedatmv = [0.0] * num_classes
    if n_0_fedatmv > 0:
        unique_server_cls_fedatmv, server_cls_counts_fedatmv = np.unique(server_labels_arr_fedatmv, return_counts=True)
        for cls_val, count_val in zip(unique_server_cls_fedatmv, server_cls_counts_fedatmv):
            if cls_val < num_classes: P_0_dist_fedatmv[cls_val] = count_val / n_0_fedatmv
            
    D_P_0_fedatmv = calculate_js_divergence(P_0_dist_fedatmv, P_dist_fedatmv)
    
    alpha_history_fedatmv, improvement_history_fedatmv = [], [] 
    acc_prev_fedatmv = 0.0 
    
    for round_idx in tqdm(range(global_round_val)):
        w_old_global_round = copy.deepcopy(net_glob_model.state_dict())
        local_loss_vals = []
        
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False)
        selected_client_labels_list_fedatmv = [] 
        num_current_samples_fedatmv = 0 
        active_clients_this_round = 0
        temp_w_locals_for_agg = []


        for i_local, client_actual_idx in enumerate(idxs_users_sampled):
            if len(client_data[client_actual_idx][0]) == 0: 
                temp_w_locals_for_agg.append(w_locals_list[i_local])
                continue
            active_clients_this_round +=1
            current_client_model_state = w_locals_list[i_local]
            updated_client_w, client_round_loss, _ = update_weights(current_client_model_state, 
                                                                    client_data[client_actual_idx], 
                                                                    eta_val, K_val)
            w_locals_list[i_local] = copy.deepcopy(updated_client_w)
            temp_w_locals_for_agg.append(updated_client_w)
            local_loss_vals.append(client_round_loss)
            selected_client_labels_list_fedatmv.extend(client_data[client_actual_idx][1])
            num_current_samples_fedatmv += len(client_data[client_actual_idx][0])

        # Client phase overhead
        current_round_client_overhead = active_clients_this_round * (personalized_model_size_bytes + personalized_model_size_bytes)
        cumulative_overhead += current_round_client_overhead

        w_aggregated_clients = Aggregation(temp_w_locals_for_agg, None) if temp_w_locals_for_agg else copy.deepcopy(w_old_global_round)
        # net_glob_model.load_state_dict(w_aggregated_clients) # Not yet, server step might modify it
        
        selected_client_labels_arr_fedatmv = np.array(selected_client_labels_list_fedatmv)
        P_t_prime_dist_fedatmv = [0.0] * num_classes
        if len(selected_client_labels_arr_fedatmv) > 0:
            unique_selected_cls_fedatmv, selected_cls_counts_fedatmv = np.unique(selected_client_labels_arr_fedatmv, return_counts=True)
            for cls_val, count_val in zip(unique_selected_cls_fedatmv, selected_cls_counts_fedatmv):
                 if cls_val < num_classes: P_t_prime_dist_fedatmv[cls_val] = count_val / len(selected_client_labels_arr_fedatmv)
        
        D_P_t_prime_fedatmv = calculate_js_divergence(P_t_prime_dist_fedatmv, P_dist_fedatmv)
        
        test_model.load_state_dict(w_aggregated_clients) 
        acc_t_fedatmv = test_inference(test_model, test_dataset) / 100.0
        
        epsilon_fedatmv = 1e-10 
        r_data_fedatmv = n_0_fedatmv / (n_0_fedatmv + num_current_samples_fedatmv + epsilon_fedatmv) if (n_0_fedatmv + num_current_samples_fedatmv + epsilon_fedatmv) !=0 else 0
        r_noniid_fedatmv = D_P_t_prime_fedatmv / (D_P_t_prime_fedatmv + D_P_0_fedatmv + epsilon_fedatmv) if (D_P_t_prime_fedatmv + D_P_0_fedatmv + epsilon_fedatmv) !=0 else 0
        
        improvement_fedatmv = 0.0
        if round_idx > 0 : 
            improvement_fedatmv = max(0.0, acc_prev_fedatmv - acc_t_fedatmv) / (acc_prev_fedatmv + epsilon_fedatmv) if (acc_prev_fedatmv + epsilon_fedatmv) !=0 else 0
        
        min_alpha_fedatmv, max_alpha_fedatmv = 0.001, 1.0 
        alpha_new_fedatmv = du_C * (1 - acc_t_fedatmv) * r_data_fedatmv * r_noniid_fedatmv + lambda_val_fedatmv * improvement_fedatmv
        alpha_new_fedatmv = max(min_alpha_fedatmv, min(max_alpha_fedatmv, alpha_new_fedatmv))
        
        alpha_history_fedatmv.append(alpha_new_fedatmv)
        improvement_history_fedatmv.append(improvement_fedatmv)
        acc_prev_fedatmv = acc_t_fedatmv 

        final_model_state = w_aggregated_clients 
        # Server update is local
        if alpha_new_fedatmv > 0.001 and n_0_fedatmv > 0:
            update_server_w, server_loss, _ = update_weights(copy.deepcopy(w_aggregated_clients), server_data, gamma_val, E_val)
            local_loss_vals.append(server_loss)
            final_model_state = ratio_combine(w_aggregated_clients, update_server_w, alpha_new_fedatmv)
        elif n_0_fedatmv > 0: 
             _, server_loss, _ = update_weights(copy.deepcopy(w_aggregated_clients), server_data, gamma_val, E_val)
             local_loss_vals.append(server_loss)
        
        net_glob_model.load_state_dict(final_model_state) 
        
        loss_avg = sum(local_loss_vals) / len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(final_model_state)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
        w_delta_mutation = FedSub(final_model_state, w_old_global_round, 1.0)
        rank_val = delta_rank(w_delta_mutation)
        if rank_val > max_rank_val: max_rank_val = rank_val
            
        tmp_radius_fedatmv = radius * (1 + scal_ratio * alpha_new_fedatmv) 
        w_locals_list = mutation_spread(round_idx, final_model_state, M_val, w_delta_mutation, tmp_radius_fedatmv)
          
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
# Global parameters 
# ... (definitions as before) ...
data_random_fix = False 
seed_num = 42
random_fix = True
seed = 2
GPU = 1
verbose = False
client_num = 100
size_per_client = 400
is_iid = False
non_iid = 0.1
server_iid = True
server_dir = 1.0
server_percentage = 0.1
server_fill = True
origin_model = 'resnet' 
dataset = 'cifar10' 
momentum = 0.5
weight_decay = 0
bc_size = 50
test_bc_size = 128
num_classes = 10 
global_round = 100 # For actual runs, use 100. For testing, can be smaller.
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

# %%
def set_random_seed(seed_val): # seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu')

if random_fix:
    set_random_seed(seed)

# Initialize these as placeholders, they will be populated based on dataset
cifar = None
test_dataset = None # This will be the actual test dataset object
client_data = [] # List of (images_np_array, labels_np_array) tuples
server_data = [[], []] # [[images_np_array], [labels_np_array]]
init_model = None # PyTorch model instance
initial_w = None # state_dict
client_data_mixed = [] # For Data_Sharing

if dataset == 'cifar100':
    num_classes = 20 
    cifar, test_dataset_obj = CIFAR100() # test_dataset_obj is the Dataset object
    test_dataset = test_dataset_obj # Assign to global test_dataset
    prob_dist = get_prob(non_iid, client_num, class_num_val=num_classes, iid_mode=is_iid) 
    client_data = create_data_all_train(prob_dist, size_per_client, cifar, N_classes=num_classes)
    
    # Modify test_dataset targets in-place if it's a torchvision dataset
    if hasattr(test_dataset, 'targets'):
        test_dataset.targets = sparse2coarse(test_dataset.targets) 
        test_dataset.targets = np.array(test_dataset.targets).astype(int) 
    elif hasattr(test_dataset, 'labels'): # some datasets use 'labels'
        test_dataset.labels = sparse2coarse(test_dataset.labels)
        test_dataset.labels = np.array(test_dataset.labels).astype(int)


    server_images, server_labels = select_server_subset(cifar, percentage=server_percentage, 
                                                        mode='iid' if server_iid else 'non-iid', 
                                                        dirichlet_alpha=server_dir)
    server_data = [server_images, server_labels]
    if origin_model == 'vgg':
        init_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'resnet': 
        init_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model {origin_model} for CIFAR100")
    initial_w = copy.deepcopy(init_model.state_dict())

elif dataset =='shake':
    num_classes = 80 
    train_dataset_obj = ShakeSpeare(True) 
    test_dataset = ShakeSpeare(False) 

    total_shake_imgs, total_shake_labels = [],[] 
    for item_data, label_data in train_dataset_obj: 
        total_shake_imgs.append(item_data.numpy()) 
        total_shake_labels.append(label_data) 
    
    if total_shake_labels and isinstance(total_shake_labels[0], torch.Tensor): # Check if not empty
        total_shake_labels = [lbl.item() for lbl in total_shake_labels]

    total_shake_imgs_arr = np.array(total_shake_imgs, dtype=object) 
    total_shake_labels_arr = np.array(total_shake_labels)

    shake_data_tuple = [total_shake_imgs_arr, total_shake_labels_arr] 
    dict_users_shake = train_dataset_obj.get_client_dic() 
    
    # client_num = len(dict_users_shake) # client_num is a global, might be pre-set.
                                       # If ShakeSpeare defines clients, use that.
                                       # For consistency with fixed client_num, this line might be removed
                                       # or ensure dict_users_shake has `client_num` entries.
                                       # Let's assume client_num global is the target.

    client_data = []
    # Ensure we create 'client_num' clients, even if dict_users_shake has more/less.
    # This part needs careful handling if dict_users_shake keys don't map 0 to client_num-1
    # Assuming dict_users_shake keys are 0...N-1 where N can be > client_num
    # We will only use the first 'client_num' clients from dict_users_shake if N > client_num
    
    sorted_client_keys = sorted(dict_users_shake.keys())
    for i in range(min(client_num, len(sorted_client_keys))): # Iterate up to client_num or available clients
        client_id_key = sorted_client_keys[i]
        indices = np.array(list(dict_users_shake[client_id_key]), dtype=np.int64)
        indices = indices[indices < len(total_shake_imgs_arr)]
        client_images_val = total_shake_imgs_arr[indices] 
        client_labels_val = total_shake_labels_arr[indices] 
        client_data.append((client_images_val, client_labels_val))
    
    # If client_data has fewer than client_num entries due to fewer available in ShakeSpeare,
    # this could be an issue for algorithms expecting M samples from client_num.
    # For now, proceeding with the number of clients actually populated.
    # Consider adjusting client_num = len(client_data) here if dynamic client count is okay.


    server_images, server_labels = select_server_subset(shake_data_tuple, percentage=server_percentage,
                                                      mode='iid' if server_iid else 'non-iid', 
                                                      dirichlet_alpha=server_dir)
    server_data = [server_images, server_labels]
    if origin_model == 'lstm':
        init_model = CharLSTM().to(device) # Ensure CharLSTM is defined correctly
    else:
        raise ValueError(f"Unsupported model {origin_model} for Shakespeare")
    initial_w = copy.deepcopy(init_model.state_dict())

elif dataset == "cifar10":
    num_classes = 10
    trans_cifar10_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_cifar10_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset_obj = torchvision.datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=trans_cifar10_train)
    test_dataset = torchvision.datasets.CIFAR10("./data/cifar10", train=False, download=True, transform=trans_cifar10_val)
    
    total_img_list, total_label_list = [], [] 
    for img_i, label_i in train_dataset_obj: 
        total_img_list.append(np.array(img_i))
        total_label_list.append(label_i)
    total_img_arr = np.array(total_img_list) 
    total_label_arr = np.array(total_label_list) 
    cifar = [total_img_arr, total_label_arr]

    prob_dist = get_prob(non_iid, client_num, class_num_val=num_classes, iid_mode=is_iid)
    client_data = create_data_all_train(prob_dist, size_per_client, cifar, N_classes=num_classes)
    
    server_images, server_labels = select_server_subset(cifar, percentage=server_percentage, 
                                                        mode="iid" if server_iid else "non-iid", 
                                                        dirichlet_alpha=server_dir)
    server_data = [server_images, server_labels]
    
    if origin_model == 'cnn':    
        init_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'resnet':
        init_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        init_model = VGG16(num_classes, 3).to(device)
    else:
        raise ValueError(f"Unsupported model {origin_model} for CIFAR10")
    initial_w = copy.deepcopy(init_model.state_dict())

# Common post-dataset setup
if client_data and server_data: # Ensure they are populated
    client_data_mixed = build_mixed_client_data(client_data, server_data, share_ratio=1.0, seed_val=seed if random_fix else None)
else:
    print("Warning: client_data or server_data is empty before build_mixed_client_data.")


print(f"Dataset: {dataset}, Model: {origin_model}, Num clients: {client_num}, Num classes: {num_classes}")
print(f"Server data size: {len(server_data[0]) if server_data and len(server_data)>0 else 0}")
if client_data:
    client_data_sizes = [len(cd[0]) if cd and len(cd)>0 else 0 for cd in client_data]
    print(f"Client data sizes (first 5): {client_data_sizes[:5]}, Total client samples: {sum(client_data_sizes)}")
else:
    print("Client data not initialized.")


# %%
def run_once():
    results_test_acc = {}
    results_train_loss = {}
    results_comm_vs_acc = {} 

    if init_model is None or initial_w is None:
        raise ValueError("init_model or initial_w is not initialized. Check dataset loading.")

    # # Server-Only
    # test_acc_so, train_loss_so, comm_so = server_only(initial_w, global_round, gamma, E)
    # results_test_acc['Server-Only'] = test_acc_so
    # results_train_loss['Server-Only'] = train_loss_so
    # results_comm_vs_acc['Server-Only'] = comm_so

    # FedAvg
    test_acc_fa, train_loss_fa, comm_fa = fedavg(initial_w, global_round, eta, K, M)
    results_test_acc['FedAvg'] = test_acc_fa
    results_train_loss['FedAvg'] = train_loss_fa
    results_comm_vs_acc['FedAvg'] = comm_fa
    
    # HybridFL
    test_acc_hfl, train_loss_hfl, comm_hfl = hybridFL(initial_w, global_round, eta, K, E, M)
    results_test_acc['HybridFL'] = test_acc_hfl
    results_train_loss['HybridFL'] = train_loss_hfl
    results_comm_vs_acc['HybridFL'] = comm_hfl

    # # Data_Sharing
    # test_acc_ds, train_loss_ds, comm_ds = Data_Sharing(initial_w, global_round, eta, K, M)
    # results_test_acc['Data_Sharing'] = test_acc_ds
    # results_train_loss['Data_Sharing'] = train_loss_ds
    # results_comm_vs_acc['Data_Sharing'] = comm_ds
    
    # CLG_SGD
    test_acc_clgsgd, train_loss_clgsgd, comm_clgsgd = CLG_SGD(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['CLG-SGD'] = test_acc_clgsgd
    results_train_loss['CLG-SGD'] = train_loss_clgsgd
    results_comm_vs_acc['CLG-SGD'] = comm_clgsgd

    # Fed_C
    test_acc_fc, train_loss_fc, comm_fc = Fed_C(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['Fed-C'] = test_acc_fc 
    results_train_loss['Fed-C'] = train_loss_fc
    results_comm_vs_acc['Fed-C'] = comm_fc

    # Fed_S
    test_acc_fs, train_loss_fs, comm_fs = Fed_S(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['Fed-S'] = test_acc_fs 
    results_train_loss['Fed-S'] = train_loss_fs
    results_comm_vs_acc['Fed-S'] = comm_fs
    
    # FedDU_modify
    test_acc_fdum, train_loss_fdum, comm_fdum = FedDU_modify(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['FedDU'] = test_acc_fdum 
    results_train_loss['FedDU'] = train_loss_fdum
    results_comm_vs_acc['FedDU'] = comm_fdum

    fedmut_model_instance = copy.deepcopy(init_model) 
    test_acc_fm, train_loss_fm, comm_fm = FedMut(fedmut_model_instance, global_round, eta, K, M)
    results_test_acc['FedMut'] = test_acc_fm
    results_train_loss['FedMut'] = train_loss_fm
    results_comm_vs_acc['FedMut'] = comm_fm
    del fedmut_model_instance

    # clgmut2_model_instance = copy.deepcopy(init_model)
    # test_acc_clgm2, train_loss_clgm2, comm_clgm2 = CLG_Mut_2(clgmut2_model_instance, global_round, eta, gamma, K, E, M)
    # results_test_acc['CLG_Mut_2'] = test_acc_clgm2
    # results_train_loss['CLG_Mut_2'] = train_loss_clgm2
    # results_comm_vs_acc['CLG_Mut_2'] = comm_clgm2
    # del clgmut2_model_instance
    
    fedatmv_model_instance = copy.deepcopy(init_model)
    test_acc_fatmv, train_loss_fatmv, comm_fatmv = FedATMV(fedatmv_model_instance, global_round, eta, gamma, K, E, M)
    results_test_acc['FedATMV'] = test_acc_fatmv
    results_train_loss['FedATMV'] = train_loss_fatmv
    results_comm_vs_acc['FedATMV'] = comm_fatmv
    del fedatmv_model_instance
    
    print("\n--- Accuracy & Loss at specific rounds/final (Original Metrics) ---")
    for algo_name in results_test_acc: 
        if len(results_test_acc[algo_name]) >= 20:
            print(f"{algo_name} - Round 20 Test Acc: {results_test_acc[algo_name][19]:.2f}%, Round 20 Train Loss: {results_train_loss[algo_name][19]:.4f}")
        if results_test_acc[algo_name]: 
             print(f"{algo_name} - Final Test Acc: {results_test_acc[algo_name][-1]:.2f}%, Final Train Loss: {results_train_loss[algo_name][-1]:.4f}")
    
    comm_output_dir = "./output/communication"
    os.makedirs(comm_output_dir, exist_ok=True)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

    plt.figure(figsize=(12, 8))
    for algo_name, comm_data_list in results_comm_vs_acc.items(): 
        if not comm_data_list: continue 
        # Convert bytes to Megabytes for better readability on the plot
        overheads_mb = [item['overhead'] / (1024 * 1024) for item in comm_data_list]
        accuracies = [item['accuracy'] for item in comm_data_list]
        plt.plot(overheads_mb, accuracies, label=algo_name, marker='o', markersize=2, linestyle='-')

    plt.xlabel('Cumulative Communication Overhead (MB)', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title(f'Test Accuracy vs. Communication Overhead ({dataset.upper()}-{origin_model.upper()})', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    comm_plot_filename = os.path.join(comm_output_dir, f'all_algos_comm_vs_acc_{dataset}_{origin_model}_{current_timestamp}.png')
    plt.savefig(comm_plot_filename)
    print(f"\nCommunication vs. Accuracy plot saved to: {comm_plot_filename}")
    plt.close()

    comm_data_filename = os.path.join(comm_output_dir, f'all_algos_comm_vs_acc_data_{dataset}_{origin_model}_{current_timestamp}.json')
    with open(comm_data_filename, 'w') as f:
        json.dump(results_comm_vs_acc, f, indent=2) # Save data with overhead in bytes
    print(f"Communication vs. Accuracy raw data saved to: {comm_data_filename}")

    output_main_dir = "./output" 
    os.makedirs(output_main_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for algo, acc in results_test_acc.items():
        if acc : plt.plot(range(1, len(acc) + 1), acc, label=algo) 
    plt.xlabel('Training Rounds', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title(f'Test Accuracy Comparison ({dataset}-{origin_model})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_main_dir, f'test_accuracy_{origin_model}_{dataset}_{current_timestamp}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    for algo, loss_vals in results_train_loss.items(): 
        if loss_vals: plt.plot(range(1, len(loss_vals) + 1), loss_vals, label=algo)
    plt.xlabel('Training Rounds', fontsize=14)
    plt.ylabel('Train Loss', fontsize=14)
    plt.title(f'Train Loss Comparison ({dataset}-{origin_model})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_main_dir, f'train_loss_{origin_model}_{dataset}_{current_timestamp}.png'))
    plt.close()

    return results_test_acc, results_train_loss 


if __name__ == '__main__':
    print(f"Starting run_once with dataset: {dataset}, model: {origin_model}, global_rounds: {global_round}")
    if initial_w is None:
        print("Error: initial_w is None. Data loading might have failed or was skipped.")
    else:
        returned_test_acc, returned_train_loss = run_once()
        print("\nrun_once execution complete.")