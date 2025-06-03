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
# v 18.0_comm_overhead
# - Added communication overhead tracking.
# - Modified CNNCifar and ResNetCifar10 to take num_classes.
# - Algorithms now return communication overhead vs accuracy.
# - run_once now plots and saves this data.

# %%
# import os # Already imported
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# %%
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

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

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
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
            print(f"Warning: Class {i} has insufficient samples. Requested {num_samples}, available {len(sub_data)}. Taking all available.")
            num_samples = len(sub_data)
        if num_samples == 0 and len(sub_data) > 0 : # if requested is 0 but available, take 0. If available is 0, then num_samples is 0.
             rand_indx = [] # No samples
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
            num_cls = int(len(cls_indices) * percentage) # Distribute server_total proportionally
            if percentage == 1.0 : num_cls = len(cls_indices) # ensure all data taken if 100%

            if num_cls > len(cls_indices): num_cls = len(cls_indices)
            if num_cls == 0 and len(cls_indices) > 0 and server_total > 0 : num_cls = 1 # ensure at least one if possible and server_total > 0

            sampled = np.random.choice(cls_indices, size=num_cls, replace=False) if len(cls_indices) > 0 and num_cls > 0 else []
            selected_indices.extend(sampled)
    elif mode == 'non-iid':
        classes_len = len(unique_classes_arr)
        prob_dist = np.random.dirichlet(np.repeat(dirichlet_alpha, classes_len))
        cls_sample_numbers = {}
        total_assigned = 0
        for i, cls_val in enumerate(unique_classes_arr):
            n_cls = int(prob_dist[i] * server_total)
            cls_sample_numbers[cls_val] = n_cls
            total_assigned += n_cls
        
        # Distribute remainder due to flooring
        diff = server_total - total_assigned
        if diff > 0:
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
    
    selected_indices = list(set(selected_indices)) # Remove duplicates if any from remainder distribution

    if server_fill and len(selected_indices) < server_total :
        shortfall = server_total - len(selected_indices)
        if shortfall > 0:
            remaining_pool = np.setdiff1d(np.arange(total_num), selected_indices, assume_unique=True)
            if shortfall > len(remaining_pool): shortfall = len(remaining_pool) # Can't pick more than available
            extra = np.random.choice(remaining_pool, shortfall, replace=False) if len(remaining_pool) > 0 else []
            selected_indices = np.concatenate([selected_indices, extra]) if len(extra) > 0 else np.array(selected_indices)
            
    selected_indices = np.array(selected_indices, dtype=int) # Ensure integer indices
    np.random.shuffle(selected_indices) # Shuffle at the end
    
    # Final check to ensure we don't exceed server_total if server_fill was aggressive or due to rounding
    if len(selected_indices) > server_total:
        selected_indices = selected_indices[:server_total]

    subset_images = images[selected_indices]
    subset_labels = labels[selected_indices]
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

    # Handle empty dataset case
    if len(dataset_val[0]) == 0:
        # print("Warning: Empty dataset provided to update_weights. Returning original weights and zero loss.")
        return model.state_dict(), 0.0, {}


    if origin_model == 'resnet' or origin_model == 'cnn' or origin_model == 'vgg':
        Tensor_set = TensorDataset(torch.Tensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device))
    elif origin_model == 'lstm':
        Tensor_set = TensorDataset(torch.LongTensor(dataset_val[0]).to(device), torch.Tensor(dataset_val[1]).long().to(device)) # Target for LSTM should be long for CrossEntropy
    
    data_loader = DataLoader(Tensor_set, batch_size=bc_size, shuffle=True)
    first_iter_gradient = None

    for iter_val in range(local_epoch): # iter_val
        batch_loss = []
        if not data_loader: # if dataloader is empty
            epoch_loss.append(0.0) # or handle as appropriate
            continue

        for batch_idx, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['output'], labels) # labels already long
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/images.shape[0] if images.shape[0] > 0 else 0.0)

            if iter_val == 0 and batch_idx == 0:
                first_iter_gradient = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        first_iter_gradient[name] = param.grad.clone()
                for name, module_val in model.named_modules(): # module_val
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
        w_diff[key] = (p_w[key] - n_w[key]) * lr_val
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

    for iter_val in range(local_epoch): # iter_val
        batch_loss = []
        if not data_loader:
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
        
        # Ensure c_i and c_s are not empty before proceeding
        if c_i and c_s:
            corrected_graident = weight_differences(c_i, c_s, learning_rate) # This seems to be delta_c_i in Scaffold
            orginal_model_weight = model.state_dict()
            # The Scaffold update is: w = w - lr * (grad - c_i + c_s)
            # The code seems to be doing: w_new = w_original - (c_i - c_s)*lr
            # This correction is applied *after* an epoch. Standard Scaffold applies correction per batch.
            # The provided code: corrected_model_weight = orginal_model_weight - (c_i - c_s)*lr
            # Let's assume the logic in weight_differences for (p_w[key] - n_w[key]) * lr is intentional.
            # If corrected_gradient = (c_s - c_i)*lr
            # Then corrected_model_weight = original_model_weight - corrected_gradient (if lr for weight_differences is 1)
            # This matches: w_new = w_old - (grad_local - c_i + c_global) * lr_local
            # Here, the correction seems to be applied differently.
            # For now, I will keep the logic as provided in the original script for Fed-C.
            # The `weight_differences` function calculates `(server_term - client_term) * some_factor`.
            # If `corrected_gradient = (c_s - c_i) * lr`, then
            # `corrected_model_weight = original_model_weight - corrected_gradient` (if the second lr is 1).
            # This looks like `w_t+1 = w_t - (c_i - c_s)*lr`.
            # Let's trace: `corrected_graident = (c_s - c_i) * learning_rate`
            # `corrected_model_weight = weight_differences(corrected_graident, orginal_model_weight, 1)`
            # `corrected_model_weight[key] = (orginal_model_weight[key] - corrected_graident[key]) * 1`
            # `corrected_model_weight[key] = orginal_model_weight[key] - (c_s[key] - c_i[key]) * learning_rate`
            # This is `w_new = w_old - (c_s - c_i) * lr_local`. This is one form of Scaffold-like update.
            corrected_model_weight = weight_differences(corrected_graident, model.state_dict(), 1) 
            model.load_state_dict(corrected_model_weight)
        elif not c_i and not c_s: # No correction if control variates are empty
            pass
        else: # Mismatched control variates
            print("Warning: Mismatched control variates in update_weights_correction. Skipping correction.")


    final_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
    return model.state_dict(), final_loss

# %%
def average_weights(w_list): # w_list
    if not w_list: return {}
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        if 'num_batches_tracked' in key:
            continue
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_list))
    return w_avg

# %%
# Baseline: server-only
def server_only(initial_w, global_round_val, gamma_val, E_val): # Renamed params
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
    test_acc_list = [] # test_acc
    train_loss_list = [] # train_loss
    comm_vs_acc_list = [] # New: For communication overhead
    cumulative_overhead = 0

    for round_idx in tqdm(range(global_round_val)): # round_idx
        update_server_w, round_loss_val, _ = update_weights(train_w, server_data, gamma_val, E_val) # round_loss_val
        train_w = update_server_w
        test_model.load_state_dict(train_w)
        train_loss_list.append(round_loss_val)
        
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication overhead for server-only is 0
        current_round_overhead = 0 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def fedavg(initial_w, global_round_val, eta_val, K_val, M_val): # Renamed params
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
    cumulative_overhead = 0
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], [] # local_loss_vals
        sampled_client_indices = random.sample(range(client_num), M_val) # sampled_client_indices
        for client_idx in sampled_client_indices: # client_idx
            if len(client_data[client_idx][0]) == 0: # Skip empty client dataset
                # print(f"Warning: Client {client_idx} has no data. Skipping for FedAvg round {round_idx}.")
                continue
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)

        if not local_weights: # All sampled clients were empty
            print(f"Warning: No clients contributed in FedAvg round {round_idx}. Using previous global model.")
            loss_avg = train_loss_list[-1] if train_loss_list else 0.0
        else:
            train_w = average_weights(local_weights)
            loss_avg = sum(local_loss_vals)/ len(local_loss_vals)
        
        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download, M clients upload
        current_round_overhead = 2 * M_val 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
            
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def hybridFL(initial_w, global_round_val, eta_val, K_val, E_val, M_val):
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
    cumulative_overhead = 0
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        sampled_client_indices = random.sample(range(client_num), M_val)

        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)

        # Server participates in training
        if len(server_data[0]) > 0:
            update_server_w, server_round_loss, _ = update_weights(train_w, server_data, eta_val, E_val) # Using eta_val and E_val for server
            local_weights.append(update_server_w)
            local_loss_vals.append(server_round_loss)
        
        if not local_weights:
            print(f"Warning: No clients/server contributed in HybridFL round {round_idx}. Using previous global model.")
            loss_avg = train_loss_list[-1] if train_loss_list else 0.0
        else:
            train_w = average_weights(local_weights)
            loss_avg = sum(local_loss_vals) / len(local_loss_vals)

        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
    
        # Communication: M clients download, M clients upload. Server training is local.
        current_round_overhead = 2 * M_val 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

def Data_Sharing(initial_w, global_round_val, eta_val, K_val, M_val, share_ratio=1):
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == 'cnn':
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'lstm':
        test_model = CharLSTM().to(device)
    else: raise ValueError("Unknown origin_model")

    local_datasets = client_data_mixed # Uses pre-mixed data
    w_global = copy.deepcopy(initial_w)
    all_test_acc_list = [] # all_test_acc
    all_train_loss_list = [] # all_train_loss
    comm_vs_acc_list = []
    cumulative_overhead = 0

    # Initial data broadcast cost is not iteratively added here for the plot's x-axis,
    # but it's a significant one-time cost for this method.
    # Iterative cost is like FedAvg.

    for r_idx in tqdm(range(global_round_val)): # r_idx
        selected_indices = np.random.choice(range(client_num), M_val, replace=False) # selected
        local_ws, local_ls = [], []

        for cid in selected_indices:
            if len(local_datasets[cid][0]) == 0: continue
            w_updated, loss_val, _ = update_weights(w_global, local_datasets[cid], eta_val, K_val) # loss_val
            local_ws.append(w_updated)
            local_ls.append(loss_val)

        if not local_ws:
            print(f"Warning: No clients contributed in Data_Sharing round {r_idx}. Using previous global model.")
            avg_loss = all_train_loss_list[-1] if all_train_loss_list else 0.0
        else:
            w_global = average_weights(local_ws)
            avg_loss = sum(local_ls) / len(local_ls)
        
        all_train_loss_list.append(avg_loss)
        test_model.load_state_dict(w_global)
        current_test_acc = test_inference(test_model, test_dataset)
        all_test_acc_list.append(current_test_acc)

        # Communication: M clients download, M clients upload (after initial broadcast)
        current_round_overhead = 2 * M_val 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

    return all_test_acc_list, all_train_loss_list, comm_vs_acc_list

def build_mixed_client_data(client_data_val, server_data_val, share_ratio=1.0, seed_val=None): # Renamed params
    if seed_val is not None:
        np.random.seed(seed_val)

    s_imgs, s_lbls = server_data_val
    s_imgs_arr = np.array(s_imgs) # s_imgs_arr
    s_lbls_arr = np.array(s_lbls) # s_lbls_arr

    if share_ratio < 1.0 and len(s_imgs_arr) > 0 :
        sel_idx = np.random.choice(len(s_imgs_arr),
                                   size=int(len(s_imgs_arr) * share_ratio),
                                   replace=False).astype(int)
        s_imgs_arr = s_imgs_arr[sel_idx]
        s_lbls_arr = s_lbls_arr[sel_idx]

    mixed_clients = []
    for imgs_c, lbls_c in client_data_val: # imgs_c, lbls_c
        # Ensure client data are numpy arrays before concatenation
        imgs_c_arr = np.array(imgs_c)
        lbls_c_arr = np.array(lbls_c)

        if len(s_imgs_arr) > 0: # Only concatenate if server data to share exists
            new_imgs = np.concatenate([imgs_c_arr, s_imgs_arr], axis=0) if len(imgs_c_arr) > 0 else s_imgs_arr
            new_lbls = np.concatenate([lbls_c_arr, s_lbls_arr], axis=0) if len(lbls_c_arr) > 0 else s_lbls_arr
        else: # No server data to share
            new_imgs = imgs_c_arr
            new_lbls = lbls_c_arr
        mixed_clients.append((new_imgs, new_lbls))
    return mixed_clients

# %%
def CLG_SGD(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
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
    cumulative_overhead = 0
    
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
        
        if local_weights: # if any client trained
             train_w = average_weights(local_weights)
        # else train_w remains from previous round or initial_w if first round

        # Server side local training
        if len(server_data[0]) > 0:
            update_server_w, server_loss, _ = update_weights(train_w, server_data, gamma_val, E_val) # server_loss
            train_w = update_server_w
            local_loss_vals.append(server_loss) # Include server loss in average if server trained
        
        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download, M clients upload. Server training is local.
        current_round_overhead = 2 * active_clients_this_round # Only count active clients
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

def Fed_C(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
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
    cumulative_overhead = 0

    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        g_i_list = []
        
        g_s = {}
        if len(server_data[0]) > 0:
             _, _, g_s = update_weights(train_w, server_data, gamma_val, 1) # Server gradient for 1 epoch

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round = 0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            _, _, g_i = update_weights(train_w, client_data[client_idx], eta_val, 1) # Client gradient for 1 epoch
            g_i_list.append(g_i if g_i else {}) # Append empty dict if no gradient

        client_iter = 0 # To iterate through g_i_list correctly
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            current_g_i = g_i_list[client_iter] if client_iter < len(g_i_list) else {}
            client_iter +=1
            
            update_client_w, client_round_loss = update_weights_correction(train_w, client_data[client_idx], eta_val, K_val, current_g_i, g_s)
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)
        
        if local_weights:
            train_w = average_weights(local_weights)
        
        if len(server_data[0]) > 0:
            update_server_w, server_loss, _ = update_weights(train_w, server_data, gamma_val, E_val)
            train_w = update_server_w
            local_loss_vals.append(server_loss)

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download model, M clients download g_s, M clients upload model
        current_round_overhead = 3 * active_clients_this_round 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

    return test_acc_list, train_loss_list, comm_vs_acc_list

def Fed_S(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
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
    cumulative_overhead = 0

    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals = [], []
        g_i_list = []
        
        g_s = {}
        if len(server_data[0]) > 0:
            _, _, g_s = update_weights(train_w, server_data, gamma_val, 1)

        sampled_client_indices = random.sample(range(client_num), M_val)
        active_clients_this_round = 0
        for client_idx in sampled_client_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round +=1
            # Client local training (standard update_weights)
            update_client_w, client_round_loss, g_i = update_weights(train_w, client_data[client_idx], eta_val, K_val) # K_val epochs for Fed-S client training
            local_weights.append(update_client_w)
            local_loss_vals.append(client_round_loss)
            # For Fed-S, g_i is the gradient from the *first step* of local training.
            # The current update_weights returns gradient of first iter of first epoch.
            # We need to ensure update_weights is called to get g_i based on *train_w* (global model) for 1 step.
            _, _, g_i_for_correction = update_weights(train_w, client_data[client_idx], eta_val, 1) # Get g_i for correction
            g_i_list.append(g_i_for_correction if g_i_for_correction else {})


        if local_weights:
            train_w_aggregated = average_weights(local_weights) # train_w_aggregated
        else: # No clients trained
            train_w_aggregated = copy.deepcopy(train_w) # Use previous global model

        # Server aggregation correction for Fed-S
        if g_i_list and g_s : # Ensure gradients are available
            g_i_average = average_weights([g for g in g_i_list if g]) # Filter out empty dicts
            if g_i_average: # If average could be computed
                # The original Fed-S paper uses w_t+1 = w_agg - η_g * K * (grad_S_avg - grad_C_avg)
                # Here, it seems to be: correction_g = (g_s - g_i_average) * K*eta
                # train_w = train_w_agg - correction_g
                correction_g = weight_differences(g_i_average, g_s, K_val * eta_val) # (g_s - g_i_average) * K*eta
                train_w = weight_differences(correction_g, copy.deepcopy(train_w_aggregated), 1) # train_w_agg - correction_g
            else: # No client gradients to average
                train_w = train_w_aggregated
        else: # No server gradient or no client gradients
            train_w = train_w_aggregated

        if len(server_data[0]) > 0:
            update_server_w, server_loss, _ = update_weights(train_w, server_data, gamma_val, E_val)
            train_w = update_server_w
            local_loss_vals.append(server_loss)

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        test_model.load_state_dict(train_w)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)
        
        # Communication: M clients download model, M clients upload model, M clients upload g_i
        current_round_overhead = 3 * active_clients_this_round 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def KL_divergence(p1, p2):
    d = 0
    for i in range(len(p1)):
        if p2[i] == 0 or p1[i] == 0: continue
        d += p1[i] * math.log(p1[i]/p2[i], 2)
    return d

def calculate_js_divergence(p1, p2):
    p3 = []
    for i in range(len(p1)): p3.append((p1[i] + p2[i])/2)
    return KL_divergence(p1, p3)/2 + KL_divergence(p2, p3)/2

def ratio_combine(w1, w2, ratio=0):
    w = copy.deepcopy(w1)
    for key in w.keys():
        if 'num_batches_tracked' in key: continue
        w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w

def FedDU_modify(initial_w, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
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
    cumulative_overhead = 0
    
    server_min_iter = 0 # server_min -> server_min_iter
    
    all_client_labels_list = [] # all_client_labels_list
    for i in range(client_num): all_client_labels_list.extend(client_data[i][1])
    all_client_labels_arr = np.array(all_client_labels_list) # all_client_labels_arr
    
    unique_classes_arr, client_counts = np.unique(all_client_labels_arr, return_counts=True) # client_counts
    # Ensure P covers all classes up to num_classes, even if some are not in all_client_labels_arr
    P_dist = [0.0] * num_classes # P_dist
    for cls_val, count_val in zip(unique_classes_arr, client_counts):
        if cls_val < num_classes: P_dist[cls_val] = count_val / len(all_client_labels_arr) if len(all_client_labels_arr) > 0 else 0.0
    
    server_labels_arr = np.array(server_data[1]) # server_labels_arr
    n_0 = len(server_labels_arr)
    
    P_0_dist = [0.0] * num_classes # P_0_dist
    if n_0 > 0:
        unique_server_cls, server_cls_counts = np.unique(server_labels_arr, return_counts=True)
        for cls_val, count_val in zip(unique_server_cls, server_cls_counts):
             if cls_val < num_classes: P_0_dist[cls_val] = count_val / n_0
    
    D_P_0 = calculate_js_divergence(P_0_dist, P_dist)
    
    # print(f"FedDU initial: n_0={n_0}, D_P_0={D_P_0:.6f}, decay_rate={decay_rate}")
    
    for round_idx in tqdm(range(global_round_val)):
        local_weights, local_loss_vals_iter = [], [] # local_loss_vals_iter
        sampled_clients_indices = random.sample(range(client_num), M_val) # sampled_clients_indices
        
        num_current_samples = 0 # num_current_samples
        active_clients_this_round = 0
        for client_idx in sampled_clients_indices:
            if len(client_data[client_idx][0]) == 0: continue
            active_clients_this_round+=1
            num_current_samples += len(client_data[client_idx][0])
            update_client_w, client_round_loss, _ = update_weights(train_w, client_data[client_idx], eta_val, K_val)
            local_weights.append(update_client_w)
            local_loss_vals_iter.append(client_round_loss)
        
        if not local_weights: # All sampled clients were empty
            w_t_half = copy.deepcopy(train_w) # Use previous global model
            # print(f"Warning: No clients contributed in FedDU round {round_idx}.")
        else:
            w_t_half = average_weights(local_weights)
        
        selected_client_labels_list = [] # selected_client_labels_list
        for client_idx in sampled_clients_indices: selected_client_labels_list.extend(client_data[client_idx][1])
        selected_client_labels_arr = np.array(selected_client_labels_list) # selected_client_labels_arr
        
        P_t_prime_dist = [0.0] * num_classes # P_t_prime_dist
        if len(selected_client_labels_arr) > 0:
            unique_selected_cls, selected_cls_counts = np.unique(selected_client_labels_arr, return_counts=True)
            for cls_val, count_val in zip(unique_selected_cls, selected_cls_counts):
                if cls_val < num_classes: P_t_prime_dist[cls_val] = count_val / len(selected_client_labels_arr)
        
        D_P_t_prime = calculate_js_divergence(P_t_prime_dist, P_dist)
        
        test_model.load_state_dict(w_t_half)
        acc_t = test_inference(test_model, test_dataset) / 100.0
        
        avg_iter_val = (num_current_samples * K_val) / (M_val * bc_size) if M_val > 0 and bc_size > 0 else 0 # avg_iter_val
        epsilon = 1e-10
        alpha_dyn = (1 - acc_t) * (n_0 * D_P_t_prime) / (n_0 * D_P_t_prime + num_current_samples * D_P_0 + epsilon) # alpha_dyn
        alpha_dyn = alpha_dyn * (decay_rate ** round_idx) * du_C
        
        server_iter_count = max(server_min_iter, int(alpha_dyn * avg_iter_val)) # server_iter_count
        
        current_round_loss_server = 0.0 # current_round_loss_server
        if alpha_dyn > 0.001 and n_0 > 0: # Server update only if alpha is significant and server has data
            actual_server_iter = math.ceil(n_0 / bc_size) * E_val if bc_size > 0 else 0 # actual_server_iter
            effective_server_iter = min(actual_server_iter, server_iter_count) # effective_server_iter
            
            if effective_server_iter > 0 : # Only train if effective_server_iter > 0
                update_server_w, current_round_loss_server, _ = update_weights(copy.deepcopy(w_t_half), server_data, gamma_val, E_val) # E_val epochs for server
                local_loss_vals_iter.append(current_round_loss_server)
                train_w = ratio_combine(w_t_half, update_server_w, alpha_dyn) # Use alpha_dyn for combining
            else: # effective_server_iter is 0
                train_w = copy.deepcopy(w_t_half)
                # No server training, so server loss is not added to local_loss_vals_iter for this path
        else:
            train_w = copy.deepcopy(w_t_half)
            # No server training, if server has data, calculate its loss on w_t_half for consistent loss reporting
            if n_0 > 0:
                _, current_round_loss_server, _ = update_weights(copy.deepcopy(w_t_half), server_data, gamma_val, E_val) # Calculate loss but don't update train_w
                local_loss_vals_iter.append(current_round_loss_server)

        test_model.load_state_dict(train_w)
        loss_avg = sum(local_loss_vals_iter) / len(local_loss_vals_iter) if local_loss_vals_iter else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download, M clients upload
        current_round_overhead = 2 * active_clients_this_round 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
# FedMut related functions (Aggregation, FedSub, delta_rank, mutation_spread)
def Aggregation(w_list, lens_list): # w_list, lens_list
    w_avg = None
    if not w_list: return {}
    
    if lens_list is None:
        total_count = len(w_list)
        lens_list = [1.0] * len(w_list)
    else:
        total_count = sum(lens_list)
        if total_count == 0: # Avoid division by zero if all lens are 0
            return copy.deepcopy(w_list[0]) if w_list else {}


    for i in range(0, len(w_list)):
        if i == 0:
            w_avg = copy.deepcopy(w_list[0])
            for k_key in w_avg.keys(): # k_key
                w_avg[k_key] = w_list[i][k_key] * lens_list[i]
        else:
            for k_key in w_avg.keys():
                w_avg[k_key] += w_list[i][k_key] * lens_list[i]

    for k_key in w_avg.keys():
        w_avg[k_key] = torch.div(w_avg[k_key], total_count)
    return w_avg

def FedSub(w_curr, w_prev, weight_val): # w_curr, w_prev, weight_val
    w_sub = copy.deepcopy(w_curr)
    for k_key in w_sub.keys():
        if 'num_batches_tracked' in k_key: # num_batches_tracked should not be subtracted
            w_sub[k_key] = w_curr[k_key] 
            continue
        w_sub[k_key] = (w_curr[k_key] - w_prev[k_key]) * weight_val
    return w_sub

def delta_rank(delta_dict):
    cnt = 0
    dict_a_list = [] # dict_a_list
    for p_key in delta_dict.keys(): # p_key
        if 'num_batches_tracked' in p_key: continue # Skip non-parameter tensors
        a_tensor = delta_dict[p_key] # a_tensor
        if not torch.is_tensor(a_tensor): continue # Skip if not a tensor
        dict_a_list.append(a_tensor.view(-1).float()) # Ensure float for norm
    if not dict_a_list: return 0.0
    
    dict_a_combined = torch.cat(dict_a_list, dim=0) # dict_a_combined
    s_norm = torch.norm(dict_a_combined, p=2, dim=0) # s_norm
    return s_norm.item()


def mutation_spread(iter_val, w_glob_val, m_clients, w_delta_val, alpha_mut): # iter_val, w_glob_val, m_clients, w_delta_val, alpha_mut
    w_locals_new_list = [] # w_locals_new_list
    ctrl_cmd_list_outer = [] # ctrl_cmd_list_outer
    
    # Beta_t in FedMut paper
    ctrl_rate_val = mut_acc_rate * (1.0 - min(iter_val * 1.0 / mut_bound if mut_bound > 0 else 1.0 , 1.0)) # ctrl_rate_val

    for k_key in w_glob_val.keys():
        if 'num_batches_tracked' in k_key : continue # Don't mutate batchnorm tracking
        ctrl_list_inner = [] # ctrl_list_inner
        for _ in range(0, int(m_clients / 2)): # i_client_pair
            ctrl_rand = random.random() # ctrl_rand
            if ctrl_rand > 0.5:
                ctrl_list_inner.append(1.0)
                ctrl_list_inner.append(1.0 * (-1.0 + ctrl_rate_val))
            else:
                ctrl_list_inner.append(1.0 * (-1.0 + ctrl_rate_val))
                ctrl_list_inner.append(1.0)
        if m_clients % 2 == 1: # Handle odd number of clients
             ctrl_list_inner.append(0.0) # The last client gets no mutation or a random small one
             # Original FedMut paper implies the last one might not participate in this symmetric mutation.
             # Or assign it a random value. For simplicity, 0.0 means it gets the global model.
             # Let's follow the provided code's structure: it will be skipped in the loop below.

        random.shuffle(ctrl_list_inner)
        ctrl_cmd_list_outer.append(ctrl_list_inner)
    
    client_counter = 0 # client_counter
    for j_client in range(m_clients): # j_client
        w_sub_mutated = copy.deepcopy(w_glob_val) # w_sub_mutated
        # The original code has: if not (cnt == m - 1 and m % 2 == 1):
        # This means the last client in an odd setup does not get mutated.
        if not (client_counter == m_clients - 1 and m_clients % 2 == 1):
            param_idx = 0 # param_idx
            for k_key in w_sub_mutated.keys():
                if 'num_batches_tracked' in k_key : continue
                if param_idx < len(ctrl_cmd_list_outer) and j_client < len(ctrl_cmd_list_outer[param_idx]):
                     # Ensure w_delta_val also doesn't have num_batches_tracked or handle it
                     if k_key in w_delta_val:
                        w_sub_mutated[k_key] = w_sub_mutated[k_key] + w_delta_val[k_key] * ctrl_cmd_list_outer[param_idx][j_client] * alpha_mut
                param_idx += 1
        client_counter += 1
        w_locals_new_list.append(w_sub_mutated)
    return w_locals_new_list

def FedMut(net_glob_model, global_round_val, eta_val, K_val, M_val): # net_glob_model
    net_glob_model.train()
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar(num_classes_arg=num_classes).to(device)
    elif origin_model == 'vgg':
        test_model = VGG16(num_classes, 3).to(device)
    else: raise NotImplementedError(f"Unknown origin_model: {origin_model}")
        
    # train_w = copy.deepcopy(net_glob_model.state_dict()) # Not used, w_old is used
    test_acc_list = []
    train_loss_list = []
    comm_vs_acc_list = []
    cumulative_overhead = 0
    
    w_locals_list = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)] # w_locals_list
    max_rank_val = 0 # max_rank_val
    
    for round_idx in tqdm(range(global_round_val)):
        w_old_global = copy.deepcopy(net_glob_model.state_dict()) # w_old_global
        local_loss_vals = []
        
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False) # idxs_users_sampled
        active_clients_this_round = 0
        
        # Client training loop
        for i_local, client_actual_idx in enumerate(idxs_users_sampled): # i_local, client_actual_idx
            if len(client_data[client_actual_idx][0]) == 0: 
                # If client has no data, its w_locals[i_local] remains as is (from previous mutation or init_model)
                # It won't contribute to aggregation if it doesn't train.
                # For simplicity, we'll assume it contributes its current (possibly mutated) model.
                # Or, more realistically, it shouldn't be part of w_locals for aggregation if it didn't train.
                # Let's only update w_locals[i_local] if training happens.
                continue 
            active_clients_this_round +=1
            
            # Load personalized model for the client
            # The current net_glob_model is actually not used for loading client state here.
            # Instead, w_locals_list[i_local] is the state for the client.
            # So, the update_weights should take w_locals_list[i_local]
            current_client_model_state = w_locals_list[i_local]
            
            updated_client_w, client_round_loss, _ = update_weights(current_client_model_state, 
                                                                    client_data[client_actual_idx], 
                                                                    eta_val, K_val)
            w_locals_list[i_local] = copy.deepcopy(updated_client_w) # Update the client's model in the list
            local_loss_vals.append(client_round_loss)

        # Global Model Generation (Aggregation)
        # Only aggregate from clients that actually trained or have valid models
        # Assuming all M_val clients in w_locals_list are part of aggregation
        w_aggregated = Aggregation(w_locals_list, None)  # w_aggregated
        
        if not w_aggregated: # if aggregation failed (e.g. all clients had no data)
            w_aggregated = copy.deepcopy(w_old_global) # Revert to old global model
            # print(f"Warning: FedMut aggregation failed in round {round_idx}. Using previous global model.")

        net_glob_model.load_state_dict(w_aggregated) # Load aggregated weights into the global model object
        
        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(w_aggregated)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download (personalized models), M clients upload
        current_round_overhead = 2 * active_clients_this_round # Based on active clients
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

        # Mutation
        w_delta_mutation = FedSub(w_aggregated, w_old_global, 1.0) # w_delta_mutation
        rank_val = delta_rank(w_delta_mutation) # rank_val
        if rank_val > max_rank_val: max_rank_val = rank_val
        
        alpha_for_mutation = radius # alpha_for_mutation (using global `radius`)
        w_locals_list = mutation_spread(round_idx, w_aggregated, M_val, w_delta_mutation, alpha_for_mutation)

    return test_acc_list, train_loss_list, comm_vs_acc_list   

# %%
def CLG_Mut_2(net_glob_model, global_round_val, eta_val, gamma_val, K_val, E_val, M_val):
    net_glob_model.train()
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
    cumulative_overhead = 0
    
    w_locals_list = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)]
    max_rank_val = 0
    # w_old_global = copy.deepcopy(net_glob_model.state_dict()) # Initialize w_old_global before the loop

    for round_idx in tqdm(range(global_round_val)):
        w_old_global_round = copy.deepcopy(net_glob_model.state_dict()) # Model state before any updates this round
        local_loss_vals = []
        
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False)
        active_clients_this_round = 0

        # Client training loop
        for i_local, client_actual_idx in enumerate(idxs_users_sampled):
            if len(client_data[client_actual_idx][0]) == 0: continue
            active_clients_this_round +=1
            current_client_model_state = w_locals_list[i_local] # Use personalized model
            updated_client_w, client_round_loss, _ = update_weights(current_client_model_state, 
                                                                    client_data[client_actual_idx], 
                                                                    eta_val, K_val)
            w_locals_list[i_local] = copy.deepcopy(updated_client_w)
            local_loss_vals.append(client_round_loss)

        w_aggregated_clients = Aggregation(w_locals_list, None) if active_clients_this_round > 0 else copy.deepcopy(w_old_global_round)
        
        # Server side local training
        w_after_server_train = w_aggregated_clients # Initialize with client aggregation
        if len(server_data[0]) > 0:
            w_after_server_train, server_loss, _ = update_weights(w_aggregated_clients, server_data, gamma_val, E_val)
            local_loss_vals.append(server_loss)
        
        net_glob_model.load_state_dict(w_after_server_train) # Update global model object

        loss_avg = sum(local_loss_vals)/ len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(w_after_server_train)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download (personalized), M clients upload. Server local.
        current_round_overhead = 2 * active_clients_this_round
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})

        # Mutation based on the direction from w_old_global_round to w_after_server_train
        w_delta_mutation = FedSub(w_after_server_train, w_old_global_round, 1.0)
        rank_val = delta_rank(w_delta_mutation)
        if rank_val > max_rank_val: max_rank_val = rank_val
        
        alpha_for_mutation = radius 
        w_locals_list = mutation_spread(round_idx, w_after_server_train, M_val, w_delta_mutation, alpha_for_mutation)

    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
def FedATMV(net_glob_model, global_round_val, eta_val, gamma_val, K_val, E_val, M_val, lambda_val_fedatmv=1): # lambda_val_fedatmv
    net_glob_model.train()
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
    cumulative_overhead = 0
    
    w_locals_list = [copy.deepcopy(net_glob_model.state_dict()) for _ in range(M_val)]
    max_rank_val = 0
    # w_old_global = copy.deepcopy(net_glob_model.state_dict()) # Initialize before loop

    # For FedDU part of FedATMV
    all_client_labels_list = []
    for i in range(client_num): all_client_labels_list.extend(client_data[i][1])
    all_client_labels_arr = np.array(all_client_labels_list)
    
    unique_classes_arr_fedatmv, client_counts_fedatmv = np.unique(all_client_labels_arr, return_counts=True) # _fedatmv suffix
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
    
    # print(f"FedATMV initial: n_0={n_0_fedatmv}, D_P_0={D_P_0_fedatmv:.6f}, radius={radius}")
    alpha_history_fedatmv, improvement_history_fedatmv = [], [] # _fedatmv suffix
    acc_prev_fedatmv = 0.0 # _fedatmv suffix
    
    for round_idx in tqdm(range(global_round_val)):
        w_old_global_round = copy.deepcopy(net_glob_model.state_dict())
        local_loss_vals = []
        
        idxs_users_sampled = np.random.choice(range(client_num), M_val, replace=False)
        selected_client_labels_list_fedatmv = [] # _fedatmv suffix
        num_current_samples_fedatmv = 0 # _fedatmv suffix
        active_clients_this_round = 0

        for i_local, client_actual_idx in enumerate(idxs_users_sampled):
            if len(client_data[client_actual_idx][0]) == 0: continue
            active_clients_this_round +=1
            current_client_model_state = w_locals_list[i_local]
            updated_client_w, client_round_loss, _ = update_weights(current_client_model_state, 
                                                                    client_data[client_actual_idx], 
                                                                    eta_val, K_val)
            w_locals_list[i_local] = copy.deepcopy(updated_client_w)
            local_loss_vals.append(client_round_loss)
            selected_client_labels_list_fedatmv.extend(client_data[client_actual_idx][1])
            num_current_samples_fedatmv += len(client_data[client_actual_idx][0])

        w_aggregated_clients = Aggregation(w_locals_list, None) if active_clients_this_round > 0 else copy.deepcopy(w_old_global_round)
        net_glob_model.load_state_dict(w_aggregated_clients) # Tentatively update global model with client aggregation
        
        # FedDU-like alpha calculation
        selected_client_labels_arr_fedatmv = np.array(selected_client_labels_list_fedatmv)
        P_t_prime_dist_fedatmv = [0.0] * num_classes
        if len(selected_client_labels_arr_fedatmv) > 0:
            unique_selected_cls_fedatmv, selected_cls_counts_fedatmv = np.unique(selected_client_labels_arr_fedatmv, return_counts=True)
            for cls_val, count_val in zip(unique_selected_cls_fedatmv, selected_cls_counts_fedatmv):
                 if cls_val < num_classes: P_t_prime_dist_fedatmv[cls_val] = count_val / len(selected_client_labels_arr_fedatmv)
        
        D_P_t_prime_fedatmv = calculate_js_divergence(P_t_prime_dist_fedatmv, P_dist_fedatmv)
        
        test_model.load_state_dict(w_aggregated_clients) # Test accuracy of client aggregated model
        acc_t_fedatmv = test_inference(test_model, test_dataset) / 100.0
        
        epsilon_fedatmv = 1e-10 # _fedatmv suffix
        r_data_fedatmv = n_0_fedatmv / (n_0_fedatmv + num_current_samples_fedatmv + epsilon_fedatmv) if (n_0_fedatmv + num_current_samples_fedatmv + epsilon_fedatmv) !=0 else 0
        r_noniid_fedatmv = D_P_t_prime_fedatmv / (D_P_t_prime_fedatmv + D_P_0_fedatmv + epsilon_fedatmv) if (D_P_t_prime_fedatmv + D_P_0_fedatmv + epsilon_fedatmv) !=0 else 0
        
        improvement_fedatmv = 0.0
        if round_idx > 0 : # acc_prev_fedatmv is from previous round
            improvement_fedatmv = max(0.0, acc_prev_fedatmv - acc_t_fedatmv) / (acc_prev_fedatmv + epsilon_fedatmv) if (acc_prev_fedatmv + epsilon_fedatmv) !=0 else 0
        
        min_alpha_fedatmv, max_alpha_fedatmv = 0.001, 1.0 # _fedatmv suffix
        alpha_new_fedatmv = du_C * (1 - acc_t_fedatmv) * r_data_fedatmv * r_noniid_fedatmv + lambda_val_fedatmv * improvement_fedatmv
        alpha_new_fedatmv = max(min_alpha_fedatmv, min(max_alpha_fedatmv, alpha_new_fedatmv))
        
        alpha_history_fedatmv.append(alpha_new_fedatmv)
        improvement_history_fedatmv.append(improvement_fedatmv)
        acc_prev_fedatmv = acc_t_fedatmv # Update for next round

        final_model_state = w_aggregated_clients # Default to client aggregated model
        if alpha_new_fedatmv > 0.001 and n_0_fedatmv > 0:
            update_server_w, server_loss, _ = update_weights(copy.deepcopy(w_aggregated_clients), server_data, gamma_val, E_val)
            local_loss_vals.append(server_loss)
            final_model_state = ratio_combine(w_aggregated_clients, update_server_w, alpha_new_fedatmv)
        elif n_0_fedatmv > 0: # alpha too small, but server has data, so calculate its loss for reporting
             _, server_loss, _ = update_weights(copy.deepcopy(w_aggregated_clients), server_data, gamma_val, E_val)
             local_loss_vals.append(server_loss)


        net_glob_model.load_state_dict(final_model_state) # Final update to global model object for this round
        
        loss_avg = sum(local_loss_vals) / len(local_loss_vals) if local_loss_vals else (train_loss_list[-1] if train_loss_list else 0.0)
        train_loss_list.append(loss_avg)
        
        test_model.load_state_dict(final_model_state)
        current_test_acc = test_inference(test_model, test_dataset)
        test_acc_list.append(current_test_acc)

        # Communication: M clients download (personalized), M clients upload. Server local.
        current_round_overhead = 2 * active_clients_this_round 
        cumulative_overhead += current_round_overhead
        comm_vs_acc_list.append({'overhead': cumulative_overhead, 'accuracy': current_test_acc})
        
        w_delta_mutation = FedSub(final_model_state, w_old_global_round, 1.0)
        rank_val = delta_rank(w_delta_mutation)
        if rank_val > max_rank_val: max_rank_val = rank_val
            
        tmp_radius_fedatmv = radius * (1 + scal_ratio * alpha_new_fedatmv) # tmp_radius_fedatmv
        w_locals_list = mutation_spread(round_idx, final_model_state, M_val, w_delta_mutation, tmp_radius_fedatmv)
          
    # Plotting alpha and improvement for FedATMV specifically
    # This plotting is specific to FedATMV, so it's inside its function.
    # The main run_once will plot the comm_vs_acc for all algos.
    # timestamp_fedatmv = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # _fedatmv suffix
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, global_round_val + 1), alpha_history_fedatmv, label="alpha_new (FedATMV)", marker='o')
    # plt.plot(range(1, global_round_val + 1), improvement_history_fedatmv, label="improvement (FedATMV)", marker='x')
    # plt.xlabel("Global Rounds")
    # plt.ylabel("Value")
    # plt.title("FedATMV: Alpha_new and Improvement vs Global Rounds")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # fedatmv_plot_dir = "./output/fedatmv_internals" # _fedatmv suffix
    # os.makedirs(fedatmv_plot_dir, exist_ok=True)
    # plt.savefig(os.path.join(fedatmv_plot_dir, f'fedatmv_alpha_improvement_{origin_model}_{du_C}_{lambda_val_fedatmv}_{timestamp_fedatmv}.png'))
    # plt.close() # Close the plot to free memory

    return test_acc_list, train_loss_list, comm_vs_acc_list

# %%
# Global parameters (ensure these are defined before use, e.g. in a main block or passed appropriately)
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
origin_model = 'resnet' # 'resnet', 'cnn', 'vgg', 'lstm'
dataset = 'cifar10' # 'cifar10', 'cifar100', 'shake'
momentum = 0.5
weight_decay = 0
bc_size = 50
test_bc_size = 128
num_classes = 10 # CIFAR10: 10, CIFAR100 (coarse): 20, Shake: 80
global_round = 100
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
test_dataset = None
client_data = []
server_data = [[], []] # [[images], [labels]]
init_model = None
initial_w = None
client_data_mixed = [] # For Data_Sharing

if dataset == 'cifar100':
    num_classes = 20 # Coarse CIFAR100
    cifar, test_dataset = CIFAR100() # Returns (train_data_tuple, test_dataset_obj)
    prob_dist = get_prob(non_iid, client_num, class_num_val=num_classes, iid_mode=is_iid) # prob_dist
    client_data = create_data_all_train(prob_dist, size_per_client, cifar, N_classes=num_classes)
    test_dataset.targets = sparse2coarse(test_dataset.targets) # This modifies test_dataset directly
    test_dataset.targets = np.array(test_dataset.targets).astype(int) # Ensure targets are int

    server_images, server_labels = select_server_subset(cifar, percentage=server_percentage, 
                                                        mode='iid' if server_iid else 'non-iid', 
                                                        dirichlet_alpha=server_dir)
    server_data = [server_images, server_labels]
    if origin_model == 'vgg':
        init_model = VGG16(num_classes, 3).to(device)
    elif origin_model == 'resnet': # Default to ResNet18 for CIFAR100 if not VGG
        init_model = ResNet18_cifar10(num_classes_arg=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model {origin_model} for CIFAR100")
    initial_w = copy.deepcopy(init_model.state_dict())

elif dataset =='shake':
    num_classes = 80 # Shakespeare specific
    train_dataset_obj = ShakeSpeare(True) # train_dataset_obj
    test_dataset = ShakeSpeare(False) # test_dataset is now a ShakeSpeare object

    total_shake_imgs, total_shake_labels = [],[] # total_shake_imgs, total_shake_labels
    for item_data, label_data in train_dataset_obj: # item_data, label_data
        total_shake_imgs.append(item_data.numpy()) # Assumes item is tensor
        total_shake_labels.append(label_data) # Assumes label is already appropriate type (e.g., int for index)
    
    # Convert labels to numpy array if they are tensors
    if isinstance(total_shake_labels[0], torch.Tensor):
        total_shake_labels = [lbl.item() for lbl in total_shake_labels]

    total_shake_imgs_arr = np.array(total_shake_imgs, dtype=object) # dtype=object for variable length sequences if not padded
    total_shake_labels_arr = np.array(total_shake_labels)

    shake_data_tuple = [total_shake_imgs_arr, total_shake_labels_arr] # shake_data_tuple
    dict_users_shake = train_dataset_obj.get_client_dic() # dict_users_shake
    client_num = len(dict_users_shake) # Update client_num based on dataset

    client_data = []
    for client_id_key in sorted(dict_users_shake.keys()): # client_id_key
        indices = np.array(list(dict_users_shake[client_id_key]), dtype=np.int64)
        # Filter out-of-bounds indices if any (robustness)
        indices = indices[indices < len(total_shake_imgs_arr)]
        client_images_val = total_shake_imgs_arr[indices] # client_images_val
        client_labels_val = total_shake_labels_arr[indices] # client_labels_val
        client_data.append((client_images_val, client_labels_val))

    server_images, server_labels = select_server_subset(shake_data_tuple, percentage=server_percentage,
                                                      mode='iid' if server_iid else 'non-iid', 
                                                      dirichlet_alpha=server_dir)
    server_data = [server_images, server_labels]
    if origin_model == 'lstm':
        init_model = CharLSTM().to(device)
    else:
        raise ValueError(f"Unsupported model {origin_model} for Shakespeare")
    initial_w = copy.deepcopy(init_model.state_dict())

elif dataset == "cifar10":
    num_classes = 10
    trans_cifar10_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_cifar10_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset_obj = torchvision.datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=trans_cifar10_train)
    test_dataset = torchvision.datasets.CIFAR10("./data/cifar10", train=False, download=True, transform=trans_cifar10_val)
    
    total_img_list, total_label_list = [], [] # total_img_list, total_label_list
    for img_i, label_i in train_dataset_obj: # img_i, label_i
        total_img_list.append(np.array(img_i))
        total_label_list.append(label_i)
    total_img_arr = np.array(total_img_list) # total_img_arr
    total_label_arr = np.array(total_label_list) # total_label_arr
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
client_data_mixed = build_mixed_client_data(client_data, server_data, share_ratio=1.0, seed_val=seed if random_fix else None)

# Print dataset stats (optional, can be verbose)
# ... (original printing logic can be added here if needed) ...
print(f"Dataset: {dataset}, Model: {origin_model}, Num clients: {client_num}, Num classes: {num_classes}")
print(f"Server data size: {len(server_data[0])}")
client_data_sizes = [len(cd[0]) for cd in client_data]
print(f"Client data sizes (first 5): {client_data_sizes[:5]}, Total client samples: {sum(client_data_sizes)}")


# %%
def run_once():
    results_test_acc = {}
    results_train_loss = {}
    results_comm_vs_acc = {} # New dictionary for communication overhead results

    # Ensure init_model is not None before proceeding
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
    results_test_acc['Fed-C'] = test_acc_fc # Changed from FedCLG-C to Fed-C for clarity
    results_train_loss['Fed-C'] = train_loss_fc
    results_comm_vs_acc['Fed-C'] = comm_fc

    # Fed_S
    test_acc_fs, train_loss_fs, comm_fs = Fed_S(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['Fed-S'] = test_acc_fs # Changed from FedCLG-S to Fed-S
    results_train_loss['Fed-S'] = train_loss_fs
    results_comm_vs_acc['Fed-S'] = comm_fs
    
    # FedDU_modify
    test_acc_fdum, train_loss_fdum, comm_fdum = FedDU_modify(initial_w, global_round, eta, gamma, K, E, M)
    results_test_acc['FedDU'] = test_acc_fdum # FedDU instead of FedDU_modify
    results_train_loss['FedDU'] = train_loss_fdum
    results_comm_vs_acc['FedDU'] = comm_fdum

    # FedMut
    # FedMut and CLG_Mut_2 take the model object, not just state_dict
    fedmut_model_instance = copy.deepcopy(init_model) # Get a fresh model instance
    test_acc_fm, train_loss_fm, comm_fm = FedMut(fedmut_model_instance, global_round, eta, K, M)
    results_test_acc['FedMut'] = test_acc_fm
    results_train_loss['FedMut'] = train_loss_fm
    results_comm_vs_acc['FedMut'] = comm_fm
    del fedmut_model_instance

    # # CLG_Mut_2
    # clgmut2_model_instance = copy.deepcopy(init_model)
    # test_acc_clgm2, train_loss_clgm2, comm_clgm2 = CLG_Mut_2(clgmut2_model_instance, global_round, eta, gamma, K, E, M)
    # results_test_acc['CLG_Mut_2'] = test_acc_clgm2
    # results_train_loss['CLG_Mut_2'] = train_loss_clgm2
    # results_comm_vs_acc['CLG_Mut_2'] = comm_clgm2
    # del clgmut2_model_instance
    
    # FedATMV
    fedatmv_model_instance = copy.deepcopy(init_model)
    test_acc_fatmv, train_loss_fatmv, comm_fatmv = FedATMV(fedatmv_model_instance, global_round, eta, gamma, K, E, M)
    results_test_acc['FedATMV'] = test_acc_fatmv
    results_train_loss['FedATMV'] = train_loss_fatmv
    results_comm_vs_acc['FedATMV'] = comm_fatmv
    del fedatmv_model_instance
    
    # --- Original Printing ---
    print("\n--- Accuracy & Loss at specific rounds/final (Original Metrics) ---")
    for algo_name in results_test_acc: # algo_name
        if len(results_test_acc[algo_name]) >= 20:
            print(f"{algo_name} - Round 20 Test Acc: {results_test_acc[algo_name][19]:.2f}%, Round 20 Train Loss: {results_train_loss[algo_name][19]:.4f}")
        if results_test_acc[algo_name]: # Check if list is not empty
             print(f"{algo_name} - Final Test Acc: {results_test_acc[algo_name][-1]:.2f}%, Final Train Loss: {results_train_loss[algo_name][-1]:.4f}")
    
    # --- Plotting and Saving Communication Overhead vs. Accuracy ---
    comm_output_dir = "./output/communication"
    os.makedirs(comm_output_dir, exist_ok=True)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # current_timestamp

    plt.figure(figsize=(12, 8))
    for algo_name, comm_data_list in results_comm_vs_acc.items(): # comm_data_list
        if not comm_data_list: continue # Skip if no data
        overheads = [item['overhead'] for item in comm_data_list]
        accuracies = [item['accuracy'] for item in comm_data_list]
        plt.plot(overheads, accuracies, label=algo_name, marker='o', markersize=3, linestyle='-')

    plt.xlabel('Cumulative Communication Overhead (Units)', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title(f'Test Accuracy vs. Communication Overhead ({dataset.upper()}-{origin_model.upper()})', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    comm_plot_filename = os.path.join(comm_output_dir, f'all_algos_comm_vs_acc_{dataset}_{origin_model}_{current_timestamp}.png')
    plt.savefig(comm_plot_filename)
    print(f"\nCommunication vs. Accuracy plot saved to: {comm_plot_filename}")
    # plt.show() # Optionally show plot if running interactively
    plt.close()

    # Save raw communication vs. accuracy data
    comm_data_filename = os.path.join(comm_output_dir, f'all_algos_comm_vs_acc_data_{dataset}_{origin_model}_{current_timestamp}.json')
    with open(comm_data_filename, 'w') as f:
        json.dump(results_comm_vs_acc, f, indent=2)
    print(f"Communication vs. Accuracy raw data saved to: {comm_data_filename}")

    # --- Original Plotting for Accuracy and Loss vs. Rounds ---
    # (Assuming these plots are still desired, they save to './output/')
    output_main_dir = "./output" # output_main_dir
    os.makedirs(output_main_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for algo, acc in results_test_acc.items():
        plt.plot(range(1, len(acc) + 1), acc, label=algo) # Use len(acc) for rounds
    plt.xlabel('Training Rounds', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title(f'Test Accuracy Comparison ({dataset}-{origin_model})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_main_dir, f'test_accuracy_{origin_model}_{dataset}_{current_timestamp}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    for algo, loss_vals in results_train_loss.items(): # loss_vals
        plt.plot(range(1, len(loss_vals) + 1), loss_vals, label=algo)
    plt.xlabel('Training Rounds', fontsize=14)
    plt.ylabel('Train Loss', fontsize=14)
    plt.title(f'Train Loss Comparison ({dataset}-{origin_model})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_main_dir, f'train_loss_{origin_model}_{dataset}_{current_timestamp}.png'))
    plt.close()


    return results_test_acc, results_train_loss # Required by multi_run.py


# This block will only run when the script is executed directly
if __name__ == '__main__':
    # Example of how to set globals if not running via multi_run.py
    # These would typically be set before calling run_once()
    # For testing, you might want to reduce global_round
    # global_round = 10 # Example: For a quick test run

    print(f"Starting run_once with dataset: {dataset}, model: {origin_model}")
    # The globals (hyperparameters, data variables like client_data, test_dataset, initial_w etc.) 
    # should be fully initialized before this call from the main script part above.
    
    # Ensure data is loaded before run_once
    if initial_w is None:
        print("Error: initial_w is None. Data loading might have failed or was skipped.")
        print("Please ensure the dataset and model initialization block runs correctly.")
    else:
        # Store the returned values, though multi_run.py captures them from globals
        # when it executes this script.
        returned_test_acc, returned_train_loss = run_once()
        print("\nrun_once execution complete.")
        # If you want to see the values when running directly:
        # print("\nReturned Test Acc from run_once (last values):")
        # for algo, acc_list_val in returned_test_acc.items():
        #     if acc_list_val: print(f"  {algo}: {acc_list_val[-1]:.2f}%")