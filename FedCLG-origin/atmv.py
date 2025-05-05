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
from utils.language_utils import word_to_indices, letter_to_vec
from utils.ShakeSpeare_reduce import ShakeSpeare

import math

from models.lstm import *

import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# %%
def FedATMV(net_glob, global_round, eta, gamma, K, E, M, ratio=0.3, lambda_val=1):
    """
    参数:
    - net_glob: 初始模型
    - global_round: 全局训练轮数
    - eta: 客户端学习率
    - gamma: 服务器学习率
    - K: 客户端本地训练轮数
    - E: 服务器本地训练轮数
    - M: 每轮采样的客户端数量
    """
    
    net_glob.train()
    
    if origin_model == 'resnet':
        test_model = ResNet18_cifar10().to(device)
    elif origin_model == "lstm":
        test_model = CharLSTM().to(device)
    elif origin_model == "cnn":
        test_model = cnncifar().to(device)
    
    train_w = copy.deepcopy(net_glob.state_dict())
    test_acc = []
    train_loss = []
    
    w_locals = []
    for i in range(M):
        w_locals.append(copy.deepcopy(net_glob.state_dict()))
    
    # 记录每轮的最大更新幅度
    max_rank = 0
    w_old = copy.deepcopy(net_glob.state_dict())
    
    # 服务器更新的最小步长
    server_min = 0
    
    # 收集所有客户端数据标签以计算全局分布
    all_client_labels = []
    for i in range(client_num):
        all_client_labels.extend(client_data[i][1])
    all_client_labels = np.array(all_client_labels)
    
    # 获取所有数据中的唯一类别
    unique_classes = np.unique(all_client_labels)
    num_classes = len(unique_classes)
    
    # 计算全局分布 (列表格式)
    P = [0] * num_classes
    for i, cls in enumerate(unique_classes):
        P[i] = np.sum(all_client_labels == cls) / len(all_client_labels)
    
    # 获取服务器数据信息
    server_labels = np.array(server_data[1])
    n_0 = len(server_labels)  # 服务器数据量
    
    # 计算服务器分布 (列表格式)
    P_0 = [0] * num_classes
    for i, cls in enumerate(unique_classes):
        P_0[i] = np.sum(server_labels == cls) / n_0 if n_0 > 0 else 0
    
    # 计算服务器数据的非IID程度
    D_P_0 = calculate_js_divergence(P_0, P)
    
    # 输出初始设置
    print(f"  服务器数据量: {n_0}")
    print(f"  服务器数据非IID度: {D_P_0:.6f}")
    print(f"  magnitude幅度(magnitude): {magnitude}")
    
    # 记录 alpha_new 和 improvement 的历史（用于后续绘图）
    alpha_history = []
    improvement_history = []
    
    
    for round in tqdm(range(global_round)):
        # 保存当前全局模型作为基准
        w_old = copy.deepcopy(net_glob.state_dict())
        
        local_weights, local_loss = [], []
        
        # 客户端侧训练 - 从总共client_num客户端中选择M个训练
        idxs_users = np.random.choice(range(client_num), M, replace=False)
        
        # 记录当前轮次选择的客户端数据总量
        selected_client_labels = []
        num_current = 0
        
        for i, idx in enumerate(idxs_users):
            # 加载variation后的初始模型
            net_glob.load_state_dict(w_locals[i])
            
            # 客户端本地训练
            update_client_w, client_round_loss, _ = update_weights(
                copy.deepcopy(net_glob.state_dict()), 
                client_data[idx], 
                eta, 
                K
            )
            w_locals[i] = copy.deepcopy(update_client_w)
            local_loss.append(client_round_loss)
            
            # 收集客户端标签和数据量
            selected_client_labels.extend(client_data[idx][1])
            num_current += len(client_data[idx][0])

        # 模型聚合 - FedAvg过程
        w_agg = Aggregation(w_locals, None)  
        
        # 将聚合模型加载到全局模型中
        net_glob.load_state_dict(w_agg)
        
        # 计算选定客户端数据的分布 (列表格式)
        selected_client_labels = np.array(selected_client_labels)
        P_t_prime = [0] * num_classes
        for i, cls in enumerate(unique_classes):
            P_t_prime[i] = np.sum(selected_client_labels == cls) / len(selected_client_labels) if len(selected_client_labels) > 0 else 0
        
        # 计算选定客户端数据的非IID程度
        D_P_t_prime = calculate_js_divergence(P_t_prime, P)
        
        
        # 评估聚合模型的准确率
        test_model.load_state_dict(w_agg)
        acc_t = test_inference(test_model, test_dataset) / 100.0  # 转换为[0,1]比例
        
        epsilon = 1e-10  # 防止除零

        # 计算数据量比例因子：服务器数据占比
        r_data = n_0 / (n_0 + num_current + epsilon)

        # 计算客户端与服务器数据的非IID程度比例
        r_noniid = D_P_t_prime / (D_P_t_prime + D_P_0 + epsilon)

        # 计算验证性能改善因子：
        # 这里假设在全局循环外已经定义了 acc_prev，并在第一轮前将其初始化为 acc_t（或设为一个合理初值）
        if round == 0:
            improvement = 0.0
            acc_prev = acc_t  # 首轮没有对比，直接保存当前准确率
        else:
            improvement = max(0.0, acc_prev - acc_t) / (acc_prev + epsilon)

        # 控制参数：λ 控制验证改善因子的影响，min_alpha 和 max_alpha 限制 α 的范围
        min_alpha = 0.001
        max_alpha = 1.0

        # 新的 α 计算公式
        alpha_new = du_C * (1 - acc_t) * r_data * r_noniid + lambda_val * improvement
        alpha_new = max(min_alpha, min(max_alpha, alpha_new))
        
        # 保存当前轮次的 α_new 与 improvement
        alpha_history.append(alpha_new)
        improvement_history.append(improvement)

        # 保存当前准确率，用于下一轮比较
        acc_prev = acc_t

        # 打印调试信息（可选）
        print(f"Round {round}: r_data={r_data:.4f}, r_noniid={r_noniid:.4f}, improvement={improvement:.4f}, alpha_new={alpha_new:.4f}")
        
        if alpha_new > 0.001:
            # 服务器本地训练部分
            update_server_w, round_loss, _ = update_weights(copy.deepcopy(w_agg), server_data, gamma, E)
            local_loss.append(round_loss)
            # 使用 ratio_combine 函数将客户端聚合模型与服务器更新进行融合
            final_model = ratio_combine(w_agg, update_server_w, alpha_new)
            net_glob.load_state_dict(final_model)
            print(f"Round {round}: Server fixing with alpha_new={alpha_new:.4f}")
        else:
            final_model = copy.deepcopy(w_agg)
            net_glob.load_state_dict(final_model)
            # 仍然计算服务器损失用于记录
            _, round_loss, _ = update_weights(copy.deepcopy(w_agg), server_data, gamma, E)
            local_loss.append(round_loss)
        
        # 测试模型性能
        test_model.load_state_dict(final_model)
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        
        # 在所有测试数据上测试
        current_acc = test_inference(test_model, test_dataset)
        test_acc.append(current_acc)
        
        # 略进行variation（使用全局更新方向作为variation方向）
        w_delta = FedSub(final_model, w_old, 1.0)
        
        # 计算模型更新w_delta的L2范数，衡量模型更新程度
        rank = delta_rank(w_delta)
        if rank > max_rank:
            max_rank = rank
            
        # 250327：加上根据alpha_new的动态调节magnitude
        tmp_magnitude = magnitude*(1 + ratio * alpha_new)
        print(f"Round {round}: tmp_magnitude: {tmp_magnitude}")
            
        # Variation扩散，为下一轮准备初始模型
        w_locals = model_variation(round, final_model, M, w_delta, tmp_magnitude)
        
        # 定期打印信息
        if round % 10 == 0:
            print(f"Round {round}: Acc={current_acc:.2f}%, Loss={loss_avg:.4f}, D_P_t={D_P_t_prime:.4f}, Alpha={alpha_new:.4f}")
            
            
            
def KL_divergence(p1, p2):
    """
    计算KL散度，与参考代码一致
    """
    d = 0
    for i in range(len(p1)):
        if p2[i] == 0 or p1[i] == 0:
            continue
        d += p1[i] * math.log(p1[i]/p2[i], 2)  # 使用以2为底的对数
    return d

def calculate_js_divergence(p1, p2):
    """
    计算Jensen-Shannon散度，与参考代码完全一致
    """
    # 创建中点分布p3
    p3 = []
    for i in range(len(p1)):
        p3.append((p1[i] + p2[i])/2)
    
    # 计算JS散度 = (KL(p1||p3) + KL(p2||p3))/2
    return KL_divergence(p1, p3)/2 + KL_divergence(p2, p3)/2

# %%
# 本地训练并更新权重，返回更新后的模型权重、平均训练损失以及第一个迭代的梯度信息
def update_weights(model_weight, dataset, learning_rate, local_epoch):
    if origin_model == 'resnet':
        model = ResNet18_cifar10().to(device)
    elif origin_model == "lstm":
        model = CharLSTM().to(device)
    elif origin_model == "cnn":
        model = cnncifar().to(device)
    
    model.load_state_dict(model_weight)

    model.train()
    epoch_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if origin_model == 'resnet' or origin_model == 'cnn':
        Tensor_set = TensorDataset(torch.Tensor(dataset[0]).to(device), torch.Tensor(dataset[1]).to(device))
    elif origin_model == 'lstm':
        Tensor_set = TensorDataset(torch.LongTensor(dataset[0]).to(device), torch.Tensor(dataset[1]).to(device))
    
    data_loader = DataLoader(Tensor_set, batch_size=bc_size, shuffle=True)

    first_iter_gradient = None  # 初始化变量来保存第一个iter的梯度

    for iter in range(local_epoch):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['output'], labels.long())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/images.shape[0])

            # 保存第一个iter的梯度
            if iter == 0 and batch_idx == 0:
                first_iter_gradient = {}
                for name, param in model.named_parameters():
                    first_iter_gradient[name] = param.grad.clone()
                # 保存 BatchNorm 层的 running mean 和 running variance
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        first_iter_gradient[name + '.running_mean'] = module.running_mean.clone()
                        first_iter_gradient[name + '.running_var'] = module.running_var.clone()

        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss), first_iter_gradient


# 加权平均聚合，lens代表了权重，如果没有定义就是普通平均
def Aggregation(w, lens):
    w_avg = None
    if lens == None:
        total_count = len(w)
        lens = []
        for i in range(len(w)):
            lens.append(1.0)
    else:
        total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg



def ratio_combine(w1, w2, ratio=0):
    """
    将两个权重进行加权平均，ratio表示w2的占比
    对应参考代码中的ratio_combine函数
    """
    w = copy.deepcopy(w1)
    for key in w.keys():
        if 'num_batches_tracked' in key:
            continue
        w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w


def FedSub(w, w_old, weight):
    w_sub = copy.deepcopy(w)
    for k in w_sub.keys():
        w_sub[k] = (w[k] - w_old[k]) * weight

    return w_sub

def delta_rank(delta_dict):
    cnt = 0
    dict_a = torch.Tensor(0)
    s = 0
    for p in delta_dict.keys():
        a = delta_dict[p]
        a = a.view(-1)
        if cnt == 0:
            dict_a = a
        else:
            dict_a = torch.cat((dict_a, a), dim=0)

        cnt += 1
        # print(sim)
    s = torch.norm(dict_a, dim=0)
    return s


def model_variation(iter, w_glob, m, w_delta, alpha):

    w_locals_new = []
    ctrl_cmd_list = []
    ctrl_rate = var_acc_rate * (
        1.0 - min(iter * 1.0 / var_bound, 1.0)
    )  # 论文中的βt，随着iter逐渐从β0减小到0

    # k代表模型中的参数数量，对每个参数按照client数量分配v（论文中是按照每一层分配）
    for k in w_glob.keys():
        ctrl_list = []
        for i in range(0, int(m / 2)):
            ctrl = random.random()  # 随机数，范围：[0,1)
            # 这里分ctrl感觉没什么必要，shuffle后都会随机掉
            if ctrl > 0.5:
                ctrl_list.append(1.0)
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
            else:
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                ctrl_list.append(1.0)
        random.shuffle(ctrl_list)  # 打乱列表
        ctrl_cmd_list.append(ctrl_list)
    cnt = 0
    for j in range(m):
        w_sub = copy.deepcopy(w_glob)
        if not (cnt == m - 1 and m % 2 == 1):
            ind = 0
            for k in w_sub.keys():
                w_sub[k] = w_sub[k] + w_delta[k] * ctrl_cmd_list[ind][j] * alpha
                ind += 1
        cnt += 1
        w_locals_new.append(w_sub)

    return w_locals_new