import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from gradient_util import compute_server_gradient, compute_client_gradient
from torchvision import datasets, transforms
from collections import Counter
from sklearn.model_selection import train_test_split
from FedCLG import fedclg_c, fedclg_s, test_model, partition_dataset, CNNModel
from config import *  # 导入 config 中的所有变量


# 定义函数运行多个实验
def run_multiple_experiments(
    num_experiments,
    global_model,
    clients_models,
    server_model,
    server_loader,
    client_loaders,
    num_rounds,
    num_clients_per_round,
    client_epochs,
    server_epochs,
    client_lr,
    server_lr,
    device,
    test_loader,
    test_intervals,
    mode="fedclg_c",  # 选择模式：fedclg_c 或 fedclg_s
):
    all_results = []  # 存储每次实验的测试准确率
    for exp in range(num_experiments):
        print(f"Experiment {exp + 1}/{num_experiments}")

        # 初始化模型（每次实验需要重新初始化）
        global_model.apply(reset_weights)  # 重置全局模型权重
        server_model.apply(reset_weights)  # 重置服务器模型权重
        for client_model in clients_models:
            client_model.apply(reset_weights)  # 重置客户端模型权重

        # 运行联邦学习算法
        if mode == "fedclg_c":
            results = fedclg_c(
                global_model,
                clients_models,
                server_model,
                server_loader,
                client_loaders,
                num_rounds,
                num_clients_per_round,
                client_epochs,
                server_epochs,
                client_lr,
                server_lr,
                device,
                test_loader,
                test_intervals,
            )
        elif mode == "fedclg_s":
            results = fedclg_s(
                global_model,
                clients_models,
                server_model,
                server_loader,
                client_loaders,
                num_rounds,
                num_clients_per_round,
                client_epochs,
                server_epochs,
                client_lr,
                server_lr,
                device,
                test_loader,
                test_intervals,
            )
        else:
            raise ValueError("Unsupported algorithm Use 'fedclg_c' or 'fedclg_s'.")

        all_results.append(results)

    # 计算平均值
    avg_results = {}
    for interval in test_intervals:
        avg_results[interval] = np.mean(
            [
                result
                for results in all_results
                for round_idx, result in results
                if round_idx == interval
            ]
        )

    # print("\n--- Summary of Results ---")
    # for interval in test_intervals:
    #     print(f"Average Accuracy at round {interval}: {avg_results[interval]:.2f}%")

    return all_results, avg_results


# 重置模型权重的辅助函数
def reset_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()


# 定义数据转换
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # 这两个值更加符合MNIST数据集的实际分布
    ]
)

# 下载MNIST数据集
train_dataset = datasets.MNIST(
    "../data/MNIST", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    "../data/MNIST", train=False, download=True, transform=transform
)

# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


# ---------从中按照类别平衡选择30000个样本-----------
# 获取数据和标签
data = train_dataset.data
targets = train_dataset.targets

# 使用 sklearn 的分层采样工具
train_indices, _ = train_test_split(
    np.arange(len(targets)),  # 样本索引
    test_size=(len(targets) - 30000) / len(targets),  # 保留 30,000 个样本
    stratify=targets,  # 按类别分层
    random_state=42,  # 设置随机种子，保证结果可复现
)

# 采样后的类别分布
sampled_targets = targets[train_indices]
from collections import Counter

print("采样后类别分布:", Counter(sampled_targets.numpy()))


# 根据选中的索引创建子数据集
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

# 获取划分后的类别分布
labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
# print("Class distribution:", Counter(labels))


# 设置客户端数量和数据划分   论文中MNIST是200个client（每个150samples）
# TODO: 这里考虑client还是全量IID数据，后续考虑调整
client_idxs = partition_dataset(train_dataset, num_clients, non_iid)


# 初始化模型
global_model = CNNModel()
server_model = CNNModel()
clients_models = [CNNModel() for _ in range(num_clients)]

# 准备服务器数据加载器（服务器拥有的小数据集）
server_dataset_size = int(len(train_dataset) * server_size)  # 服务器数据集占1%
# 将数据集随机分为server和client两部分
# TODO：这里是完全随机的，没有考虑平衡类别。后续可以调整
server_dataset, _ = torch.utils.data.random_split(
    train_dataset, [server_dataset_size, len(train_dataset) - server_dataset_size]
)
server_loader = torch.utils.data.DataLoader(
    server_dataset, batch_size=batch_size, shuffle=True
)

# 输出获取 server_dataset 中所有样本的标签
server_labels = [train_dataset.dataset.targets[i] for i in server_dataset.indices]
# print("Server dataset class distribution:", Counter(server_labels))

# 准备客户端数据加载器
client_loaders = []
for idxs in client_idxs:
    client_dataset = torch.utils.data.Subset(train_dataset, idxs)
    loader = torch.utils.data.DataLoader(
        client_dataset, batch_size=batch_size, shuffle=True
    )
    client_loaders.append(loader)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running device:", device)


# 修改后的主程序逻辑
all_results_c, avg_results_c = run_multiple_experiments(
    num_experiments,
    global_model,
    clients_models,
    server_model,
    server_loader,
    client_loaders,
    num_rounds,
    num_clients_per_round,
    client_epochs,
    server_epochs,
    client_lr,
    server_lr,
    device,
    test_loader,
    test_intervals,
    mode="fedclg_c",  # 或者 "fedclg_s"
)


all_results_s, avg_results_s = run_multiple_experiments(
    num_experiments,
    global_model,
    clients_models,
    server_model,
    server_loader,
    client_loaders,
    num_rounds,
    num_clients_per_round,
    client_epochs,
    server_epochs,
    client_lr,
    server_lr,
    device,
    test_loader,
    test_intervals,
    mode="fedclg_s",  # 或者 "fedclg_s"
)

# 输出FedCLG-C的运行结果
print("\n--- FedCLG-C: All Experiment Results ---")
for exp_idx, results in enumerate(all_results_c):
    print(f"Experiment {exp_idx + 1}: {results}")

print("\n--- FedCLG-C: Average Experiment Results  ---")
for interval in test_intervals:
    print(f"Average Accuracy at round {interval}: {avg_results_c[interval]:.2f}%")

# 输出FedCLG-S的运行结果
print("\n--- FedCLG-C: All Experiment Results ---")
for exp_idx, results in enumerate(all_results_s):
    print(f"Experiment {exp_idx + 1}: {results}")

print("\n--- FedCLG-S: Average Experiment Results  ---")
for interval in test_intervals:
    print(f"Average Accuracy at round {interval}: {avg_results_s[interval]:.2f}%")
