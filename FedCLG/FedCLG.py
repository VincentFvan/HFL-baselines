import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from gradient_util import compute_server_gradient, compute_client_gradient
from torchvision import datasets, transforms
from collections import Counter
from sklearn.model_selection import train_test_split
from config import *  # 导入 config 中的所有变量


# 先定义一个简单的CNN模型（原文采用的是LeNet-5）
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 将数据集划分给客户端
def partition_dataset(dataset, num_clients, non_iid):
    dataset_size = len(dataset)
    print("MNIST数据集大小：", dataset_size)
    data_per_client = int(dataset_size / num_clients)
    client_idxs = []

    if non_iid:
        # 非独立同分布划分
        rand_idxs = np.random.permutation(dataset_size)
        labels = np.array(dataset.dataset.targets)[rand_idxs]
        idxs_train = rand_idxs

        idxs_labels = np.vstack((idxs_train, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

        idxs = idxs_labels[0, :]
        for i in range(num_clients):
            client_idxs.append(idxs[i * data_per_client : (i + 1) * data_per_client])
    else:
        # 独立同分布划分
        rand_idxs = np.random.permutation(dataset_size)
        for i in range(num_clients):
            client_idxs.append(
                rand_idxs[i * data_per_client : (i + 1) * data_per_client]
            )
    return client_idxs


def client_update(
    client_model, optimizer, train_loader, epoch, device, correction=None
):
    client_model.train()
    for _ in range(epoch):  # client local epoch
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            if correction is not None:
                # 应用校正项
                for param, corr in zip(client_model.parameters(), correction):
                    param.grad.data += corr.to(device)
            optimizer.step()


def server_update(
    global_model, server_model, server_optimizer, server_loader, epoch, device
):
    server_model.load_state_dict(global_model.state_dict())
    server_model.train()
    # print(f"Server Learning Rate: {server_optimizer.param_groups[0]['lr']}")
    for _ in range(epoch):
        for data, target in server_loader:
            data, target = data.to(device), target.to(device)
            server_optimizer.zero_grad()
            output = server_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            server_optimizer.step()
    global_model.load_state_dict(server_model.state_dict())
    # 将server模型参数更新到global模型


def fedclg_c(
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
):

    num_clients = len(clients_models)
    global_model.to(device)
    server_model.to(device)

    # 初始化优化器
    server_optimizer = optim.SGD(server_model.parameters(), lr=server_lr)
    # 为server设置学习率调度器
    server_scheduler = optim.lr_scheduler.ExponentialLR(server_optimizer, gamma)

    # 初始化客户端优化器和调度器
    client_optimizers = []
    client_schedulers = []
    for client_idx in range(num_clients):
        client_model = clients_models[client_idx]
        optimizer = optim.SGD(client_model.parameters(), lr=client_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        client_optimizers.append(optimizer)
        client_schedulers.append(scheduler)

    results = []  # 用于记录指定轮次的结果

    for round_idx in range(num_rounds):
        print(f"Round {round_idx+1}/{num_rounds}")

        # 获取当前客户端的学习率（所有客户端的学习率此时应相同）
        # current_lr = client_optimizers[0].param_groups[0]["lr"]
        # print(f"Client Learning Rate: {current_lr}")

        # ---------- 服务器计算梯度 g_s ----------
        # TODO: 这部分需要在server上进行，之后要区分；
        # TODO：并且这里server需要将g_s传给client，后面考虑体现
        server_model.load_state_dict(
            global_model.state_dict()
        )  # 同步全局模型参数到服务器
        g_s = compute_server_gradient(
            server_model, server_loader, device
        )  # 调用服务器梯度计算函数

        # 选择参与的客户端
        selected_clients = np.random.choice(
            range(num_clients), num_clients_per_round, replace=False
        )

        # 用于存储客户端的更新 Δ_i 和梯度 g_i
        delta_list = []
        g_i_list = []

        # 客户端更新
        for client_idx in selected_clients:
            client_model = clients_models[client_idx]
            client_model.load_state_dict(
                global_model.state_dict()
            )  # 将全局模型参数分发到clients
            client_model.to(device)

            # 获取已初始化的优化器和调度器
            optimizer = client_optimizers[client_idx]
            scheduler = client_schedulers[client_idx]

            # 更新优化器的参数组的参数引用
            optimizer.param_groups[0]["params"] = list(client_model.parameters())

            # ---------- 客户端计算梯度 g_i ----------
            g_i = compute_client_gradient(
                client_model, client_loaders[client_idx], device
            )  # 调用客户端梯度计算函数
            g_i_list.append(g_i)  # 上传客户端梯度到服务器

            # 计算校正项c_i = g_s - g_i  (这个理论上是要在server上进行，后续看怎么调整)
            correction = []
            for gs, gi in zip(g_s, g_i):
                correction.append(gs - gi)

            # 客户端训练
            client_update(
                client_model,
                optimizer,
                client_loaders[client_idx],
                client_epochs,
                device,
                correction,
            )

            # 计算模型更新 Δ_i = x_i^K - x_t
            delta_i = {}
            for key in global_model.state_dict().keys():
                # TODO：从计算图中分离张量（不用进行梯度计算），并移动到cpu上进行
                delta_i[key] = (
                    (client_model.state_dict()[key] - global_model.state_dict()[key])
                    .detach()
                    .cpu()
                )
            delta_list.append(delta_i)

            # 将客户端模型移回 CPU
            client_model.to("cpu")

        # ---------- 服务器端聚合更新 ----------
        # TODO：这一段开始是在server上进行的

        # 初始化累计更新
        aggregated_update = {}
        for key in global_model.state_dict().keys():
            # TODO：这里将服务器聚合过程转移到cpu上进行
            aggregated_update[key] = torch.zeros_like(
                global_model.state_dict()[key], device="cpu"
            )

        # print(f"aggregated_update[{key}] is on {aggregated_update[key].device}")
        # print(f"delta_i[{key}] is on {delta_i[key].device}")

        # 服务器聚合
        for i in range(len(delta_list)):
            delta_i = delta_list[i]
            for idx, key in enumerate(global_model.state_dict().keys()):
                # delta_i[key] 是 Δ_i
                aggregated_update[key] += delta_i[key] / num_clients_per_round  # 平均化

        # 更新全局模型参数 x_{t+1}^s = x_t + η_g * aggregated_update
        global_model.to(
            device
        )  # 这里用gpu并行计算的原因是，模型更新操作(add_)涉及多个张量的加法操作
        with torch.no_grad():
            for key in global_model.state_dict().keys():
                # global_lr是全局学习率
                global_model.state_dict()[key].add_(
                    aggregated_update[key].to(device), alpha=global_lr
                )

        # 服务器本地训练
        # TODO: 这里将这一步放在同一个gpu上，后续可以修改
        server_update(
            global_model,
            server_model,
            server_optimizer,
            server_loader,
            server_epochs,
            device,
        )

        # 如果是指定的测试轮次，记录测试结果
        #  TODO: 但这里测试会增加额外的时延，后面要考虑
        if round_idx + 1 in test_intervals:
            accuracy = test_model(global_model, test_loader, device, verbose=False)
            print(f"Accuracy at round {round_idx+1}: {accuracy:.2f}%")
            results.append((round_idx + 1, accuracy))

        for optimizer, scheduler in zip(client_optimizers, client_schedulers):
            optimizer.step()  # 先执行参数更新
            scheduler.step()  # 再调整学习率
            adjust_learning_rate(optimizer, min_lr)  # 可选：设置最小学习率
        server_optimizer.step()  # 服务器优化器更新
        server_scheduler.step()  # 调整服务器学习率
        adjust_learning_rate(server_optimizer, min_lr)  # 可选：设置最小学习率

    return results


def adjust_learning_rate(optimizer, min_lr):
    for param_group in optimizer.param_groups:
        if param_group["lr"] < min_lr:
            param_group["lr"] = min_lr


def fedclg_s(
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
):

    num_clients = len(clients_models)
    global_model.to(device)
    server_model.to(device)

    results = []  # 用于记录指定轮次的结果

    # 初始化优化器
    server_optimizer = optim.SGD(server_model.parameters(), lr=server_lr)
    # 为server设置学习率调度器
    server_scheduler = optim.lr_scheduler.ExponentialLR(server_optimizer, gamma)

    # 初始化客户端优化器和调度器
    client_optimizers = []
    client_schedulers = []
    for client_idx in range(num_clients):
        client_model = clients_models[client_idx]
        optimizer = optim.SGD(client_model.parameters(), lr=client_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        client_optimizers.append(optimizer)
        client_schedulers.append(scheduler)

    for round_idx in range(num_rounds):
        print(f"Round {round_idx+1}/{num_rounds}")

        # 获取当前客户端的学习率（所有客户端的学习率此时应相同）
        # current_lr = client_optimizers[0].param_groups[0]["lr"]
        # print(f"Client Learning Rate: {current_lr}")

        # ---------- 服务器计算梯度 g_s ----------
        # TODO: 这部分需要在server上进行，之后要区分
        server_model.load_state_dict(
            global_model.state_dict()
        )  # 同步全局模型参数到服务器
        g_s = compute_server_gradient(
            server_model, server_loader, device
        )  # 调用服务器梯度计算函数

        # 选择参与的客户端
        selected_clients = np.random.choice(
            range(num_clients), num_clients_per_round, replace=False
        )

        # 用于存储客户端的更新 Δ_i 和梯度 g_i
        delta_list = []
        g_i_list = []

        # 客户端更新
        client_grads = []
        for client_idx in selected_clients:
            client_model = clients_models[client_idx]
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)  # 转移到gpu上进行

            # 获取已初始化的优化器和调度器
            optimizer = client_optimizers[client_idx]
            scheduler = client_schedulers[client_idx]

            # 更新优化器的参数组的参数引用
            optimizer.param_groups[0]["params"] = list(client_model.parameters())

            # ---------- 客户端计算梯度 g_i ----------
            # TODO: 这部分在client上实现
            g_i = compute_client_gradient(
                client_model, client_loaders[client_idx], device
            )  # 调用客户端梯度计算函数
            g_i_list.append(g_i)  # 上传客户端梯度到服务器

            client_update(
                client_model,
                optimizer,
                client_loaders[client_idx],
                client_epochs,
                device,
            )

            # 计算模型更新 Δ_i = x_i^K - x_t
            delta_i = {}
            for key in global_model.state_dict().keys():
                delta_i[key] = (
                    (client_model.state_dict()[key] - global_model.state_dict()[key])
                    .detach()
                    .cpu()
                )
            delta_list.append(delta_i)

            # 将客户端模型移回 CPU
            client_model.to("cpu")

        # ---------- 服务器端聚合更新 ----------
        # TODO：这一段开始是在server上进行的

        # 初始化累计更新
        aggregated_update = {}
        for key in global_model.state_dict().keys():
            aggregated_update[key] = torch.zeros_like(
                global_model.state_dict()[key], device="cpu"
            )

        for i in range(len(delta_list)):
            delta_i = delta_list[i]
            g_i = g_i_list[i]
            # 计算校正项 K * eta * (g_s - g_i)
            correction = []
            for gs, gi in zip(g_s, g_i):
                correction.append(
                    client_epochs * client_lr * (gs - gi)
                )  # K * η * (g_s - g_i)
            # 将校正项应用于对应的参数
            for idx, key in enumerate(global_model.state_dict().keys()):
                # delta_i[key] 是 Δ_i
                # correction[idx] 是校正项
                aggregated_update[key] += (
                    delta_i[key] - correction[idx]
                ) / num_clients_per_round  # 平均化

        # 更新全局模型参数 x_{t+1}^s = x_t + η_g * aggregated_update
        global_model.to(
            device
        )  # 这里用gpu并行计算的原因是，模型更新操作(add_)涉及多个张量的加法操作
        with torch.no_grad():
            for key in global_model.state_dict().keys():
                # global_lr是全局学习率
                global_model.state_dict()[key].add_(
                    aggregated_update[key].to(device), alpha=global_lr
                )

        # 服务器本地训练
        # TODO: 这里将这一步放在同一个gpu上，后续可以修改
        server_update(
            global_model,
            server_model,
            server_optimizer,
            server_loader,
            server_epochs,
            device,
        )

        # 如果是指定的测试轮次，记录测试结果
        #  TODO: 但这里测试会增加额外的时延，后面要考虑
        if round_idx + 1 in test_intervals:
            accuracy = test_model(global_model, test_loader, device, verbose=False)
            print(f"Accuracy at round {round_idx+1}: {accuracy:.2f}%")
            results.append((round_idx + 1, accuracy))

        for optimizer, scheduler in zip(client_optimizers, client_schedulers):
            optimizer.step()  # 先执行参数更新
            scheduler.step()  # 再调整学习率
            adjust_learning_rate(optimizer, min_lr)  # 可选：设置最小学习率
        server_optimizer.step()  # 服务器优化器更新
        server_scheduler.step()  # 调整服务器学习率
        adjust_learning_rate(server_optimizer, min_lr)  # 可选：设置最小学习率

    return results


# 测试函数（改为返回值）
def test_model(model, test_loader, device, verbose=True):
    model.eval()
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():  # 在测试过程中不需要计算梯度，节省内存和加速计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # TODO: 这里因为后面还要训练，不移动到CPU上
    # model.to("cpu")
    accuracy = 100 * correct / total
    if verbose:
        print(f"测试集上的准确率为：{accuracy:.2f}%")
    return accuracy
