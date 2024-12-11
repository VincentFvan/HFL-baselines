import torch
import torch.nn as nn

def compute_server_gradient(server_model, server_loader, device):
    """
    在服务器端计算全局模型的梯度 g_s
    """
    server_model.eval()  # 设置为评估模式
    g_s = []
    for data, target in server_loader:
        data, target = data.to(device), target.to(device)
        output = server_model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()  # 反向传播计算梯度
        for param in server_model.parameters():
            g_s.append(param.grad.data.clone())  # 保存服务器的梯度 g_s
        break  # 仅使用一个批次数据计算 g_s，减少计算开销
    g_s = [g.cpu() for g in g_s]  # 将梯度从 GPU 转移到 CPU
    return g_s

def compute_client_gradient(client_model, client_loader, device):
    """
    在客户端计算本地模型的梯度 g_i
    """
    client_model.eval()  # 设置为评估模式
    g_i = []
    for data, target in client_loader:
        data, target = data.to(device), target.to(device)
        client_model.zero_grad()  # 清除模型的梯度
        output = client_model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()  # 反向传播计算梯度
        for param in client_model.parameters():
            g_i.append(param.grad.data.clone())  # 保存客户端的梯度 g_i
        break  # 仅使用一个批次数据计算 g_i
    g_i = [g.cpu() for g in g_i]  # 将梯度从 GPU 转移到 CPU
    return g_i