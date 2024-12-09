import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms


# 先定义一个简单的CNN模型（原文采用的是LeNet-5）
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
       
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)      
        x = x.view(x.size(0), -1)     
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 这两个值更加符合MNIST数据集的实际分布
])

# 下载MNIST数据集
train_dataset = datasets.MNIST('../data/MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data/MNIST', train=False, download=True, transform=transform)


# 将数据集划分给客户端
def partition_dataset(dataset, num_clients, non_iid):
    dataset_size = len(dataset)
    print("MNIST数据集大小：", dataset_size)
    data_per_client = int(dataset_size / num_clients)
    client_idxs = []

    if non_iid:
        # 非独立同分布划分
        rand_idxs = np.random.permutation(dataset_size)
        labels = np.array(dataset.targets)[rand_idxs]
        idxs_train = rand_idxs
        
        idxs_labels = np.vstack((idxs_train, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

        idxs = idxs_labels[0, :]
        for i in range(num_clients):
            client_idxs.append(idxs[i * data_per_client: (i + 1) * data_per_client])
    else:
        # 独立同分布划分
        rand_idxs = np.random.permutation(dataset_size)
        for i in range(num_clients):
            client_idxs.append(rand_idxs[i * data_per_client: (i + 1) * data_per_client])
    return client_idxs

# 设置客户端数量和数据划分   论文中MNIST是200个client（每个150samples）
num_clients = 200
non_iid = True  # 设置为True表示非IID划分
client_idxs = partition_dataset(train_dataset, num_clients, non_iid)


def client_update(client_model, optimizer, train_loader, epoch, device, correction=None):
    client_model.train()
    for _ in range(epoch):
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
            
def server_update(global_model, server_model, server_optimizer, server_loader, epoch, device):
    server_model.load_state_dict(global_model.state_dict())
    server_model.train()
    for _ in range(epoch):
        for data, target in server_loader:
            data, target = data.to(device), target.to(device)
            server_optimizer.zero_grad()
            output = server_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            server_optimizer.step()
    global_model.load_state_dict(server_model.state_dict())
    
    

def fedclg_c(global_model, clients_models, server_model, server_loader, client_loaders, num_rounds, num_clients_per_round, 
             client_epochs, server_epochs, client_lr, server_lr, device):
    num_clients = len(clients_models)
    global_model.to(device)
    server_model.to(device)

    # 初始化优化器
    server_optimizer = optim.SGD(server_model.parameters(), lr=server_lr)
    
    for round_idx in range(num_rounds):
        print(f'Round {round_idx+1}/{num_rounds}')
        # 选择参与的客户端
        selected_clients = np.random.choice(range(num_clients), num_clients_per_round, replace=False)
        # 服务器计算梯度g_s
        server_model.load_state_dict(global_model.state_dict())
        server_model.eval()
        g_s = []
        for data, target in server_loader:
            data, target = data.to(device), target.to(device)
            output = server_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            for param in server_model.parameters():
                g_s.append(param.grad.data.clone())
            break  # 仅计算一批用于计算g_s，实际可根据论文进行修改

        # 客户端更新
        for client_idx in selected_clients:
            client_model = clients_models[client_idx]
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)
            optimizer = optim.SGD(client_model.parameters(), lr=client_lr)
            # 计算校正项c_i = g_s - g_i
            client_model.eval()
            g_i = []
            for data, target in client_loaders[client_idx]:
                data, target = data.to(device), target.to(device)
                output = client_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                for param in client_model.parameters():
                    g_i.append(param.grad.data.clone())
                break  # 仅计算一批用于计算g_i，实际可根据论文进行修改

            correction = []
            for gs, gi in zip(g_s, g_i):
                correction.append(gs - gi)
            
            # 客户端训练
            client_update(client_model, optimizer, client_loaders[client_idx], client_epochs, device, correction)
            
            # 从客户端收集更新
            client_model.to('cpu')
            client_params = client_model.state_dict()
            if client_idx == selected_clients[0]:
                global_params = client_params
            else:
                for key in global_params:
                    global_params[key] += client_params[key]

        # 聚合更新全局模型参数
        for key in global_params:
            global_params[key] = global_params[key] / num_clients_per_round
        global_model.load_state_dict(global_params)

        # 服务器本地训练
        server_update(global_model, server_model, server_optimizer, server_loader, server_epochs, device)
        
        
def fedclg_s(global_model, clients_models, server_model, server_loader, client_loaders, num_rounds, num_clients_per_round, 
             client_epochs, server_epochs, client_lr, server_lr, device):
    num_clients = len(clients_models)
    global_model.to(device)
    server_model.to(device)

    # 初始化优化器
    server_optimizer = optim.SGD(server_model.parameters(), lr=server_lr)
    
    for round_idx in range(num_rounds):
        print(f'Round {round_idx+1}/{num_rounds}')
        # 选择参与的客户端
        selected_clients = np.random.choice(range(num_clients), num_clients_per_round, replace=False)
        # 服务器计算梯度g_s
        server_model.load_state_dict(global_model.state_dict())
        server_model.eval()
        g_s = []
        for data, target in server_loader:
            data, target = data.to(device), target.to(device)
            output = server_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            for param in server_model.parameters():
                g_s.append(param.grad.data.clone())
            break  # 仅计算一批用于计算g_s，实际可根据论文进行修改

        # 客户端更新
        client_grads = []
        for client_idx in selected_clients:
            client_model = clients_models[client_idx]
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)
            optimizer = optim.SGD(client_model.parameters(), lr=client_lr)
            
            # 客户端训练
            client_update(client_model, optimizer, client_loaders[client_idx], client_epochs, device)
            
            # 计算g_i
            client_model.eval()
            g_i = []
            for data, target in client_loaders[client_idx]:
                data, target = data.to(device), target.to(device)
                output = client_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                for param in client_model.parameters():
                    g_i.append(param.grad.data.clone())
                break  # 仅计算一批用于计算g_i，实际可根据论文进行修改
            client_grads.append(g_i)
            
            # 从客户端收集更新
            client_model.to('cpu')
            client_params = client_model.state_dict()
            if client_idx == selected_clients[0]:
                global_params = client_params
            else:
                for key in global_params:
                    global_params[key] += client_params[key]
        
        # 计算校正项并更新全局模型
        for idx, client_idx in enumerate(selected_clients):
            g_i = client_grads[idx]
            correction = []
            for gs, gi in zip(g_s, g_i):
                correction.append(gi - gs)
            # 更新客户端参数
            client_model = clients_models[client_idx]
            client_model.load_state_dict(global_params)
            for param, corr in zip(client_model.parameters(), correction):
                param.data -= client_lr * corr / num_clients_per_round
            # 累计更新后的参数
            client_params = client_model.state_dict()
            if idx == 0:
                updated_params = client_params
            else:
                for key in updated_params:
                    updated_params[key] += client_params[key]
        for key in updated_params:
            updated_params[key] = updated_params[key] / num_clients_per_round
        global_model.load_state_dict(updated_params)

        # 服务器本地训练
        server_update(global_model, server_model, server_optimizer, server_loader, server_epochs, device)
        
  
        
# 初始化模型
global_model = CNNModel()
server_model = CNNModel()
clients_models = [CNNModel() for _ in range(num_clients)]

# 准备服务器数据加载器（服务器拥有的小数据集）
server_dataset_size = int(len(train_dataset) * 0.01)  # 服务器数据集占1%
server_dataset, _ = torch.utils.data.random_split(train_dataset, [server_dataset_size, len(train_dataset) - server_dataset_size])
server_loader = torch.utils.data.DataLoader(server_dataset, batch_size=32, shuffle=True)

# 准备客户端数据加载器
client_loaders = []
for idxs in client_idxs:
    client_dataset = torch.utils.data.Subset(train_dataset, idxs)
    loader = torch.utils.data.DataLoader(client_dataset, batch_size=32, shuffle=True)
    client_loaders.append(loader)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置训练参数
num_rounds = 10
num_clients_per_round = 10
client_epochs = 1
server_epochs = 1
client_lr = 0.01
server_lr = 0.01

# 选择要运行的算法：fedclg_c 或 fedclg_s
# fedclg_c(global_model, clients_models, server_model, server_loader, client_loaders, num_rounds, num_clients_per_round, 
#          client_epochs, server_epochs, client_lr, server_lr, device)

fedclg_s(global_model, clients_models, server_model, server_loader, num_rounds, num_clients_per_round, 
         client_epochs, server_epochs, client_lr, server_lr, device)


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    model.to('cpu')
    print(f'测试集上的准确率为：{100 * correct / total}%')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
test_model(global_model, test_loader, device)