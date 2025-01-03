import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 启用 CUDA 同步错误报告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 指定 GPU 1
gpu = 1
if torch.cuda.is_available() and gpu >= 0 and gpu < torch.cuda.device_count():
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
print(torch.cuda.device_count())


# 定义一个简化的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型并移动到设备
try:
    model = SimpleCNN().to(device)
    print("Model initialized on device successfully.")
except RuntimeError as e:
    print(f"Error initializing model on device: {e}")

# 创建一个假数据进行前向传播
dummy_input = torch.randn(1, 3, 32, 32).to(device)
try:
    output = model(dummy_input)
    print("Forward pass successful. Output shape:", output.shape)
except RuntimeError as e:
    print(f"Error during forward pass: {e}")
