import torch
import torchvision
import numpy
import matplotlib
import sklearn
import seaborn

print("PyTorch 版本：", torch.__version__)
print("Torchvision 版本：", torchvision.__version__)
print("NumPy 版本：", numpy.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print(matplotlib.__version__)
print(sklearn.__version__)
print(seaborn.__version__)