# 设置训练参数
num_rounds = 50
num_clients_per_round = 4  # MNIST参加数目设置为4
client_epochs = 1
server_epochs = 1
client_lr = 0.05  # MNIST: 从0.01，0.05，0.25三个中调优
server_lr = 0.05  # MNIST: 从0.01，0.05，0.25三个中调优
global_lr = 1  # MNIST: 全局学习率为1
min_lr = 0.001  # 最小学习率（衰减之后的最小学习率）
gamma = 0.99  # 衰减系数

server_size = 0.01  # Server数据量占比（MNIST:0.01）
batch_size = 64  # client和server的训练batch size

test_intervals = [5, 10, 20, 50]  # 指定的测试轮次
num_experiments = 5  # 设置重复实验次数

num_clients = 200
non_iid = True  # 设置为True表示非IID划分
