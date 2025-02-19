import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
from utils.language_utils import word_to_indices, letter_to_vec

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])
    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    """
    解析给定目录下的数据，要求数据格式为 .json 文件
    返回:
      clients:  客户端列表
      groups:   分组列表
      train_data: 训练数据字典
      test_data: 测试数据字典
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    assert train_clients == test_clients
    assert train_groups == test_groups
    return train_clients, train_groups, train_data, test_data

class ShakeSpeare(Dataset):
    def __init__(self, train=True, reduce_factor=0.5):
        """
        参数:
          - train: 是否构建训练集 (True) 或测试集 (False)
          - reduce_factor: 每个客户端内每个类别保留的比例；例如 0.5 表示只保留一半样本
        """
        super(ShakeSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data(
            "./data/shakespeare/train", "./data/shakespeare/test"
        )
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            # 针对每个客户端分别处理
            for i in range(len(train_clients)):
                self.dic_users[i] = set()
                cur_x = train_data_temp[train_clients[i]]["x"]
                cur_y = train_data_temp[train_clients[i]]["y"]

                # 按类别对当前客户端数据进行分组
                class_indices = defaultdict(list)
                for j, label in enumerate(cur_y):
                    class_indices[label].append(j)

                # 针对每个类别采样 reduce_factor 部分的数据
                selected_indices = []
                for label, indices in class_indices.items():
                    half_count = int(len(indices) * reduce_factor)
                    if half_count == 0 and len(indices) > 0:
                        half_count = 1
                    selected = np.random.choice(indices, half_count, replace=False)
                    selected_indices.extend(selected)
                selected_indices.sort()

                # 逐个追加数据，并记录“追加前”的总长度作为该样本全局索引
                for j in selected_indices:
                    current_idx = len(train_data_x)
                    self.dic_users[i].add(current_idx)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            # 测试集直接全部使用
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]["x"]
                cur_y = test_data_temp[train_clients[i]]["y"]
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        indices = torch.LongTensor(np.array(indices))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("测试集没有 dic_users！")