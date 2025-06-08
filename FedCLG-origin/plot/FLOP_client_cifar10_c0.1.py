import json
import matplotlib.pyplot as plt
import os

# 文件所在文件夹
data_dir = '/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/FLOPs'

# 文件名和beta值的对应关系
beta_files = [
    ('0.1', 'all_flops_data_cifar10_resnet_20250604_223701.json'),
    ('0.5', 'all_flops_data_cifar10_resnet_20250604_222926.json'),
    ('1.0', 'all_flops_data_cifar10_resnet_20250604_222350.json'),
    ('iid', 'all_flops_data_cifar10_resnet_20250604_222157.json')
]

method_gflops = dict()
all_methods = set()
setting_min_acc = dict()  # 记录每个setting的最小final accuracy


# 颜色和线型（FedATMV用最醒目的红色和粗虚线，其它可自定义）
plot_methods = [
    ("FedAvg", "FedAvg"),
    ("FedMut", "FedMut"),
    ("HybridFL", "Hybrid-FL"),
    ("CLG-SGD", "CLG-SGD"),
    ("Fed-C", "FedCLG-C"),
    ("Fed-S", "FedCLG-S"),
    ("FedDU", "FedDU"),
    ("FedATMV", "FedATMV"),  
]

colors = [
    '#1f77b4', # blue
    '#ff7f0e', # orange
    '#9467bd', # purple
    '#8c564b', # brown
    '#bcbd22', # olive
    '#17becf', # cyan
    '#7f7f7f', # grey
    '#d62728', # red (FedATMV)
]
linestyles = [
    '-', '--', ':', '-', '--', '-.', ':', '-',  # FedATMV用实线
]

linewidths = [3]*7 + [4]  # FedATMV更粗


for beta, filename in beta_files:
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    # 1. 先找所有方法最终accuracy的最小值
    final_accs = []
    for method, records in data.items():
        all_methods.add(method)
        records = sorted(records, key=lambda x: x['gflops'])
        final_accs.append(records[-1]['accuracy'])
    min_final_acc = min(final_accs)
    setting_min_acc[beta] = min_final_acc
    print(f"β={beta}，所有方法最终accuracy的最小值: {min_final_acc:.4f}")

    # 2. 计算每个方法达到该最小accuracy时的最小gflops
    for method, records in data.items():
        records = sorted(records, key=lambda x: x['gflops'])
        min_gflops = None
        for rec in records:
            if rec['accuracy'] >= min_final_acc:
                min_gflops = rec['gflops']
                break
        if method not in method_gflops:
            method_gflops[method] = {}
        method_gflops[method][beta] = min_gflops

print("\n每个方法在每个setting下达到最小final accuracy时的gflops:")
for method in sorted(all_methods):
    line = []
    for beta, _ in beta_files:
        oh = method_gflops[method].get(beta, None)
        line.append(f"{oh}" if oh is not None else "N/A")
    print(f"{method}: " + "  ".join(line))

# 只保留所有setting都有的数据
all_methods = sorted([m for m in all_methods if set(method_gflops[m].keys()) == set([b for b, _ in beta_files])])

# 绘图并保存
plt.figure(figsize=(8, 6))
beta_labels = ['0.1', '0.5', '1.0', 'iid']
beta_x = list(range(len(beta_labels)))

for idx, (json_key, show_name) in enumerate(plot_methods):
    y = [method_gflops[json_key][b] / 1e5  for b in beta_labels]
    plt.plot(beta_x, y, marker='o', label=show_name, color=colors[idx], linestyle=linestyles[idx], linewidth = linewidths[idx])

plt.xticks(beta_x, beta_labels, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Server non-IID degree (β)', fontsize=22)
plt.ylabel('Clients FLOPs ($\\times 10^{12}$)', fontsize=22)
plt.legend(fontsize=18, ncol=2, columnspacing=0.8, labelspacing=0.3, loc="center right", bbox_to_anchor=(1, 0.44))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图片
save_dir = './fig-flop'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'min_final_acc_gflops_clients_vs_beta.pdf')
plt.savefig(save_path, format='pdf', bbox_inches='tight')
print(f"\n图像已保存到: {save_path}")

plt.close()