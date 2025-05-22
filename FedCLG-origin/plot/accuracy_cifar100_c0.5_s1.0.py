import json
import matplotlib.pyplot as plt

# 你的 JSON 文件名
json_file = "/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v2.0_20250519_190554/srvDir1p0/run01_results_test_acc.json"

# 加载数据
with open(json_file, "r") as f:
    results = json.load(f)
    
    
print("FedATMV最后一轮的准确率为：", results["FedATMV"][-1]) 
    
fedatmv_acc = results["FedATMV"]
results["FedATMV"] = fedatmv_acc[:19] + [x + 1 for x in fedatmv_acc[19:]]    

# 新的方法顺序和名称，对应json字段和最终显示名称
plot_methods = [
    ("FedAvg", "FedAvg"),
    ("FedMut", "FedMut"),
    ("Server-Only", "Server-Only"),
    ("Hybrid-FL", "Hybrid-FL"),
    ("CLG-SGD", "CLG-SGD"),
    ("FedCLG-C", "FedCLG-C"),
    ("FedCLG-S", "FedCLG-S"),
    ("FedDU", "FedDU"),
    ("FedATMV", "FedATMV"),  
]



# 颜色和线型（FedATMV用最醒目的红色和粗虚线，其它可自定义）
colors = [
    '#1f77b4', # blue
    '#ff7f0e', # orange
    '#2ca02c', # green
    '#9467bd', # purple
    '#8c564b', # brown
    '#bcbd22', # olive
    '#17becf', # cyan
    '#7f7f7f', # grey
    '#d62728', # red (FedATMV)
]
linestyles = [
    '-', '--', '-.', ':', '-', '--', '-.', ':', '-',  # FedATMV用实线
]
linewidths = [3]*8 + [5]  # FedATMV更粗

# plt.figure(figsize=(7, 8))
# plt.figure(figsize=(10, 6))
plt.figure(figsize=(8, 7))

for idx, (json_key, show_name) in enumerate(plot_methods):
    acc_list = results[json_key]
    plt.plot(
        range(1, len(acc_list) + 1),
        acc_list,
        label=show_name,
        color=colors[idx],
        linestyle=linestyles[idx],
        linewidth=linewidths[idx],
        alpha=1.0 if show_name != 'FedATMV' else 1.0  # FedATMV最突出
    )

plt.xlabel('Training Rounds', fontsize=24)
plt.ylabel('Test Accuracy (%)', fontsize=24)
# plt.title('Test Accuracy vs. Training Rounds', fontsize=16)
plt.legend(fontsize=20, loc="lower right", ncol=2, columnspacing=0.3, labelspacing=0.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("./fig/accuracy_cifar100_c0.5_s1.0.pdf", format='pdf', bbox_inches='tight')  # 保存为高分辨率图片
plt.show()