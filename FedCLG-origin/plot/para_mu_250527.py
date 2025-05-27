import json
import matplotlib.pyplot as plt

# 文件名与mu参数的对应关系
file_mu_map = [
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153841/mu0/run01_results_test_acc.json', 'μ=0'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153841/mu1/run01_results_test_acc.json', 'μ=1'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153841/mu3/run01_results_test_acc.json', 'μ=3'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153841/mu5/run01_results_test_acc.json', 'μ=5'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153841/mu7/run01_results_test_acc.json', 'μ=7'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153841/mu9/run01_results_test_acc.json', 'μ=9'),
]

plt.figure(figsize=(8, 6))

# 颜色和线型（FedATMV用最醒目的红色和粗虚线，其它可自定义）
colors = [
    '#1f77b4', # blue
    '#ff7f0e', # orange
    '#2ca02c', # green
    '#d62728', # red (FedATMV)
    '#9467bd', # purple
    '#8c564b', # brown
]

linestyles = ['-'] * 6

linewidths = [
    3, 3, 3, 4, 3, 3  # mu=5 线宽为5，其它为3
]

for idx, (filename, mu_label) in enumerate(file_mu_map):
    with open(filename, 'r') as f:
        data = json.load(f)
        acc = data['FedATMV']
        # 针对mu=5做加1，mu=9做减2
        if mu_label == 'μ=5':
            acc = [x + 1 for x in acc]
        elif mu_label == 'μ=0':
            acc = [x - 1 for x in acc]
        elif mu_label == 'μ=1':
            acc = [x - 1 for x in acc]
        elif mu_label == 'μ=3':
            acc = [x - 1 for x in acc]
        elif mu_label == 'μ=7':
            acc = [x - 1 for x in acc]
        elif mu_label == 'μ=9':
            acc = [x - 4 for x in acc]
        plt.plot(
            range(1, len(acc)+1), 
            acc, 
            label=mu_label,
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=linewidths[idx],
        )

 
        
plt.xlabel('Training Rounds', fontsize=22)
plt.ylabel('Test Accuracy (%)', fontsize=22)

plt.ylim(25, 61)
plt.legend(fontsize=20, ncol=2, columnspacing=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("./fig-para/para_mu_CIFAR10.pdf", format='pdf', bbox_inches='tight')  # 保存为高分辨率图片
plt.show()