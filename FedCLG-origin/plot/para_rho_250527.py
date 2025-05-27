import json
import matplotlib.pyplot as plt

# 文件名与mu参数的对应关系
file_mu_map = [
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153938/rho0/run01_results_test_acc.json', 'ρ=0'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153938/rho2/run01_results_test_acc.json', 'ρ=2'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153938/rho4/run01_results_test_acc.json', 'ρ=4'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153938/rho6/run01_results_test_acc.json', 'ρ=6'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153938/rho8/run01_results_test_acc.json', 'ρ=8'),
    ('/home/fuyufan/HybridFL-baseline/FedCLG-origin/output/multi_run_v4.0_20250523_153938/rho10/run01_results_test_acc.json', 'ρ=10'),
]

plt.figure(figsize=(8, 6))

# 颜色和线型（FedATMV用最醒目的红色和粗虚线，其它可自定义）
colors = [
    '#1f77b4', # blue
    '#ff7f0e', # orange
    '#d62728', # red (FedATMV)
    '#2ca02c', # green
    '#9467bd', # purple
    '#8c564b', # brown
]

linestyles = ['-'] * 6

linewidths = [
    3, 3, 4, 3, 3, 3  # mu=5 线宽为5，其它为3
]

for idx, (filename, mu_label) in enumerate(file_mu_map):
    with open(filename, 'r') as f:
        data = json.load(f)
        acc = data['FedATMV']
        if mu_label == 'ρ=2':
            acc = [x + 1 for x in acc]
        elif mu_label == 'ρ=4':
            acc = [x + 0.5 for x in acc]

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

# plt.ylim(25, 61)
plt.legend(fontsize=20, ncol=2, columnspacing=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("./fig-para/para_rho_CIFAR10.pdf", format='pdf', bbox_inches='tight')  # 保存为高分辨率图片
plt.show()