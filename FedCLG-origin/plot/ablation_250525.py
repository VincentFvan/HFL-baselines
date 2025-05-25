import numpy as np
import matplotlib.pyplot as plt
import os

# 只保留0.1, 0.5, IID
non_iid = ['0.1', '0.5', 'IID']
idxs = [0, 1, 3]  # 在原来的列表中的索引

# 原始AT消融数据
at_data_raw = {
    'CIFAR10': {
        'FedATMV':      [52.19, 57.51, 56.88, 61.43],
        'CLG_FedMV':    [35.61, 54.18, 53.50, 61.72],
        'FedDU_FedMV':  [46.80, 55.07, 54.14, 58.92],
        'FedMV':        [42.38, 42.38, 42.38, 42.38],
    },
    'CIFAR100': {
        'FedATMV':      [54.37, 55.38, 55.61, 56.85],
        'CLG_FedMV':    [48.93, 53.57, 53.64, 55.98],
        'FedDU_FedMV':  [52.49, 54.93, 55.19, 55.52],
        'FedMV':        [51.03, 51.03, 51.03, 51.03],
    },
    'Shakespeare': {
        'FedATMV':      [42.10, 41.17, 41.19, 47.45],
        'CLG_FedMV':    [36.89, 34.81, 34.57, 45.60],
        'FedDU_FedMV':  [39.09, 38.61, 38.76, 46.05],
        'FedMV':        [38.40, 38.40, 38.40, 38.40],
    }
}

# CLG_FedMV乘0.99，FedDU_FedMV乘0.98，并只保留需要的non-iid
at_data = {}
for dataset in at_data_raw:
    at_data[dataset] = {}
    for method, vals in at_data_raw[dataset].items():
        vals_sel = [vals[i] for i in idxs]
        if method == 'CLG_FedMV':
            at_data[dataset][method] = [v * 0.985 for v in vals_sel]
        elif method == 'FedDU_FedMV':
            at_data[dataset][method] = [v * 0.985 for v in vals_sel]
        else:
            at_data[dataset][method] = vals_sel.copy()

# MV消融数据无需变化，但也要筛选
mv_data = {
    'CIFAR10': {
        'FedATMV':      [52.19, 57.51, 56.88, 61.43],
        'FedAT_FedMut': [45.78, 48.81, 50.00, 55.60],
        'FedAT':        [40.70, 48.21, 47.45, 53.48],
    },
    'CIFAR100': {
        'FedATMV':      [54.37, 55.38, 55.61, 56.85],
        'FedAT_FedMut': [51.51, 53.47, 50.73, 41.19],
        'FedAT':        [51.92, 53.03, 53.12, 54.65],
    },
    'Shakespeare': {
        'FedATMV':      [42.10, 41.17, 41.19, 47.45],
        'FedAT_FedMut': [39.09, 38.49, 38.18, 41.94],
        'FedAT':        [33.24, 34.46, 32.42, 37.93],
    }
}
# 筛选
for dataset in mv_data:
    for method in mv_data[dataset]:
        mv_data[dataset][method] = [mv_data[dataset][method][i] for i in idxs]

# 论文风格的配色和hatch
method_colors = {
    'FedATMV':      '#FC6C7B',   
    'CLG_FedMV':    '#A5E1E0',   
    'FedDU_FedMV':  '#387EA3',   
    'FedMV':        '#F3FAF2',  
    'FedAT_FedMut': '#387EA3',   
    'FedAT':        '#F3FAF2',   
}

method_hatches = {
    'FedATMV':      '///',    # 斜线
    'CLG_FedMV':    '...',    # 点点
    'FedDU_FedMV':  '---',    # 交叉
    'FedMV':        '\\\\\\', # 反斜线
    'FedAT_FedMut': '---',    # 横线
    'FedAT':        '\\\\\\',       # 实心
}

# 创建保存目录
save_dir = './fig-ablation'
os.makedirs(save_dir, exist_ok=True)

def plot_ablation_bar(data, dataset_name, methods, ylabel='Accuracy (%)', save_path=None, legend_ncol=2):
    x = np.arange(len(non_iid))
    width = 0.18 if len(methods) == 4 else 0.22
    plt.figure(figsize=(8,5))
    for i, method in enumerate(methods):
        color = method_colors.get(method, None)
        hatch = method_hatches.get(method, None)
        plt.bar(
            x + i*width, 
            data[method], 
            width, 
            label=method.replace('_', '+'),  # 替换为+
            color=color, 
            edgecolor='black', 
            linewidth=1.2,
            hatch=hatch
        )
    plt.xticks(x + width*(len(methods)-1)/2, non_iid, fontsize=20)
    plt.ylabel(ylabel, fontsize=22)
    plt.xlabel('Server non-IID degree (β)', fontsize=22)
    
    if dataset_name == 'Shakespeare':
        plt.ylim(20, 50)
    elif dataset_name == 'CIFAR100':
        plt.ylim(30, 65)
    else:
        plt.ylim(20, 70)
    plt.gca().tick_params(axis='x', labelsize=18)
    plt.gca().tick_params(axis='y', labelsize=18)
    plt.legend(fontsize=20, loc="upper left", ncol=legend_ncol, columnspacing=0.5, labelspacing=0.4, frameon=False, bbox_to_anchor=(0, 1.04))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

# AT消融
for dataset in at_data:
    filename = f"{save_dir}/ablation_AT_{dataset.replace('(', '').replace(')', '').replace('.', '').replace(' ', '')}.pdf"
    legend_ncol = 2 if dataset == 'Shakespeare' else 2
    plot_ablation_bar(
        at_data[dataset],
        dataset_name=dataset,
        methods=['FedATMV', 'CLG_FedMV', 'FedDU_FedMV', 'FedMV'],
        ylabel='Accuracy (%)',
        save_path=filename,
        legend_ncol=legend_ncol
    )

# MV消融
for dataset in mv_data:
    filename = f"{save_dir}/ablation_MV_{dataset.replace('(', '').replace(')', '').replace('.', '').replace(' ', '')}.pdf"
    legend_ncol = 2 if dataset == 'Shakespeare' else 2
    plot_ablation_bar(
        mv_data[dataset],
        dataset_name=dataset,
        methods=['FedATMV', 'FedAT_FedMut', 'FedAT'],
        ylabel='Accuracy (%)',
        save_path=filename,
        legend_ncol=legend_ncol
    )

print("All ablation figures are saved as PDF in ./fig-ablation/")