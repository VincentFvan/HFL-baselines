

import runpy, datetime, json, csv, numpy as np
from pathlib import Path

# ========== 可调参数 ==========
n_repeat   = 1                    # 重复次数
src_script = "atmv.py"      # 原脚本文件名
out_root   = Path("output")       # 统一输出目录
# ==============================

timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir    = out_root / f"multi_run_{timestamp}"
exp_dir.mkdir(parents=True, exist_ok=True)

all_runs_test  = []   # list[dict]
all_runs_loss  = []

# ---------- 多次执行 ----------
for r in range(n_repeat):
    print(f"\n===== Run {r+1}/{n_repeat} =====")
    # runpy.run_path 会把脚本当作 __main__ 来执行，
    # 返回其 *最终* 的全局变量字典
    globs = runpy.run_path(src_script)

    # 拿到我们在 exp_single.py 里 return 出来的两个结果
    results_test_acc  = globs["results_test_acc"]
    results_train_loss = globs["results_train_loss"]

    # 保存本次原始结果
    run_dir = exp_dir / f"run_{r+1:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results_test_acc.json").write_text(
        json.dumps(results_test_acc, indent=2))
    (run_dir / "results_train_loss.json").write_text(
        json.dumps(results_train_loss, indent=2))

    all_runs_test.append(results_test_acc)
    all_runs_loss.append(results_train_loss)

# ---------- 统计均值/标准差 ----------
def mean_std(x):            # 无偏标准差
    arr = np.asarray(x, dtype=float)
    return arr.mean(), arr.std(ddof=1)

def safe_pick(seq, idx, fallback="last"):
    """
    当 seq 长度 <= idx 时的安全取值：
      fallback = "last"  →  使用最后一个元素
      fallback = None    →  返回 np.nan
    """
    if len(seq) > idx:
        return seq[idx]
    if fallback == "last":
        return seq[-1]
    return np.nan

ROUND_K = 19          # round-20 的索引（0-based）

summary_csv = exp_dir / "summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    header = ["algo_metric", "pos"] + \
             [f"run{i+1}" for i in range(n_repeat)] + ["mean", "std"]
    writer.writerow(header)

    algos = all_runs_test[0].keys()
    for algo in algos:
        # round-20
        acc20  = [safe_pick(run[algo], ROUND_K)  for run in all_runs_test]
        loss20 = [safe_pick(run[algo], ROUND_K)  for run in all_runs_loss]
        # final
        accF   = [run[algo][-1] for run in all_runs_test]
        lossF  = [run[algo][-1] for run in all_runs_loss]

        for tag, vals in (("acc_round20",  acc20),
                          ("acc_final",    accF),
                          ("loss_round20", loss20),
                          ("loss_final",   lossF)):
            m, s = mean_std(vals)
            writer.writerow([algo, tag, *vals, m, s])

print(f"\n全部结果已保存到 {exp_dir.resolve()}")