
import runpy, json, datetime, csv, os
from pathlib import Path

# 可调参数 ---------------------------------------------------------------------
n_repeat   = 1                                   # 每个配置重复次数
src_script = "script_v16.0_shake_250519.py"            # 实际实验脚本
out_root   = Path("output")         # 统一输出目录
# ------------------------------------------------------------------------------

timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir    = out_root / f"multi_run_v2.0_{timestamp}"
exp_dir.mkdir(parents=True, exist_ok=True)

# 三种服务器数据配置
cfgs = [
    # dict(name="srvDir0p1", server_iid=False, server_dir=0.1),
    dict(name="srvDir1p0", server_iid=False, server_dir=1.0),
    dict(name="srvIID"   , server_iid=True , server_dir=0.1),   # dir 数值随意
]

# ---------- CSV：仅记录 final ----------
summary_csv = exp_dir / "summary.csv"
with summary_csv.open("w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["cfg", "algo", "metric", "value"])  # 表头

    # ================= 主循环 =================
    for cfg in cfgs:
        print(f"\n==========  Running config: {cfg['name']}  ==========")
        cfg_dir = exp_dir / cfg["name"]
        cfg_dir.mkdir(exist_ok=True)

        for r in range(n_repeat):
            print(f"----- repeat {r+1}/{n_repeat} -----")

            # 将 server 参数注入脚本
            globs = runpy.run_path(
                src_script,
                init_globals=dict(
                    server_iid = cfg["server_iid"],
                    server_dir = cfg["server_dir"],
                )
            )

            results_test_acc   = globs["results_test_acc"]
            results_train_loss = globs["results_train_loss"]

            # ---------- JSON 保存全部轮次 ----------
            (cfg_dir / f"run{r+1:02d}_results_test_acc.json"
             ).write_text(json.dumps(results_test_acc,  indent=2))
            (cfg_dir / f"run{r+1:02d}_results_train_loss.json"
             ).write_text(json.dumps(results_train_loss, indent=2))

            # ---------- CSV 只写 final ----------
            for algo in results_test_acc.keys():
                acc_final  = results_test_acc[algo][-1]
                loss_final = results_train_loss[algo][-1]
                writer.writerow([cfg["name"], algo, "acc_final" , acc_final])
                writer.writerow([cfg["name"], algo, "loss_final", loss_final])

print(f"\n全部数据保存于: {exp_dir.resolve()}")