
import runpy, json, datetime, csv, os
from pathlib import Path

# 可调参数 ---------------------------------------------------------------------
n_repeat   = 1                                   # 每个配置重复次数
# src_script = "script_v18.0_cifar10_mu_250523.py"            # 实际实验脚本
# src_script = "script_v18.0_cifar10_rho_250523.py"
src_script = "script_v18.0_cifar10_theta_250523.py"  
out_root   = Path("output")         # 统一输出目录
# ------------------------------------------------------------------------------

timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir    = out_root / f"multi_run_v4.0_{timestamp}"
exp_dir.mkdir(parents=True, exist_ok=True)

# 三种服务器数据配置
# cfgs = [
#     dict(name="mu0", du_C=0),
#     dict(name="mu1", du_C=1),
#     dict(name="mu3", du_C=3),
#     dict(name="mu5", du_C=5),
#     dict(name="mu7", du_C=7),
#     dict(name="mu9", du_C=9),
# ]

# cfgs = [
#     dict(name="rho0", radius=0),
#     dict(name="rho2", radius=2),
#     dict(name="rho4", radius=4),
#     dict(name="rho6", radius=6),
#     dict(name="rho8", radius=8),
#     dict(name="rho10", radius=10),
# ]

cfgs = [
    dict(name="theta0", scal_ratio=0),
    dict(name="theta1", scal_ratio=0.1),
    dict(name="theta3", scal_ratio=0.3),
    dict(name="theta5", scal_ratio=0.5),
    dict(name="theta7", scal_ratio=0.7),
    dict(name="theta9", scal_ratio=0.9),
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
                    # du_C = cfg["du_C"],
                    # radius = cfg["radius"],
                    scal_ratio = cfg["scal_ratio"],
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