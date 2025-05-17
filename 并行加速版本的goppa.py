import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import galois
import os
from joblib import Parallel, delayed  # 并行计算模块

# ===== 图像保存函数 =====
def plot_attack_result(correlations, e, top_indices, plot_id=0, noise_std=0.0, save_dir="attack_plots"):
    n = len(correlations)
    x = np.arange(n)
    plt.figure(figsize=(10, 5))
    plt.plot(x, correlations, label="|ρ_j|", color='blue', linewidth=1.0)
    true_pos = np.where(e == 1)[0]
    plt.scatter(true_pos, correlations[true_pos], color='red', label='真实错误位')
    plt.scatter(top_indices, correlations[top_indices], color='green', marker='^', label='Top-k预测')
    plt.title("单次攻击效果")
    plt.xlabel("位置索引 j")
    plt.ylabel("相关性 |ρ_j|")
    plt.legend()
    plt.grid(True)
    save_dir = os.path.join(save_dir, f"noise_{noise_std:.2f}")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"attack_plot_{plot_id}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] 图像已保存至：{filename}")

# ===== Goppa 校验矩阵 =====
def generate_goppa_H(m, n, t):
    q = 2 ** m
    GF = galois.GF(q)
    while True:
        g = galois.irreducible_poly(q, t)
        while True:
            L = GF.Random(n)
            if len(set(L.tolist())) == n:
                break
        if np.all(g(L) != 0):
            break
    H = np.zeros((t, n), dtype=GF)
    g_L_inv = 1 / g(L)
    for i in range(t):
        H[i, :] = (L ** i) * g_L_inv
    H_bin = np.vstack([h.vector() for h in H.flatten()])
    H_bin = H_bin.reshape(t * m, n)
    return H_bin

# ===== 攻击流程辅助函数 =====
def generate_error_vector(n, t):
    e = np.zeros(n, dtype=int)
    pos = np.random.choice(n, size=t, replace=False)
    e[pos] = 1
    return e

def compute_syndrome(H, e):
    return np.mod(H @ e, 2)

def simulate_leakage(syndrome, noise_std):
    return np.array(syndrome) + np.random.normal(0, noise_std, len(syndrome))

# ✅ 并行攻击函数
def perform_attack_parallel(H, leakage, top_k=10, n_jobs=-1):
    n = H.shape[1]

    def compute_corr(j):
        e_j = np.zeros(n, dtype=int)
        e_j[j] = 1
        s_j = np.mod(H @ e_j, 2)
        corr, _ = pearsonr(s_j, leakage)
        return abs(corr)

    correlations = Parallel(n_jobs=n_jobs)(
        delayed(compute_corr)(j) for j in range(n)
    )
    correlations = np.array(correlations)
    top_indices = np.argsort(correlations)[-top_k:]
    return correlations, top_indices

def evaluate_success(e, top_indices):
    return any(e[i] == 1 for i in top_indices)

# ✅ 主实验逻辑
def experiment(m, n, t, noise_std, trials=20, top_k=10, plot_examples=3, n_jobs=-1):
    H = generate_goppa_H(m, n, t)
    successes = 0
    plotted = 0
    for trial in range(trials):
        e = generate_error_vector(n, t)
        syndrome = compute_syndrome(H, e)
        leakage = simulate_leakage(syndrome, noise_std)
        correlations, top_indices = perform_attack_parallel(H, leakage, top_k=top_k, n_jobs=n_jobs)
        if evaluate_success(e, top_indices):
            successes += 1
        if plotted < plot_examples:
            plot_attack_result(correlations, e, top_indices, plot_id=plotted, noise_std=noise_std)
            plotted += 1
    return successes / trials

# ===== 参数配置与实验启动 =====
m = 2
n = 250
t = 10
top_k = 128
trials = 20
n_jobs = -1  # 使用所有核心
noise_levels = np.linspace(0, 2, 6)
success_rates = []

for noise in noise_levels:
    print(f"▶ 正在运行 σ = {noise:.2f} ...")
    rate = experiment(m, n, t, noise_std=noise, trials=trials, top_k=top_k, n_jobs=n_jobs)
    success_rates.append(rate)

# ===== 绘制成功率曲线 =====
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, success_rates, marker='o', label="攻击成功率")
plt.xlabel("噪声 σ")
plt.ylabel("成功率")
plt.title("攻击成功率随噪声变化（并行加速）")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("attack_success_vs_noise.png", dpi=300, bbox_inches='tight')
plt.show()
