import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import galois
import os
from joblib import Parallel, delayed

# ===== 保存图像函数 =====
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
    # 自动分组保存
    save_dir = os.path.join(save_dir, f"noise_{noise_std:.2f}")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"attack_plot_{plot_id}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] 图像已保存至：{filename}")

# ===== 生成 Goppa 校验矩阵 H（兼容 galois 0.4.6） =====

def generate_goppa_H(m, n, t):
    q = 2 ** m
    GF = galois.GF(q)

    while True:
        g = galois.irreducible_poly(q, t)  # 旧版本调用
        # 替代 replace=False：手动去重生成 L
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


# ===== 攻击部分（不变） =====
# 生成稀疏错误向量 e
def generate_error_vector(n, t):
    e = np.zeros(n, dtype=int)
    pos = np.random.choice(n, size=t, replace=False)
    e[pos] = 1
    return e

# 计算 syndrome
def compute_syndrome(H, e):
    return np.mod(H @ e, 2)

# 模拟泄漏轨迹（海明重量+加噪声）
def simulate_leakage(syndrome, noise_std):
    # 使用 syndrome 的 Hamming weight 模拟泄漏 + 噪声
    leakage = np.array(syndrome) + np.random.normal(0, noise_std, len(syndrome))
    return leakage

# 并行攻击函数
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

"""
# 执行攻击（计算所有位置的相关系数）
def perform_attack(H, leakage, top_k=10):
    n = H.shape[1]
    correlations = []
    for j in range(n):
        e_j = np.zeros(n, dtype=int)
        e_j[j] = 1
        s_j = np.mod(H @ e_j, 2)
        corr, _ = pearsonr(s_j, leakage)
        correlations.append(abs(corr))
    top_indices = np.argsort(correlations)[-top_k:]
    return np.array(correlations), top_indices
"""

# 判断攻击是否成功
def evaluate_success(e, top_indices):
    return any(e[i] == 1 for i in top_indices)

# 主实验逻辑
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

"""
# 主函数：执行多次实验，统计成功率，并绘制单次攻击效果
def experiment(m, n, t, noise_std, trials=20, top_k=10, plot_examples=3,):
    H = generate_goppa_H(m, n, t)
    successes = 0
    plotted = 0
    for trial in range(trials):
        e = generate_error_vector(n, t)
        syndrome = compute_syndrome(H, e)
        leakage = simulate_leakage(syndrome, noise_std)
        correlations, top_indices = perform_attack(H, leakage, top_k,)
        if evaluate_success(e, top_indices):
            successes += 1

        # 可视化前几次攻击效果
        if plotted < plot_examples:
            visualize_attack(correlations, e, top_indices)
            plotted += 1

    return successes / trials


#绘制单次攻击图
def visualize_attack(correlations, e, top_indices):
    n = len(correlations)
    x = np.arange(n)
    plt.figure(figsize=(10, 5))
    plt.plot(x, correlations, label="|ρ_j|")
    true_pos = np.where(e == 1)[0]
    plt.scatter(true_pos, correlations[true_pos], color='red', label='真实错误位')
    plt.scatter(top_indices, correlations[top_indices], color='green', marker='^', label='Top-k预测')
    plt.title("单次攻击效果")
    plt.xlabel("位置索引 j")
    plt.ylabel("相关性 |ρ_j|")
    plt.legend()
    plt.grid(True)
    plt.show()
"""

# ===== 实验参数 =====

m = 2
n = 250
t = 10
top_k = 128
trials = 20
n_jobs = -1
noise_levels = np.linspace(0, 2, 6)
success_rates = []

for noise in noise_levels:
    print(f" σ = {noise:.2f} ...")
    rate = experiment(m, n, t, noise_std=noise, trials=trials, top_k=top_k,n_jobs=n_jobs)
    success_rates.append(rate)

# ===== 绘图 =====

plt.figure(figsize=(10, 6))
plt.plot(noise_levels, success_rates, marker='o', label="攻击成功率")
plt.xlabel("噪声 σ")
plt.ylabel("成功率")
plt.title("攻击成功率随噪声变化")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("attack_success_vs_noise.png", dpi=300, bbox_inches='tight')
plt.show()
