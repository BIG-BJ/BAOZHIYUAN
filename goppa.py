import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import galois

# ===== 生成 Goppa 校验矩阵 H（兼容 galois 0.4.6） =====

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


# ===== 攻击部分 =====

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

# 判断攻击是否成功
def evaluate_success(e, top_indices):
    return any(e[i] == 1 for i in top_indices)

# 主函数：执行多次实验，统计成功率，并绘制单次攻击效果
def experiment(m, n, t, noise_std, trials=20, top_k=10, plot_examples=3):
    H = generate_goppa_H(m, n, t)
    successes = 0
    plotted = 0
    for trial in range(trials):
        e = generate_error_vector(n, t)
        syndrome = compute_syndrome(H, e)
        leakage = simulate_leakage(syndrome, noise_std)
        correlations, top_indices = perform_attack(H, leakage, top_k)
        if evaluate_success(e, top_indices):
            successes += 1

        # 可视化前几次攻击效果
        if plotted < plot_examples:
            visualize_attack(correlations, e, top_indices)
            plotted += 1

    return successes / trials

# 绘制单次攻击效果图
def visualize_attack(correlations, e, top_indices):
    n = len(correlations)
    x = np.arange(n)
    plt.figure(figsize=(10, 5))
    plt.plot(x, correlations, label="|ρ_j|")
    true_pos = np.where(e == 1)[0]
    plt.scatter(true_pos, correlations[true_pos], color='red', label='Истинные единичные позиции')
    plt.scatter(top_indices, correlations[top_indices], color='green', marker='^', label='Предсказанные позиции')
    plt.title("Визуализация атаки: корреляция по позициям")
    plt.xlabel("Индекс позиции")
    plt.ylabel("Абсолютное значение корреляции |ρ_j|")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===== 实验参数 =====

m = 16
n = 2688
t = 150
top_k = 128
trials = 100
noise_levels = np.linspace(0, 2, 6)
success_rates = []

for noise in noise_levels:
    print(f" σ = {noise:.2f} ...")
    rate = experiment(m, n, t, noise_std=noise, trials=trials, top_k=top_k)
    success_rates.append(rate)

# ===== 绘图 =====

plt.figure(figsize=(10, 6))
plt.plot(noise_levels, success_rates, marker='o', label="Вероятность успешной атаки")
plt.xlabel("Уровень шума (σ)")
plt.ylabel("Вероятность успешной атаки")
plt.title("Успешность горизонтальной атаки при разных уровнях шума")
plt.grid(True)
plt.legend()
plt.show()
