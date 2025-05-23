import torch
import numpy as np
import pandas as pd

# Funções auxiliares
def logsumexp3(a, b, c):
    m = max(a, b, c)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))

def get_target_prime(targets, s, blank):
    return blank if s % 2 == 0 else targets[s // 2]

# Configuração
T = 20
C = 5
target_len = 4
blank = 0

torch.manual_seed(42)
log_probs = torch.randn(T, C).log_softmax(1)
targets = torch.randint(1, C, (target_len,))
S = 2 * target_len + 1

# Expand target
l = [blank]
for t in targets:
    l.append(t.item())
    l.append(blank)

# Alpha
alpha = np.full((T, S), -np.inf, dtype=np.float32)
alpha[0, 0] = log_probs[0, blank].item()
if S > 1:
    alpha[0, 1] = log_probs[0, l[1]].item()

for t in range(1, T):
    for s in range(S):
        a = alpha[t - 1, s]
        b = alpha[t - 1, s - 1] if s > 0 else -np.inf
        c = -np.inf
        if s > 1 and l[s] != blank and l[s] != l[s - 2]:
            c = alpha[t - 1, s - 2]
        alpha[t, s] = logsumexp3(a, b, c) + log_probs[t, l[s]].item()

# Beta
beta = np.full((T, S), -np.inf, dtype=np.float32)
beta[T - 1, S - 1] = log_probs[T - 1, l[S - 1]].item()
if S > 1:
    beta[T - 1, S - 2] = log_probs[T - 1, l[S - 2]].item()

for t in reversed(range(T - 1)):
    for s in range(S):
        a = beta[t + 1, s] + log_probs[t + 1, l[s]].item()
        b = beta[t + 1, s + 1] + log_probs[t + 1, l[s + 1]].item() if s + 1 < S else -np.inf
        c = -np.inf
        if s + 2 < S and l[s] != blank and l[s] != l[s + 2]:
            c = beta[t + 1, s + 2] + log_probs[t + 1, l[s + 2]].item()
        beta[t, s] = logsumexp3(a, b, c)

# logZ
l1 = alpha[T - 1, S - 1]
l2 = alpha[T - 1, S - 2] if S > 1 else -np.inf
m = max(l1, l2)
logZ = m + np.log(np.exp(l1 - m) + np.exp(l2 - m))

# log_probs selecionado por caminho
log_probs_expanded = np.array([[log_probs[t, l[s]].item() for s in range(S)] for t in range(T)])
alpha_beta = alpha + beta

# Exportar
def export_csv(matrix, name):
    df = pd.DataFrame(matrix, columns=[f"s={i}" for i in range(S)])
    df.insert(0, "t", list(range(T)))
    df.to_csv(f"{name}.csv", index=False)

export_csv(alpha, "alpha")
export_csv(beta, "beta")
export_csv(alpha_beta, "alpha_plus_beta")
export_csv(log_probs_expanded, "log_probs_expanded")

print("Arquivos CSV gerados: alpha.csv, beta.csv, alpha_plus_beta.csv, log_probs_expanded.csv")
print("Você pode abrir esses arquivos no Excel para analisar")
