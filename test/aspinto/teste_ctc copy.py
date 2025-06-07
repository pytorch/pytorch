import numpy as np
import torch

# --------- Parte 1: gerar os dados como no exemplo ---------
torch.manual_seed(0)
T, N, C = 50, 1, 20
log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, C, (N, 10), dtype=torch.long)
input_lengths = torch.tensor([T])
target_lengths = torch.tensor([10])

# --------- Parte 2: computar CTC no PyTorch (referência) ---------
loss_fn = torch.nn.CTCLoss()
loss_torch = loss_fn(log_probs, targets, input_lengths, target_lengths)
loss_torch.backward()
grads_torch = log_probs.grad.detach().squeeze(1).numpy()

# --------- Parte 3: sua implementação numpy ---------
def log_sum_exp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return max(a, b) + np.log1p(np.exp(-abs(a - b)))

def ctc_loss_full(log_probs, target, reduction="mean"):
    """
    log_probs: numpy array (T, C) de log-softmax
    target: list de ints (sem blanks)
    reduction: 'none' | 'mean' | 'sum'
    """
    T, C = log_probs.shape
    N = len(target)
    S = 2 * N + 1

    l = np.full(S, 0, dtype=int)
    l[1::2] = target

    alpha = np.full((T, S), -np.inf, dtype=np.float64)
    beta = np.full((T, S), -np.inf, dtype=np.float64)

    alpha[0, 0] = log_probs[0, l[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, l[1]]

    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1, s]
            b = alpha[t - 1, s - 1] if s - 1 >= 0 else -np.inf
            c = -np.inf
            if s - 2 >= 0 and l[s] != 0 and l[s] != l[s - 2]:
                c = alpha[t - 1, s - 2]
            alpha[t, s] = log_probs[t, l[s]] + log_sum_exp(log_sum_exp(a, b), c)

    beta[T - 1, S - 1] = log_probs[T - 1, l[S - 1]]
    if S > 1:
        beta[T - 1, S - 2] = log_probs[T - 1, l[S - 2]]

    for t in range(T - 2, -1, -1):
        for s in range(S):
            a = beta[t + 1, s] + log_probs[t + 1, l[s]]
            b = beta[t + 1, s + 1] + log_probs[t + 1, l[s + 1]] if s + 1 < S else -np.inf
            c = -np.inf
            if s + 2 < S and l[s] != 0 and l[s] != l[s + 2]:
                c = beta[t + 1, s + 2] + log_probs[t + 1, l[s + 2]]
            beta[t, s] = log_sum_exp(log_sum_exp(a, b), c)

    logZ = log_sum_exp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    loss = -logZ

    if reduction == "mean":
        loss /= N
    elif reduction == "none":
        pass  # retorna perda total (sem normalização)
    elif reduction == "sum":
        pass  # mesma coisa aqui (só 1 amostra)

    # Max grad diff: 3.9208846
    grads = np.exp(log_probs)  # dL/dy = y - posterior
    for t in range(T):
        for s in range(S):
            c = l[s]
            contrib = np.exp(alpha[t, s] + beta[t, s] - log_probs[t, c] - logZ)
            grads[t, c] -= contrib

    return loss, grads

# --------- Parte 4: rodar implementação numpy ---------
log_probs_np = log_probs.detach().squeeze(1).numpy()
target_np = targets[0].tolist()
loss_np, grads_np = ctc_loss_full(log_probs_np, target_np)

# --------- Parte 5: comparação ---------
print("CTC Loss PyTorch:", loss_torch.detach().item())
print("CTC Loss NumPy  :", loss_np)
print("Loss difference :", abs(loss_np - loss_torch.detach().item()))
print("Max grad diff   :", np.max(np.abs(grads_torch - grads_np)))
