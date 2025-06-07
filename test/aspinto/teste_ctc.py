import numpy as np
import torch
import pandas as pd

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
def logsumexp3(a, b, c):
    """Efficient logsumexp of three values in log-space"""
    max_val = max(a, max(b, c))
    if max_val == -np.inf:
        return -np.inf
    return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val))

def get_target_prime(targets, batch_offset, stride, s, blank=0):
    if s % 2 == 0:
        return blank
    else:
        return targets[batch_offset + (s // 2) * stride]

def get_target_prime_simple(targets, s, blank=0):
    return blank if s % 2 == 0 else targets[s // 2]

def compute_ctc_loss_and_grad_batch(log_probs, target, blank=0, reduction="mean", zero_infinity=False):
    T, C = log_probs.shape
    N = len(target)
    S = 2 * N + 1

    # Expand target with blanks
    l = np.full(S, blank, dtype=np.int32)
    for i in range(N):
        l[2 * i + 1] = target[i]

    print("l:", l)

    alpha = np.full((T, S), -np.inf, dtype=np.float32)
    beta = np.full((T, S), -np.inf, dtype=np.float32)
    grads = np.exp(log_probs).astype(np.float32)

    alpha[0, 0] = log_probs[0, blank]
    if S > 1:
        alpha[0, 1] = log_probs[0, l[1]]

    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1, s]
            b = alpha[t - 1, s - 1] if s > 0 else -np.inf
            c = -np.inf
            if s > 1 and l[s] != blank and l[s] != l[s - 2]:
                c = alpha[t - 1, s - 2]
            alpha[t, s] = logsumexp3(a, b, c) + log_probs[t, l[s]]

    beta[T - 1, S - 1] = log_probs[T - 1, l[S - 1]]
    if S > 1:
        beta[T - 1, S - 2] = log_probs[T - 1, l[S - 2]]

    for t in range(T - 2, -1, -1):
        for s in range(S):
            a = beta[t + 1, s] + log_probs[t + 1, l[s]]
            b = beta[t + 1, s + 1] + log_probs[t + 1, l[s + 1]] if s + 1 < S else -np.inf
            c = -np.inf
            if s + 2 < S and l[s] != blank and l[s] != l[s + 2]:
                c = beta[t + 1, s + 2] + log_probs[t + 1, l[s + 2]]
            beta[t, s] = logsumexp3(a, b, c)

    # Converter para DataFrame (2D para Excel)
    # log_probs_df = pd.DataFrame(log_probs, index=range(log_probs.shape[0]), columns=[f"s={s}" for s in range(log_probs.shape[1])])
    # log_probs_df.index.name = "t"
    # targets_df = pd.DataFrame(targets, index=range(targets.shape[0]), columns=[f"s={s}" for s in range(targets.shape[1])])
    # targets_df.index.name = "t"
    # alpha_df = pd.DataFrame(alpha, index=range(alpha.shape[0]), columns=[f"s={s}" for s in range(alpha.shape[1])])
    # alpha_df.index.name = "t"
    # beta_df = pd.DataFrame(beta, index=range(beta.shape[0]), columns=[f"s={s}" for s in range(beta.shape[1])])
    # beta_df.index.name = "t"

    # log_probs_df.reset_index(inplace=True)
    # alpha_df.reset_index(inplace=True)
    # beta_df.reset_index(inplace=True)

    # Criar um Excel com duas abas: Alpha e Beta
    # with pd.ExcelWriter("alpha_beta.xlsx", engine="openpyxl") as writer:
    #     log_probs_df.to_excel(writer, sheet_name="log_probs", index=False)
    #     targets_df.to_excel(writer, sheet_name="targets", index=False)
    #     alpha_df.to_excel(writer, sheet_name="Alpha", index=False)
    #     beta_df.to_excel(writer, sheet_name="Beta", index=False)

    print(alpha)

    logZ = max(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    logZ = logZ + np.log(np.exp(alpha[T - 1, S - 1] - logZ) + np.exp(alpha[T - 1, S - 2] - logZ))

    loss = -logZ
    if reduction == "mean" and N > 0:
        loss /= N

    # Compute gradient
    for s in range(S):
        c = l[s]
        for t in range(T):
            grads[t, c] -= np.exp(alpha[t, s] + beta[t, s] - logZ - log_probs[t, c])

    return loss, grads

# Wrapper for batching
def ctc_loss_optimized(log_probs_batch, targets_batch, blank=0, reduction="mean"):
    B = len(targets_batch)
    losses = []
    grads_batch = []
    for b in range(B):
        loss, grads = compute_ctc_loss_and_grad_batch(log_probs_batch[b], targets_batch[b], blank, reduction)
        losses.append(loss)
        grads_batch.append(grads)
    return np.mean(losses) if reduction == "mean" else np.sum(losses), grads_batch

# --------- Parte 4: rodar implementação numpy ---------
log_probs_np = log_probs.detach().squeeze(1).numpy()
target_np = targets[0].tolist()
loss_np, grads_np = ctc_loss_optimized([log_probs_np], [target_np])

# --------- Parte 5: comparação ---------
print("CTC Loss PyTorch:", loss_torch.detach().item())
print("CTC Loss NumPy  :", loss_np)
print("Loss difference :", abs(loss_np - loss_torch.detach().item()))
print("Max grad diff   :", np.max(np.abs(grads_torch - grads_np)))
