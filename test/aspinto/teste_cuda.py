import torch

# Seed para reprodutibilidade
torch.manual_seed(0)

# CPU (referência)
log_probs_cpu = torch.randn(50, 1, 20, device="cpu").log_softmax(2).detach().requires_grad_()
targets_cpu = torch.randint(1, 20, (1, 10), dtype=torch.long)
input_lengths = torch.tensor([50], dtype=torch.long)
target_lengths = torch.tensor([10], dtype=torch.long)

# CUDA (GPU)
log_probs_cuda = log_probs_cpu.detach().clone().to("cuda").requires_grad_()
targets_cuda = targets_cpu.to("cuda")

# Perda CTC
loss_fn = torch.nn.CTCLoss()

# CPU forward + backward
loss_cpu = loss_fn(log_probs_cpu, targets_cpu, input_lengths, target_lengths)
loss_cpu.backward()
grads_cpu = log_probs_cpu.grad.clone()

# CUDA forward + backward
loss_cuda = loss_fn(log_probs_cuda, targets_cuda, input_lengths, target_lengths)
loss_cuda.backward()
grads_cuda = log_probs_cuda.grad.to("cpu")  # para comparação

# Comparações
print("Loss CPU:", loss_cpu.item())
print("Loss CUDA:", loss_cuda.item())
print("Loss diff:", abs(loss_cpu.item() - loss_cuda.item()))
print("Max grad diff:", (grads_cpu - grads_cuda).abs().max().item())
print("Loss close:", torch.allclose(loss_cpu, loss_cuda, rtol=1e-4, atol=1e-5))
print("Gradients close:", torch.allclose(grads_cpu, grads_cuda, rtol=1e-4, atol=1e-5))
