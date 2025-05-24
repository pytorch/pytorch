import torch

# Seed para reprodutibilidade
torch.manual_seed(0)

# CPU (referência)
log_probs_cpu = torch.randn(50, 1, 20, device="cpu").log_softmax(2).detach().requires_grad_()
targets_cpu = torch.randint(1, 20, (1, 10), dtype=torch.long)
input_lengths = torch.tensor([50], dtype=torch.long)
target_lengths = torch.tensor([10], dtype=torch.long)

# MPS (GPU)
log_probs_mps = log_probs_cpu.detach().clone().to("mps").requires_grad_()
targets_mps = targets_cpu.to("mps")

# print(log_probs_mps)
# print(targets_mps)

# Perda CTC
loss_fn = torch.nn.CTCLoss()

# print("shape cpu:", log_probs_cpu.shape)     # ex: (50, 1, 20)
# print("strides cpu:", log_probs_cpu.stride())  # ex: (20, 20, 1)

# print("shape mps:", log_probs_mps.shape)     # ex: (50, 1, 20)
# print("strides mps:", log_probs_mps.stride())  # ex: (20, 20, 1)
# print("log_probs_mps[0, 0, 0].detach().item():", log_probs_mps[0, 0, 0].detach().item())  # offset = 0
# print("log_probs_mps[0, 0, 1].detach().item():", log_probs_mps[0, 0, 1].detach().item())  # offset = 1
# print("log_probs_mps[0, 0, 2].detach().item():", log_probs_mps[0, 0, 2].detach().item())  # offset = 2
# print("log_probs_mps[0, 0, 3].detach().item():", log_probs_mps[0, 0, 3].detach().item())  # offset = 3

# CPU forward + backward
loss_cpu = loss_fn(log_probs_cpu, targets_cpu, input_lengths, target_lengths)
loss_cpu.backward()
grads_cpu = log_probs_cpu.grad.clone()

# MPS forward + backward
loss_mps = loss_fn(log_probs_mps, targets_mps, input_lengths, target_lengths)
loss_mps.backward()
grads_mps = log_probs_mps.grad.to("cpu")  # para comparação
# print(grads_mps)
# print(grads_cpu)

# Comparações
print("Loss CPU:", loss_cpu.detach().item())
print("Loss MPS:", loss_mps.detach().item())
print("Loss diff:", abs(loss_cpu.detach().item() - loss_mps.detach().item()))
print("Max grad diff:", (grads_cpu - grads_mps).abs().max().item())
print("Loss close:", torch.allclose(loss_cpu.detach(), loss_mps.detach(), rtol=1e-4, atol=1e-5))
print("Gradients close:", torch.allclose(grads_cpu, grads_mps, rtol=1e-4, atol=1e-5))
