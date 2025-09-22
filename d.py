import torch

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True, mode='reduce-overhead')
def get_mask(W: torch.Tensor, percentage_nonzeros: torch.Tensor):
    total_elements = W.numel()
    k = total_elements * percentage_nonzeros
    top_k_indices = torch.topk(torch.abs(W).flatten(), k.int())[1]
    mask = torch.zeros(total_elements, dtype=torch.bool, device=W.device)
    mask.scatter_(0, top_k_indices, True)
    mask = mask.view(W.shape)
    return mask

x = torch.randn((128, 64), device='cuda')
p = torch.tensor(0.50, device='cuda')
get_mask(x, p)
