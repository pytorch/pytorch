import torch
import random

def gen_two_four_sparse_mask(r, c, dtype=torch.float16, device="cuda"):
    def random_mask_choice(i=None):
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
        return choices[random.randint(0, len(choices) - 1) if i is None else i]

    mask_entries = [random_mask_choice() for i in range(r * c // 4)]
    return (
        torch.tensor(mask_entries, dtype=dtype, device=device).view(r, c).contiguous()
    )
import torch


def verify_tensor_is_block_sparse(tensor, block_size=4, zeros_per_block=2):
    if not tensor.is_contiguous():
        raise Exception("Tensor is not contiguous")
    contiguous_flattened = tensor.view(-1)
    # okay if not the same tensor since values will be the same
    block_tensor = contiguous_flattened.reshape(-1, block_size)
    assert ((block_tensor == 0).sum(dim=1) == zeros_per_block).all()


def print_model_sparsity(model, verbose=True):
    total_params = 0
    total_nonzero_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nonzero = torch.count_nonzero(param.data)
            if verbose:
                size = ", ".join(map(str, param.data.size()))
                print(
                    f"{name: <60}| {size: <15} |{param.data.numel():10d}|{nonzero:10d}"
                )
            total_nonzero_params += nonzero
            total_params += param.data.numel()
    print(f"=== Total Number of Parameters: {total_params} ===")
    print(f"=== Total Number of Non-Zero Parameters: {total_nonzero_params} ===")
    return total_params


