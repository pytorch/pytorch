# torchrun --nproc_per_node=2 test.py
import os

import torch
from torch.distributed import init_process_group
from torch.distributed._tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh


# Check if CUDA is available
if torch.cuda.is_available():
    print("cuda is available")
    device_type = "cuda"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
else:
    print("cuda is not available")
    device_type = "cpu"
    init_process_group(backend="gloo")

device_mesh = init_device_mesh(
    device_type=device_type, mesh_shape=(2,), mesh_dim_names=("tp",)
)


def test_case(plan=1):
    original_tensor = torch.tensor([1.0, 2.0], device=device_type, requires_grad=True)
    original_tensor = distribute_tensor(original_tensor, device_mesh, [Shard(0)])
    out = original_tensor.sum()
    if plan == 1:
        out = DTensor.from_local(
            out.to_local(grad_placements=[Replicate()]),
            out.device_mesh,
            out.placements,
            run_check=False,
        )
    elif plan == 2:
        pass  # do nothing

    loss = out.full_tensor().sum()
    loss.backward()
    grad_original = original_tensor.grad
    grad = original_tensor.grad.full_tensor()
    return loss, grad, grad_original


loss1, grad1, grad_original1 = test_case(1)
loss2, grad2, grad_original2 = test_case(2)
print(f"Plan 1 Loss: {loss1} Grad: {grad1} {grad_original1}")
print(f"Plan 2 Loss: {loss2} Grad: {grad2} {grad_original2}")
