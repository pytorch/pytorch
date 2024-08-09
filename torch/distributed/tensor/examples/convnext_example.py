# mypy: allow-untyped-defs
"""
The following example demonstrates how to train a ConvNeXt model
with intermediate activations sharded across mutliple GPUs via DTensor

To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 convnext_example.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)


WORLD_SIZE = 4
ITER_TIME = 20


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format=torch.contiguous_format):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in [torch.contiguous_format]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format=torch.contiguous_format)
        self.pwconv1 = nn.Conv2d(
            dim, 4 * dim, kernel_size=1, stride=1
        )  # nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(
            4 * dim, dim, kernel_size=1, stride=1
        )  # nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * self.drop_path(x)
        x = input_x + x
        return x


class DownSampling(nn.Module):
    def __init__(self, dim_in=3, dim_out=2, down_scale=4, norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        if norm_first:
            self.norm = LayerNorm(dim_in, eps=1e-6, data_format=torch.contiguous_format)
            self.conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=down_scale, stride=down_scale
            )
        else:
            self.conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=down_scale, stride=down_scale
            )
            self.norm = LayerNorm(
                dim_out, eps=1e-6, data_format=torch.contiguous_format
            )

    def forward(self, x):
        if self.norm_first:
            return self.conv(self.norm(x))
        else:
            return self.norm(self.conv(x))


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=10,
        depths=[1, 1],  # noqa: B006
        dims=[2, 4],  # noqa: B006
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = DownSampling(in_chans, dims[0], 4, norm_first=False)
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = DownSampling(dims[i], dims[i + 1], 2, norm_first=True)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(init_weights)

    def forward(self, x):
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x


def _conv_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    for name, param in module.named_parameters():
        dist_spec = [Replicate()]
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        dist_param.register_hook(lambda grad: grad.redistribute(placements=dist_spec))
        name = "_".join(name.split("."))
        module.register_parameter(name, dist_param)


def train_convnext_example():
    device_type = "cuda"
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh(device_type, (world_size,))
    rank = mesh.get_rank()

    in_shape = [7, 3, 512, 1024]
    output_shape = [7, 1000]

    torch.manual_seed(12)
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        drop_path_rate=0.0,
        num_classes=1000,
    ).to(device_type)
    model = distribute_module(model, mesh, _conv_fn, input_fn=None, output_fn=None)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)

    x = torch.randn(*in_shape).to(device_type).requires_grad_()
    y_target = (
        torch.empty(output_shape[0], dtype=torch.long)
        .random_(output_shape[1])
        .to(device_type)
    )
    x = distribute_tensor(x, mesh, [Shard(3)])
    y_target = distribute_tensor(y_target, mesh, [Replicate()])

    # warm up
    y = model(x)
    loss = criterion(y, y_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    forward_time = 0.0
    backward_time = 0.0
    start = time.time()
    for i in range(ITER_TIME):
        t1 = time.time()
        y = model(x)
        torch.cuda.synchronize()
        t2 = time.time()

        loss = criterion(y, y_target)
        optimizer.zero_grad()

        t3 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t4 = time.time()

        optimizer.step()

        forward_time += t2 - t1
        backward_time += t4 - t3
    torch.cuda.synchronize()
    end = time.time()
    max_reserved = torch.cuda.max_memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    print(
        f"rank {rank}, {ITER_TIME} iterations, average latency {(end - start)/ITER_TIME*1000:10.2f} ms"
    )
    print(
        f"rank {rank}, forward {forward_time/ITER_TIME*1000:10.2f} ms, backward {backward_time/ITER_TIME*1000:10.2f} ms"
    )
    print(
        f"rank {rank}, max reserved {max_reserved/1024/1024/1024:8.2f} GiB, max allocated {max_allocated/1024/1024/1024:8.2f} GiB"
    )
    dist.destroy_process_group()


train_convnext_example()
