# Owner(s): ["oncall: distributed"]

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp._fsdp_param_group import (
    RegisterPostBackwardHook,
)
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp.wrap import wrap


class MLP(nn.Module):
    def __init__(self, dim: int, device: torch.device, dim_multiplier: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        return z


# NOTE: We take the GPT2 implementation from nanoGPT: https://github.com/karpathy/nanoGPT
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        (B, T, C) = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTMLP(nn.Module):  # renamed to avoid name conflict
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPTMLP(config)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    checkpoint_activations: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        wte = nn.Embedding(config.vocab_size, config.n_embd)
        wpe = nn.Embedding(config.block_size, config.n_embd)
        torch.nn.init.normal_(wte.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(wpe.weight, mean=0.0, std=0.02)
        blocks: List[Block] = []
        for _ in range(config.n_layer):
            block = Block(config)
            block = wrap(block)
            blocks.append(block)
        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wpe=wpe,
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(blocks),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            if self.config.checkpoint_activations:
                # We only support composition with non-reentrant AC
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return loss


class StudentTeacher(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.teacher_backbone = nn.Linear(dim, dim)
        self.student_backbone = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim)
        for param_name, param in self.named_parameters():
            if "weight" in param_name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in param_name:
                torch.nn.init.zeros_(param)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        t1 = self.teacher_backbone(x1)
        t2 = self.head(t1).detach()
        s1 = self.student_backbone(x2)
        s2 = self.head(s1)
        loss = (s2 * t2).sum()
        return loss


class DoubleLinear(nn.Module):
    def __init__(self, dim: int, use_second_linear: bool = True):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.use_second_linear = use_second_linear

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_second_linear:
            return self.relu(self.lin1(x)), self.relu(self.lin2(x))
        return self.relu(self.lin1(x))


class ModelWithParamsAndBuffers(nn.Module):
    """
    This model can be used for testing FSDP initialization including
    meta-device initialization and broadcasting module states. We manually
    set the seed in each :meth:`reset_parameters` because FSDP's meta-device
    init cannot ensure that the random seed is consumed in an identical order
    as the non-meta-device init. However, since each :meth:`reset_parameters`
    uses an initialization method that depends on shape, we do exercise that
    a parameter is not initialized multiple times.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = DeterministicLinear(3, 3, device=device)
        self.l2 = DeterministicLinear(3, 3, device=device)
        self.l3 = ModuleWithParamsAndBuffers(device)
        self.l4 = ModuleWithParamsAndBuffers(device)
        self.p = nn.Parameter(torch.ones(3, 3, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(3)
        # Use an initialization method that depends on shape
        torch.nn.init.xavier_uniform_(self.p, 0.5)

    def forward(self, x: torch.Tensor):
        return self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x)))))))


class ModuleWithParamsAndBuffers(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.lin1 = DeterministicLinear(3, 3, device=device)
        self.lin2 = DeterministicLinear(3, 3, device=device)
        self.buf_mod = BufferModule(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.buf_mod(F.relu(self.lin2(F.relu(self.lin1(x)))))


class BufferModule(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.register_buffer("buf", torch.empty((3, 3), device=device))
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(3)
        # Use an initialization method that depends on shape
        torch.nn.init.xavier_uniform_(self.buf, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.buf


class DeterministicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device: torch.device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim, device=device))
        self.reset_parameters()

    def reset_parameters(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]):
        torch.manual_seed(3)
        # Use an initialization method that depends on shape
        torch.nn.init.xavier_uniform_(self.weight, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


@contextlib.contextmanager
def patch_all_gather(new_all_gather_into_tensor: Callable):
    orig_all_gather = dist.all_gather_into_tensor
    dist.all_gather_into_tensor = new_all_gather_into_tensor
    try:
        yield
    finally:
        dist.all_gather_into_tensor = orig_all_gather


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter_tensor: Callable):
    orig_reduce_scatter = dist.reduce_scatter_tensor
    dist.reduce_scatter_tensor = new_reduce_scatter_tensor
    try:
        yield
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter


@contextlib.contextmanager
def patch_register_post_backward_hook_backward(new_backward: Callable):
    orig_backward = RegisterPostBackwardHook.backward
    RegisterPostBackwardHook.backward = new_backward
    try:
        yield
    finally:
        RegisterPostBackwardHook.backward = orig_backward


def reduce_scatter_with_numel_assert(
    cls,
    orig_reduce_scatter: Callable,
    numel: int,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
):
    if len(args) > 0:
        output = args[0]
    elif "output" in kwargs:
        output = kwargs["output"]
    else:
        raise AssertionError(
            f"Cannot get reduce-scatter output from\nargs: {args}\nkwargs: {kwargs}"
        )
    cls.assertEqual(output.numel(), numel)
    return orig_reduce_scatter(*args, **kwargs)


def reduce_scatter_with_dtype_assert(
    cls,
    orig_reduce_scatter: Callable,
    dtype: torch.dtype,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
):
    if len(args) > 0:
        output = args[0]
    elif "output" in kwargs:
        output = kwargs["output"]
    else:
        raise AssertionError(
            f"Cannot get reduce-scatter output from\nargs: {args}\nkwargs: {kwargs}"
        )
    cls.assertEqual(output.dtype, dtype)
    return orig_reduce_scatter(*args, **kwargs)


def get_active_memory_bytes() -> int:
    return torch.cuda.memory_stats()["active_bytes.all.peak"]


def reduce_scatter_grad(
    tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None
) -> Optional[torch.Tensor]:
    """
    Returns the reduce-scattered gradient, including an averaging by the world
    size to emulate data parallel semantics. If the gradient is ``None``, then
    this simply returns ``None``.
    """
    if tensor.grad is None:
        return None
    group = group or dist.distributed_c10d._get_default_group()
    if tensor.grad.ndim == 0:
        raise NotImplementedError()
    world_size = group.size()
    rank = group.rank()
    chunks = torch.chunk(tensor.grad, world_size, dim=0)
    dim_0_numel_to_pad = world_size * chunks[0].size(0) - tensor.grad.size(0)
    padded_grad = (
        F.pad(tensor.grad, [0, dim_0_numel_to_pad], dim=0)
        if dim_0_numel_to_pad > 0
        else tensor.grad
    )
    padded_sharded_grad = torch.zeros_like(chunks[0])
    dist.reduce_scatter_tensor(padded_sharded_grad, padded_grad, group=group)
    padded_sharded_grad.div_(world_size)
    sharded_grad = padded_sharded_grad[: chunks[rank].size(0)]
    return sharded_grad


def all_reduce_grad(
    tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None
) -> Optional[torch.Tensor]:
    """
    Returns the all-reduced gradient, including an averaging by the world size
    to emulate data parallel semantics. If the gradient is ``None``, then this
    just simply returns ``None``.
    """
    if tensor.grad is None:
        return None
    group = group or dist.distributed_c10d._get_default_group()
    dist.all_reduce(tensor.grad, group=group)
    tensor.grad.div_(group.size())
    return tensor.grad


def check_train_parity(
    cls,  # unit test class
    local_batch_size: int,
    inp_dims: Tuple[int, ...],
    ref_model: nn.Module,  # non-distributed
    ref_optim: torch.optim.Optimizer,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    group: Optional[dist.ProcessGroup] = None,
    is_sharded: bool = True,
):
    group = group or dist.distributed_c10d._get_default_group()
    rank = group.rank()
    world_size = group.size()
    torch.manual_seed(1)  # same on all ranks
    global_batch_size = local_batch_size * world_size
    for iter_idx in range(6):
        global_inp = torch.rand((global_batch_size, *inp_dims), device="cuda")
        local_inp = global_inp[
            rank * local_batch_size : (rank + 1) * local_batch_size
        ].detach()
        losses: List[torch.Tensor] = []
        for _model, inp in ((ref_model, global_inp), (model, local_inp)):
            losses.append(_model(inp).sum())
            losses[-1].backward()
            if _model is ref_model:
                for param in ref_model.parameters():
                    if param.grad is not None:
                        param.grad.div_(world_size)
        dist.all_reduce(losses[1])  # partial -> replicated
        cls.assertEqual(losses[0], losses[1])
        if is_sharded:
            check_sharded_grad_parity(cls, ref_model, model)
        else:
            raise NotImplementedError()
        for _optim in (optim, ref_optim):
            _optim.step()
            _optim.zero_grad(set_to_none=(iter_idx % 2))


def check_sharded_param_parity(
    cls,  # unit test class
    unsharded_model: nn.Module,
    sharded_model: nn.Module,
    group: Optional[dist.ProcessGroup] = None,
):
    group = group or dist.distributed_c10d._get_default_group()
    rank = group.rank()
    world_size = group.size()
    for (param_name, param), (ref_name, ref_param) in zip(
        sharded_model.named_parameters(), unsharded_model.named_parameters()
    ):
        cls.assertEqual(param_name, ref_name)
        chunks = torch.chunk(ref_param, world_size, dim=0)
        if rank >= len(chunks):
            continue  # padding-only parameter
        else:
            ref_sharded_param = chunks[rank]
        param = param._local_tensor if isinstance(param, DTensor) else param
        cls.assertEqual(param, ref_sharded_param)


def check_sharded_grad_parity(
    cls,  # unit test class
    unsharded_model: nn.Module,
    sharded_model: nn.Module,
    group: Optional[dist.ProcessGroup] = None,
):
    group = group or dist.distributed_c10d._get_default_group()
    rank = group.rank()
    world_size = group.size()
    for (param_name, param), (ref_name, ref_param) in zip(
        sharded_model.named_parameters(), unsharded_model.named_parameters()
    ):
        cls.assertEqual(param_name, ref_name)
        if param.grad is None:
            cls.assertEqual(ref_param.grad, None)
            continue
        chunks = torch.chunk(ref_param.grad, world_size, dim=0)
        if rank >= len(chunks):
            ref_sharded_grad = None  # padding-only gradient
        else:
            ref_sharded_grad = chunks[rank]
        grad = (
            param.grad._local_tensor if isinstance(param.grad, DTensor) else param.grad
        )
        cls.assertEqual(grad, ref_sharded_grad)
