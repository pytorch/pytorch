import torch
from torch.nn.attention import flex_attention
from torch.utils._debug_mode import DebugMode
from torch import nn
import functools

from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    _WARNINGS_SHOWN,
    and_masks,
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
    flex_attention_hop,
    noop_mask,
    or_masks,
)

"""
@torch.compile()
def f(x, y):
    return x + y

with DebugMode() as debug_mode:
    print(f(torch.randn(2), torch.randn(2)))

print(debug_mode.debug_string())
"""

# NEXT

"""
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

def _get_custom_policy(no_recompute_list=None, must_recompute_list=None):
    def _custom_policy(ctx, func, *args, **kwargs):
        if no_recompute_list is not None and func in no_recompute_list:
            return CheckpointPolicy.MUST_SAVE
        if must_recompute_list is not None and func in must_recompute_list:
            return CheckpointPolicy.MUST_RECOMPUTE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    return _custom_policy

def context_fn_must_recompute_mm():
    must_recompute_list = [
        torch.ops.aten.mm.default,
    ]
    return create_selective_checkpoint_contexts(
        _get_custom_policy(
            must_recompute_list=must_recompute_list,
        ),
    )

@torch.compile(fullgraph=True)
def mm(x, y):
    return torch.matmul(x, y)

def gn(x):
    return torch.sigmoid(mm(x, x))

def fn(x):
    return torch.utils.checkpoint.checkpoint(
        gn,
        x,
        use_reentrant=False,
        context_fn=context_fn_must_recompute_mm,
    )

x = torch.randn(4, 4, requires_grad=True, device='cuda')
with DebugMode() as debug_mode:
    print(fn(x))

print(debug_mode.debug_string())
"""

class FlexAttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # In-projections (query, key, value)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Out-projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project queries, keys, and values
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply flex attention
        attn_output = torch.compile()(flex_attention)(
        #attn_output = flex_attention(
            q,
            k,
            v,
        )

        # Reshape output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        # Out projection
        output = self.out_proj(attn_output)

        return output

from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
)

# TODO: do this
ops_to_save = [torch.ops.aten.mm.default]
context_fn = functools.partial(
    create_selective_checkpoint_contexts, ops_to_save
)

# Define a model that uses FlexAttention with selective activation checkpointing
class SacModule(nn.Module):
    def __init__(self, hidden_size, num_heads, context_fn):
        super().__init__()
        self.flex_attn = FlexAttentionModule(hidden_size, num_heads)
        self.context_fn = context_fn

    def forward(self, x):
        def flex_attn_fn(x):
            return self.flex_attn(x)

        output = checkpoint(
            flex_attn_fn,
            x,
            use_reentrant=False,
            context_fn=self.context_fn,
        )

        return output

device='cuda'
flex_module = SacModule(hidden_size=512, num_heads=8, context_fn=context_fn).to(
    device, dtype=torch.bfloat16
)
x = torch.ones(8, 1024, 512, device=device, dtype=torch.bfloat16, requires_grad=True)

with DebugMode() as debug_mode:
    output_module = flex_module(x)
    grad_output = torch.ones_like(output_module)
    grad_module = torch.autograd.grad(
        outputs=output_module, inputs=x, grad_outputs=grad_output, retain_graph=True
    )[0]

print(debug_mode.debug_string())
