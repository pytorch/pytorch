import sympy
import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from numpy import shape
from torch import nn
from torch._subclasses import fake_tensor, FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.utils._python_dispatch import return_and_correct_aliasing


shape_env = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


def to_fake_tensor(t):
    fake_tensor = fake_mode.from_tensor(
        t.clone(),
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC for _ in range(t.dim())],
        ),
    )

    shape_env.var_to_val.update(
        {
            sympy.sympify(sym): integer
            for sym, integer in zip(fake_tensor.size(), t.size())
        }
    )

    return fake_tensor


class SpecialTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, real_tensor: torch.Tensor, fake_tensor: FakeTensor):
        kwargs = {}
        kwargs["strides"] = fake_tensor.stride()
        kwargs["storage_offset"] = 0
        kwargs["device"] = fake_tensor.device
        kwargs["layout"] = fake_tensor.layout
        kwargs["requires_grad"] = fake_tensor.requires_grad
        kwargs["dtype"] = fake_tensor.dtype

        cls.real_tensor = real_tensor
        cls.fake_tensor = fake_tensor

        out = torch.Tensor._make_wrapper_subclass(cls, cls.fake_tensor.shape, **kwargs)
        return out

    def __init__(
        self,
        real_tensor: torch.Tensor,
        fake_tensor: FakeTensor,
    ):
        self.real_tensor = real_tensor
        self.fake_tensor = fake_tensor

    def __repr__(self):
        return f"SpecialTensor(sym_shape:{self.fake_tensor.shape}, real_shape:{self.real_tensor.shape})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print(
            func, [arg.shape if isinstance(arg, torch.Tensor) else arg for arg in args]
        )
        if kwargs is None:
            kwargs = {}

        # Compute real
        real_args, real_kwargs = pytree.tree_map_only(
            SpecialTensor, lambda x: x.real_tensor, (args, kwargs)
        )

        def eval_args(args, var_to_val):
            args, spec = pytree.tree_flatten(args)
            args = pytree.tree_map_only(
                torch.SymInt,
                lambda x: sympy.sympify(x).subs(var_to_val),
                args,
            )
            return pytree.tree_unflatten(args, spec)

        real_args = eval_args(real_args, shape_env.var_to_val)
        real_out = func(*real_args, **real_kwargs)
        real_out_flat, spec = pytree.tree_flatten(real_out)

        # Compute fake
        fake_args, fake_kwargs = pytree.tree_map_only(
            SpecialTensor, lambda x: x.fake_tensor, (args, kwargs)
        )
        fake_out = func(*fake_args, **fake_kwargs)
        fake_out_flat, spec = pytree.tree_flatten(fake_out)

        # Combine outputs into a new SpecialTensor
        def fn(r, f):
            if isinstance(r, torch.Tensor) and isinstance(f, FakeTensor):
                return SpecialTensor(r, f)
            else:
                return r

        out_flat = pytree.tree_map(fn, real_out_flat, fake_out_flat)
        out = pytree.tree_unflatten(out_flat, spec)

        return return_and_correct_aliasing(func, args, kwargs, out)


class TransformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Define model parameters
        self.n_local_heads = 8
        self.head_dim = 128
        self.dim = 4096
        self.n_head = 32
        self.vocab_size = 128256

        self.dtype = torch.bfloat16

        # Initialize token embeddings and layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim).to(
            dtype=self.dtype
        )
        total_head_dim = (self.n_head + 2 * self.n_local_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, total_head_dim).to(dtype=self.dtype)
        self.wo = nn.Linear(self.dim, self.dim).to(dtype=self.dtype)

    def create_kv_cache(self):
        max_batch_size = 16
        max_seq_length = 1024
        cache_shape = (
            max_batch_size,
            self.n_local_heads,
            max_seq_length,
            self.head_dim,
        )
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=self.dtype, device="cuda")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=self.dtype, device="cuda")
        )

    def reset_kv_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def f_1(self, idx):
        x = self.tok_embeddings(idx)
        kv_size = self.n_local_heads * self.head_dim
        print(x.shape)
        yy = self.wqkv(x)
        q, k, v = yy.split([self.dim, kv_size, kv_size], dim=-1)

        return q, k, v

    def f_2(self, q, k, v, bsz, seqlen):
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        return q, k, v

    def apply_rotary_emb(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * freqs_cis[..., 0]
                - xshaped[..., 1] * freqs_cis[..., 1],
                xshaped[..., 1] * freqs_cis[..., 0]
                + xshaped[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        return x_out2.type_as(x)

    def f_3(self, q, k, v, freqs_cis):
        q = self.apply_rotary_emb(q, freqs_cis)
        k = self.apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        return q, k, v

    def kv_update(self, input_pos, k_val, v_val, k_cache, v_cache):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

    def f_4(self, k, v, input_pos, k_cache, v_cache):
        k, v = self.kv_update(input_pos, k, v, k_cache, v_cache)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        return k, v

    def f_5(self, q, k, v, mask):
        outs = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0
        )
        y = outs[0]
        return (y,)

    def f_6(self, y, bsz, seqlen):
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return (y,)

    def f_7(self, y):
        y = self.wo(y)
        return (y,)

    def f_attention(self, x, freqs_cis, mask, input_pos, k_cache, v_cache):
        bsz, seqlen = x.shape

        q, k, v = self.f_1(x)
        q, k, v = self.f_2(q, k, v, bsz, seqlen)
        q, k, v = self.f_3(q, k, v, freqs_cis)
        k, v = self.f_4(k, v, input_pos, k_cache, v_cache)
        (y,) = self.f_5(q, k, v, mask)
        (y,) = self.f_6(y, bsz, seqlen)
        (y,) = self.f_7(y)

        return (y,)


shape_env = ShapeEnv()

batchsize = 16
seqlen = 1023

x = torch.randint(0, 128256, (batchsize, seqlen))
freqs_cis = torch.randn(seqlen, 64, 2).to(dtype=float)
mask = torch.ones([batchsize, 1, seqlen, 16])
input_pos = torch.arange(0, seqlen, dtype=torch.int32)

model = TransformerModel()
model.to("cuda")

model.create_kv_cache()
inputs = [x, freqs_cis, mask, input_pos, model.k_cache, model.v_cache]
inputs = pytree.tree_map(lambda x: x.to("cuda"), inputs)

inputs = pytree.tree_map(lambda x: SpecialTensor(x, to_fake_tensor(x)), inputs)

outs = model.f_attention(*inputs)
print(outs)


# t1 = torch.randn(5, 6)
# t2 = torch.randn(5, 6)
# a = SpecialTensor(t1, to_fake_tensor(t1))
# b = SpecialTensor(t2, to_fake_tensor(t2))
#
# z = a + b
# print(z.shape)
#
#
# t1 = torch.randn(5, 6)
# t2 = torch.randn(6, 7)
# a = SpecialTensor(t1, to_fake_tensor(t1))
# b = SpecialTensor(t2, to_fake_tensor(t2))
#
# z = torch.ops.aten.mm(a, b)
# print(z.shape)
#
#
# t1 = torch.randn(5, 6)
# a = SpecialTensor(t1, to_fake_tensor(t1))
# ll = torch.nn.Linear(6, 7)
#
# z = ll(a)
# print(z.shape)
#
#
# t1 = torch.randn(5, 6, 7)
# a = SpecialTensor(t1, to_fake_tensor(t1))
#
# z = a.view(a.shape[0] * a.shape[1], a.shape[2])
# z = a.view(5 * 6, 7)
# print(z.shape)
#
#
# t1 = torch.randn(5, 6, 7)
# a = SpecialTensor(t1, to_fake_tensor(t1))
# ll = torch.nn.Linear(7, 7)
#
# z = ll(a)
# print(z.shape)
# print(z.real_tensor.shape)
