# Owner(s): ["module: functorch"]
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import gc
from unittest import skip, skipIf

from attn_ft import BertSelfAttention as BertSelfAttentionA, Linear
from attn_positional import BertSelfAttention as BertSelfAttentionB

import functorch.dim
import torch
from functorch.dim import Dim, DimList, dimlists, dims, stack, Tensor
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TestCase,
)


try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None

from contextlib import contextmanager
from time import perf_counter


measure_perf = False
if measure_perf:
    from torchdim.magic_trace import magic_trace
else:

    @contextmanager
    def magic_trace(*args, **kwargs):
        yield


@contextmanager
def measure(what):
    b = perf_counter()
    yield
    e = perf_counter()
    print(f"{what}: {e - b:.20f} seconds")


def triu(A):
    i, j = dims()
    a = A[i, j]
    zero = torch.tensor(0, dtype=torch.float)  # XXX - torch.where is janky...
    return torch.where(i <= j, a, zero).order(i, j)


def gpu_time(lmb, name, r=100):
    b = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    # with magic_trace(name + ".fxt"):
    for _ in range(r):
        lmb()
    b.record()
    for _ in range(r):
        lmb()
    e.record()
    e.synchronize()
    elapsed = b.elapsed_time(e)
    # with torch.profiler.profile(schedule=torch.profiler.schedule(
    #     wait=0,
    #     warmup=1,
    #     active=2), on_trace_ready=tensorboard_trace_handler(name), with_stack=True) as profiler:
    #     for _ in range(3):
    #         lmb()
    #         profiler.step()
    print(name, elapsed / r)
    return elapsed / r


@skipIfTorchDynamo("Bad interaction")
class TestMin(TestCase):
    def setUp(self):
        super().setUp()
        gc.disable()
        gc.collect()
        self.interesting = set()
        for o in gc.get_objects():
            if isinstance(o, (torch.Tensor, Dim, Tensor, DimList)):
                self.interesting.add(id(o))
        if "cuda" in self._testMethodName:
            self.mem_allocated = torch.cuda.memory_allocated()

    def tearDown(self):
        interesting = []
        for o in gc.get_objects():
            if (
                isinstance(o, (torch.Tensor, Dim, Tensor, DimList))
                and id(o) not in self.interesting
            ):
                interesting.append(o)

        extra_memory = 0
        if "cuda" in self._testMethodName:
            extra_memory += torch.cuda.memory_allocated() - self.mem_allocated

        #  nolevels = _n_levels_in_use() == 0
        if extra_memory != 0 or len(interesting) != 0:
            import refcycle

            refcycle.garbage().export_image("garbage.pdf")
        gc.collect()
        # assert nolevels, f"cleanup failed? {_n_levels_in_use()}"
        self.assertEqual(
            extra_memory, 0, f"extra cuda memory left allocated: {extra_memory}"
        )
        self.assertEqual(
            len(interesting),
            0,
            (
                f"extra torch.Tensor, Dim, or Tensor left allocated: {len(interesting)} objects of types:"
                f"{[type(t) for t in interesting]}"
            ),
        )

    def test_manual_stuff(self):
        A_ = torch.rand(3, 4)
        B_ = torch.rand(4, 5)
        i, j, k = dims()
        A = A_[i, k]
        B = B_[k, j]
        C = (A.expand(j) * B.expand(i)).sum(k)
        torch.testing.assert_close(C.order(i, j), torch.mm(A_, B_))
        torch.testing.assert_close(torch.triu(A_, 0), triu(A_))

        D_ = torch.randint(0, 3, (6,))
        d = dims()
        D = D_[d]

        A.index([i], [D]).order(k, d)

    def attn(
        self,
        batch_size=1,
        sequence_length=4,
        hidden_size=6,
        num_attention_heads=3,
        linear=Linear,
        device=None,
        time=False,
    ):
        def maybe_to(x):
            return x if device is None else x.to(device)

        attention_probs_dropout_prob = 0.0
        A = maybe_to(
            BertSelfAttentionA(
                hidden_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                linear=linear,
            )
        )
        B = maybe_to(
            BertSelfAttentionB(
                hidden_size, num_attention_heads, attention_probs_dropout_prob
            )
        )

        A.load_state_dict(B.state_dict())
        hidden_state = maybe_to(torch.rand(batch_size, sequence_length, hidden_size))
        b_out = B(hidden_state)
        a_out = A(hidden_state)
        torch.testing.assert_close(
            a_out, b_out
        )  # why does a simple matmul not do the right thing?

        if time:
            gpu_time(lambda: B(hidden_state), "positional", r=3)
            gpu_time(lambda: A(hidden_state), "first_class", r=3)

        for approach in ("relative_key", "relative_key_query"):
            A = maybe_to(
                BertSelfAttentionA(
                    hidden_size,
                    num_attention_heads,
                    attention_probs_dropout_prob,
                    approach,
                    sequence_length,
                    linear=linear,
                )
            )
            B = maybe_to(
                BertSelfAttentionB(
                    hidden_size,
                    num_attention_heads,
                    attention_probs_dropout_prob,
                    approach,
                    sequence_length,
                )
            )
            A.load_state_dict(B.state_dict())

            hidden_state = maybe_to(
                torch.rand(batch_size, sequence_length, hidden_size)
            )
            b_out = B(hidden_state)
            a_out = A(hidden_state)
            torch.testing.assert_close(a_out, b_out)

            if time:
                gpu_time(lambda: B(hidden_state), "positional", r=3)
                gpu_time(lambda: A(hidden_state), "first_class", r=3)

        A = maybe_to(
            BertSelfAttentionA(
                hidden_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                None,
                None,
                linear=linear,
            )
        )
        B = maybe_to(
            BertSelfAttentionB(
                hidden_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                None,
                None,
            )
        )
        A.load_state_dict(B.state_dict())

        hidden_state = maybe_to(torch.rand(batch_size, sequence_length, hidden_size))
        past_key_value = (
            maybe_to(
                torch.rand(
                    batch_size,
                    num_attention_heads,
                    sequence_length,
                    hidden_size // num_attention_heads,
                )
            ),
            maybe_to(
                torch.rand(
                    batch_size,
                    num_attention_heads,
                    sequence_length,
                    hidden_size // num_attention_heads,
                )
            ),
        )

        b_out = B(hidden_state, past_key_value=past_key_value)
        a_out = A(hidden_state, past_key_value=past_key_value)
        torch.testing.assert_close(a_out, b_out)

        if time:
            gpu_time(lambda: B(hidden_state), "positional", r=3)
            gpu_time(lambda: A(hidden_state), "first_class", r=3)

    def test_attn(self):
        self.attn()

    def test_inplace(self):
        # some embeddings table
        embeddings = torch.zeros(10, 3)

        # some sparse updates to the embeddings
        indices = torch.arange(2) + 1
        values = torch.rand(2, 3)

        i, n, f = dims()

        embeddings[indices[i], f] += values[i, f]

    def test_adapt(self):
        def f():
            ci, co = dims()

        # python 3.11 adapts bytecode after a number of iterations
        # check that we still match names correctly
        for _ in range(10):
            f()

    @skipIf(not TEST_CUDA, "no CUDA")
    def test_attn_cuda(self):
        # size from the BERT paper, 90% pretraining of sequence length 128
        self.attn(
            batch_size=256,
            hidden_size=768,
            sequence_length=128,
            num_attention_heads=12,
            device="cuda",
            time=measure_perf,
            linear=torch.nn.Linear,
        )

    def test_stack(self):
        i, j, d = dims()
        A = torch.rand(4, 5)
        _r = stack([A[i, j]], d, j)
        # a, b = r.unbind(d)
        # self.assertTrue(torch.allclose(a.order(i, j), i.expand(j).order(i, j)))
        # self.assertTrue(torch.allclose(b.order(i, j), j.expand(i).order(i, j)))

    def test_max(self):
        ap = torch.rand(2, 3, 2)
        i, j, k = dims()
        a = ap[i, j, k]
        r, i0 = a.max(dim=k)
        torch.testing.assert_close(r.order(i, j), ap.max(2)[0])

    def test_mm(self):
        i, j, k, q = dims()
        a = torch.rand(3, 4)
        b = torch.rand(4, 5)
        a_ = a[i, k]
        b_ = b[k, j]
        q.size = 1
        _r = (a_.expand(j, q) * b_.expand(i, q)).sum(k).order(q, i, j)
        # r = (a_*b_).sum(k).order(q, i, j)
        # print(r)
        # print(a @ b)

    def test_with_dims_split(self):
        a = torch.arange(3 * 12).view(3, 12)
        i, j, k = dims()
        k.size = 4
        r = a[i, [j, k]]
        x = r.order(i, [j, k])
        torch.testing.assert_close(a, x)

    def test_hello(self):
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        i, j, k = dims()

        # r = A[i]*4
        r = (A[i, k] * B[k, j]).sum(k).order(i, j)
        torch.testing.assert_close(r, A @ B)

        self.assertEqual(A.sum(), A[i].sum((0, i)))
        self.assertEqual(A.sum(), A[i].sum((-1, i)))

        torch.testing.assert_close(A.sum(), A[i].sum(0, keepdim=True).sum((0, i)))
        torch.testing.assert_close(A[i].std(i, True), A.std(0, True))

        torch.testing.assert_close(A[i, k].max(i)[0].order(k), A.max(0)[0])
        torch.testing.assert_close(A.sort(1)[0], A[i, k].sort(k)[0].order(i, k))
        # XXX - chunk changes the size of a dimension, has to take a new dimension...
        # assert torch.allclose(A.chunk(2,1)[0], A[i, k].chunk(2, k)[0].order(i, k))
        torch.testing.assert_close(A[i].renorm(1, i, 7).order(i), A.renorm(1, 0, 7))
        torch.testing.assert_close(
            A.expand(5, -1, -1), A[i, k].expand(j).order(j, i, k)
        )

        z = dims()
        C = torch.arange(2)
        torch.testing.assert_close(A[:, 0:2], A[i, k].index(k, C[z]).order(i, z))

        o, l = dims()
        o.size = 2
        r = A[i, k].index(k, (o, l))
        torch.testing.assert_close(r.order(i, o, l), A.view(-1, 2, 2))
        rr = r.index((o, l), k)
        torch.testing.assert_close(A, rr.order(i, k))

        r = i + k - 1
        r2 = torch.arange(3)[:, None] + torch.arange(4)[None, :] - 1
        torch.testing.assert_close(r.order(i, k), r2)

        # test with ...
        torch.testing.assert_close(A.T, A[..., k].order(k))

        # test with dimlist
        a_, b_ = dimlists()
        torch.testing.assert_close(A[i, a_].order(*a_, i), A.T)
        # test with one bound dimlist
        torch.testing.assert_close(A[:, a_].order(*a_), A.T)
        # test with a dimlist that will end up empty
        torch.testing.assert_close(A[i, b_, k].order(i, k, *b_), A)
        # test with too few things
        (A[i] + i)
        torch.testing.assert_close((A[i] + i).order(i), A + torch.arange(3)[:, None])
        # test with too many elements
        try:
            A[1, ..., 1, 1]
            raise NotImplementedError
        except IndexError:
            pass
        c, d = dims()
        c.size = 2
        torch.testing.assert_close(A[i, [c, d]].order(i, c, d), A.view(3, 2, 2))

        torch.testing.assert_close(
            A[c + 1, c + 0].order(c), A[torch.arange(2) + 1, torch.arange(2)]
        )

        C = torch.rand(4, 7)
        c_, x, y, z = dims()

        a, b, c = C.split((3, 3, 1), dim=1)
        s = dims()
        ref = C.split((3, 3, 1), dim=1)
        t = C[s, c_].split((x, y, z), dim=c_)
        for a, b, d in zip(ref, t, (x, y, z)):
            torch.testing.assert_close(a, b.order(s, d))

        D = torch.rand(3, 4, 5)
        torch.testing.assert_close(
            D.transpose(0, 1).flatten(1, 2), D[i, k, j].order((i, j)).order(k)
        )

        r = [id(x) for x in torch.rand_like(A[i, k]).dims]
        if not (id(i) in r and id(k) in r):
            raise AssertionError("Expected i and k to be in dims")
        r = [id(x) for x in torch.nn.functional.dropout(A[i, k]).dims]
        if not (id(i) in r and id(k) in r):
            raise AssertionError("Expected i and k to be in dims")

    def test_simple(self):
        i, j, k = dims()
        x = torch.rand(3, 4)
        z = x[i, j]
        (z + z + z + z)
        (z.order(i, j))

    def test_mm_fuse(self):
        i, j, k = dims()
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)

        C = (A[i, k] * B[k, j]).sum(k).order(i, j)
        torch.testing.assert_close(C, A @ B)

    def test_time_mm_fuse(self):
        i, j, k = dims()
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)

        for _ in range(10):
            r0 = A @ B

        for _ in range(10):
            a = A[i, k]
            b = B[k, j]
            r1 = (a * b).sum(k)

        with measure("pp"):
            for _ in range(10000):
                A @ B
        # magic_trace_stop_indicator()

        with measure("fc"):
            for _ in range(10000):
                (A[i, k] * B[k, j]).sum(k).order(i, j)

        with magic_trace("f.fxt"):
            for _ in range(10000):
                (A[i, k] * B[k, j]).sum(k).order(i, j)

        with magic_trace("p.fxt"):
            for _ in range(10000):
                A @ B

        # magic_trace_stop_indicator()

        torch.testing.assert_close(r1.order(i, j), r0)

    def test_compare_dims(self):
        i, j = dims()
        i.size = 3
        j.size = 4
        (i < j)  # noqa: B015

    def test_seg(self):
        i, k = dims()
        i.size = 4
        k.size = 3
        i + k - 1

    def test_expand(self):
        A = torch.rand(3, 4)
        i = dims()
        self.assertEqual(list(A[i].expand(2, 4).order(i).size()), [3, 2, 4])

    def test_network(self):
        if resnet18 is None:
            self.skipTest("no torchvision")
        rn = resnet18(
            norm_layer=lambda x: torch.nn.BatchNorm2d(x, track_running_stats=False)
        )
        rn.train()
        img = torch.rand(1, 1, 2, 3, 224, 224)
        imgf = img.view(2, 3, 224, 224)

        i, j = dims()
        r = rn(img[i, j])
        r = r.order(i, j).view(2, 1000)
        r2 = rn(imgf)
        torch.testing.assert_close(r2, r, atol=1e-6, rtol=1e-7)

    def test_dim_args(self):
        a = dimlists()
        if not isinstance(a, DimList):
            raise AssertionError(f"Expected DimList, got {type(a)}")
        a = dims()
        b = dimlists()
        if not isinstance(a, Dim):
            raise AssertionError(f"Expected Dim, got {type(a)}")
        if not isinstance(b, DimList):
            raise AssertionError(f"Expected DimList, got {type(b)}")
        self.assertEqual(str(a), "a")
        a, b = dims(sizes=[3, 4])
        self.assertEqual(a.size, 3)
        self.assertEqual(b.size, 4)
        a = dims(sizes=[3])
        b = dimlists(sizes=[4])
        self.assertEqual(len(b), 4)
        a = dims()
        b = dimlists(sizes=[[4, 5]])
        self.assertEqual(b[0].size, 4)
        self.assertEqual(b[1].size, 5)

    def test_diag(self):
        i = dims()
        A = torch.rand(4, 4)
        (A[i, i])

    def test_softmax_split(self):
        a = torch.rand(16)
        g, i = dims(sizes=[2, None])
        a2 = a[[i, g],]

        m_b, _ = a2.max(i)
        f_b = torch.exp(a2 - m_b)
        l_b = f_b.sum(i)

        m, _ = m_b.max(g)
        c = torch.exp(m_b - m)
        f = (c * f_b).order((i, g))
        l = (c * l_b).sum(g)
        torch.testing.assert_close(f / l, torch.nn.functional.softmax(a, dim=0))

    def test_index(self):
        A = torch.rand(3, 4)
        i, j, k = dims()

        o, l = dims()
        o.size = 2
        r = A[i, k].index(k, [o, l])
        torch.testing.assert_close(r.order(i, o, l), A.view(-1, 2, 2))
        rr = r.index([o, l], k)
        torch.testing.assert_close(A, rr.order(i, k))
        z = dims()
        C = torch.arange(2)
        x = A[i, k].index(k, C[z]).order(i, z)
        torch.testing.assert_close(A[:, 0:2], x)

        C = torch.rand(3, 4, 5)
        ik = dims()
        torch.testing.assert_close(
            C.index((0, 2), ik).order(ik), C.permute(0, 2, 1).reshape(15, 4)
        )

    # failures that came up from monkey patching some operators...
    def test_monkey(self):
        A = torch.rand(3, 4)
        A[0, 0] = 5
        x = torch.randn(3, 4, 4, 4, 3)
        x_clone1 = x.clone()
        ia = torch.tensor([0, 2, 1])
        ib = torch.tensor([0, 2, 1])
        first_shape = x[:, ia, None, ib, 0].shape
        x_clone1[:, ia, None, ib, 0] = torch.randn(first_shape).to(x_clone1)
        x = torch.autograd.Variable(torch.tensor([]))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        a = [z[2], z[0] + 3]
        x.new(a)
        # self.assertEqual(x.new([z[2], z[0] + 3]).tolist(), [3, 4])

    def test_index_placement(self):
        A = torch.rand(1, 2, 3, 4)

        i, j = dims(sizes=[2, 4])

        a = A[:, i + 0, :, j + 0]
        r = a.order(i, j)

        torch.testing.assert_close(A.permute(1, 3, 0, 2), r)

    def test_order(self):
        i, j = dims()
        A = torch.rand(3, 4, 5)
        torch.testing.assert_close(A[i].order(1, i), A.permute(2, 0, 1))

    def test_mask(self):
        a = torch.rand(5)
        i, j = dims(sizes=[a.size(0), a.size(0)])
        ((i >= j) * a[i]).sum(j).order(i)

    def test_eq(self):
        i, j = dims(sizes=[3, 3])
        self.assertEqual((i == j).sum((i, j)), 3)

    def test_dims_with_size(self):
        x = dims(3)
        self.assertEqual(len(x), 3)
        if not isinstance(x[0], Dim):
            raise AssertionError(f"Expected Dim, got {type(x[0])}")

        class Foo:
            pass

        y = Foo()
        z, y.x, q = dims(3)
        self.assertEqual(str(z), "z")
        self.assertEqual(str(y.x), "d1")
        self.assertEqual(str(q), "d2")

    def test_dir(self):
        i, j = dims(sizes=[3, 3])
        dir(i <= j)

    def test_doc(self):
        self.assertEqual(Tensor.clamp.__doc__, torch.Tensor.clamp.__doc__)

    def test_embed(self):
        embeddings = torch.rand(8, 32)
        ids = torch.tensor([1, 0, 3, 4])

        # slow but Pythonic
        values_ = torch.empty(4, 32)
        for batch in range(ids.size(0)):
            for feature in range(embeddings.size(1)):
                values_[batch, feature] = embeddings[ids[batch], feature]

        # with torchdim, single indexing kernel
        batch, feature = dims(2)
        values = embeddings[ids[batch], feature].order(batch, feature)

        torch.testing.assert_close(values, values_)

    def test_functorch(self):
        A = torch.rand(3, 4, 5)
        B = torch.rand(3, 4, 5)
        C = torch.rand(5, 2)

        i, j = dims()

        AA = torch.mm(A[i], C)  # 3, 4, 2
        BB = torch.mm(B[j], C)  # 3, 4, 2
        self.assertEqual(list(torch.mm(AA.T, BB).order(i, j).shape), [3, 3, 2, 2])

    def test_permute_orig(self):
        d = dims(1)
        t_fc = torch.rand(1, 2, 3, 4)[d]
        self.assertEqual(
            t_fc.permute(dims=(1, 0, 2)).shape, t_fc.permute(1, 0, 2).shape
        )

    def test_order_keyword(self):
        d = dims(1)
        t = torch.rand(3)[d]
        self.assertRaises(TypeError, lambda: t.order(wrong=3))

    def test_big_split(self):
        total = 0
        l = []
        while total < 6400:
            l.append(torch.randint(2, 10, (1,)).item())
            total += l[-1]
        x = torch.randn(total, 1)
        x.split(l, 0)


skip_functorch_only = ["test_time_mm_fuse", "test_attn_cuda"]


class TestMinFunctorchOnly(TestMin):
    def setUp(self):
        super().setUp()
        functorch.dim.POINTWISE_OPTIMIZE = False

    def tearDown(self):
        functorch.dim.POINTWISE_OPTIMIZE = True
        super().tearDown()


for n in skip_functorch_only:
    setattr(TestMinFunctorchOnly, n, skip("skip_functorch_only")(lambda self: None))

if __name__ == "__main__":
    run_tests()
