# Owner(s): ["module: nn"]
import itertools
import random
import unittest
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIf,
    skipMeta,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    _assertGradAndGradgradChecks,
    dtype2prec_DONTUSE,
    instantiate_parametrized_tests,
    IS_JETSON,
    parametrize as parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
)


class TestEmbeddingNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_embedding_max_norm_unsorted_repeating_indices(self):
        def create_embedding(device):
            # Seed RNG so we get the same Embedding each time
            torch.manual_seed(0)
            return torch.nn.Embedding(
                num_embeddings=20, embedding_dim=64, max_norm=1.0
            ).to(device)

        ix = torch.arange(2, device="cpu", dtype=torch.long).repeat(2000)
        out_cpu = create_embedding("cpu")(ix)

        ix = ix.to("cuda")
        out = create_embedding("cuda")(ix)
        self.assertEqual(out.cpu(), out_cpu)

    def test_embedding_sparse_basic(self):
        embedding = nn.Embedding(10, 20, sparse=True)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long)
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_embedding_sparse_empty_tensor(self):
        embedding = nn.Embedding(0, 0, sparse=True)
        input = torch.tensor([], dtype=torch.int64)
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

        embedding = nn.Embedding(10, 0, sparse=True)
        input = torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]])
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_move_sparse_half_embedding(self):
        embedding = nn.Embedding(10, 3, sparse=True)
        self.assertEqual(embedding.weight.device.type, "cpu")
        self.assertEqual(embedding.weight.dtype, torch.get_default_dtype())
        embedding.to(torch.float16)
        self.assertEqual(embedding.weight.dtype, torch.float16)
        self.assertEqual(embedding.embedding_dim, 3)
        self.assertEqual(embedding.num_embeddings, 10)

        if torch.cuda.is_available():
            embedding.to("cuda")
            self.assertEqual(embedding.weight.device.type, "cuda")
            embedding.to("cpu")
            self.assertEqual(embedding.weight.device.type, "cpu")

    def test_embedding_max_norm(self):
        embedding = nn.Embedding(22, 5, max_norm=1.0)
        input = torch.tensor([2, 8, 8, 6], dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    @parametrize_test(
        "dtype",
        (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.double,
        ),
    )
    def test_embedding_from_pretrained(self, dtype):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        embedding = nn.Embedding.from_pretrained(a)
        self.assertEqual(a, embedding.weight.data)

        input = torch.LongTensor([0, 1])
        output = embedding(input)
        self.assertEqual(a, output)

    def test_embedding_bag_from_pretrained(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        embedding = nn.EmbeddingBag.from_pretrained(a)
        self.assertEqual(a, embedding.weight)

        input = torch.tensor([0, 1], dtype=torch.long)
        output = embedding(input, torch.arange(input.size(0)))
        self.assertEqual(a, output)

    def test_embedding_from_pretrained_padding_idx(self):
        padding_idx = 2
        padding_vec = torch.ones(3) * 7
        embeddings = torch.rand(4, 3, requires_grad=True)
        with torch.no_grad():
            embeddings[padding_idx] = padding_vec
        embedding_nn = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.assertEqual(embedding_nn.weight[padding_idx], padding_vec)

    def test_embedding_bag_from_pretrained_padding_idx(self):
        padding_idx = 2
        embeddings = torch.rand(4, 3, requires_grad=True)
        embedding_nn = nn.EmbeddingBag.from_pretrained(
            embeddings, padding_idx=padding_idx
        )
        self.assertEqual(embedding_nn.weight, embeddings)

    def test_embedding_from_pretrained_options(self):
        with set_default_dtype(torch.double):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            opts = {
                "max_norm": 2.0,
                "norm_type": 0.5,
                "scale_grad_by_freq": False,
                "sparse": True,
            }
            embedding = nn.Embedding.from_pretrained(a, **opts)
            input = torch.LongTensor([0, 1])
            output = embedding(input)
            # test output and that weight matrix was renormalized
            self.assertEqual(a, output)
            self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
            self.assertTrue(
                output.data.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all()
            )

    def test_embedding_functional(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        embeddings = torch.rand(4, 3, requires_grad=True)

        embed_old = torch.nn.Embedding(4, 3)
        embed_old.weight.data = embeddings.data
        # A silly test for eager, this test is useful for when we run under PYTORCH_TEST_WITH_DYNAMO=1
        # as it ensures that setattr correctly works.
        self.assertEqual(embed_old.weight.data, embeddings.data)
        res_old = embed_old(a)

        res_F = F.embedding(a, embeddings)
        self.assertEqual(res_old, res_F)

        embed_old = torch.nn.Embedding(4, 3)
        embed_old = embed_old.from_pretrained(embeddings, padding_idx=2)
        res_old = embed_old(a)
        res_F = F.embedding(a, embeddings, padding_idx=2)

        self.assertEqual(res_old, res_F)

    # https://github.com/pytorch/pytorch/issues/130806
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @largeTensorTest("40GB", device="cuda")
    def test_large_tensors(self):
        input = torch.randint(low=0, high=16032, size=[131072], device="cuda")
        w = torch.randn([16032, 16384], device="cuda")
        out = torch.nn.functional.embedding(input, w)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(out.numel(), 2147483648)

    def test_embedding_bag_functional(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        embeddings = torch.rand(4, 3, requires_grad=True)

        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old.weight = torch.nn.Parameter(embeddings)
        res_old = embed_old(a)

        res_F = F.embedding_bag(a, embeddings)
        self.assertEqual(res_old, res_F)

        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old = embed_old.from_pretrained(embeddings, padding_idx=2)
        res_old = embed_old(a)
        res_F = F.embedding_bag(a, embeddings, padding_idx=2)

        self.assertEqual(res_old, res_F)

    # Make sure that error is thrown if padding_idx is out of bounds
    def test_embedding_bag_padding_idx_error(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        num_embeddings = 4
        num_features = 3
        embeddings = torch.rand(num_embeddings, num_features, requires_grad=True)

        functional_err_msg = r"padding_idx must be within the number of embeddings"
        module_err_msg = r"padding_idx must be within num_embeddings"

        for padding_idx in range(-(num_embeddings + 2), (num_embeddings + 2)):
            if (padding_idx < -num_embeddings) or (padding_idx >= num_embeddings):
                with self.assertRaisesRegex(RuntimeError, functional_err_msg):
                    F.embedding_bag(a, embeddings, padding_idx=padding_idx)
                with self.assertRaisesRegex(AssertionError, module_err_msg):
                    torch.nn.EmbeddingBag(
                        num_embeddings, num_features, padding_idx=padding_idx
                    )
            else:
                F.embedding_bag(a, embeddings, padding_idx=padding_idx)
                torch.nn.EmbeddingBag(
                    num_embeddings, num_features, padding_idx=padding_idx
                )

    def test_embeddingbag_from_pretrained(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        embeddingbag = nn.EmbeddingBag.from_pretrained(a)
        self.assertEqual(a, embeddingbag.weight.data)

        input = torch.LongTensor([[0, 1]])
        output = embeddingbag(input)
        self.assertEqual(a.mean(0, keepdim=True), output)

    def test_embeddingbag_from_pretrained_options(self):
        with set_default_dtype(torch.double):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            opts = {
                "max_norm": 2.0,
                "norm_type": 0.5,
                "scale_grad_by_freq": False,
                "mode": "max",
                "sparse": False,
            }
            embeddingbag = nn.EmbeddingBag.from_pretrained(a, **opts)

            input = torch.LongTensor([[0, 1]])
            output = embeddingbag(input)
            self.assertEqual(a.max(0, keepdim=True)[0], output)
            self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
            self.assertTrue(
                a.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all()
            )

    def test_embeddingbag_include_last_offset(self):
        # Test case from https://github.com/pytorch/pytorch/issues/89677
        embeddingbag = nn.EmbeddingBag(100, 3, include_last_offset=True, padding_idx=61)
        input = torch.tensor([0, 1, 2, 3])
        out = embeddingbag(input, torch.tensor([0, 3, 3]))
        out2 = embeddingbag(input, torch.tensor([0, 3, 4]))

        weight = embeddingbag.weight
        row0 = weight[0:3].mean(0)
        row1 = weight[3]
        ref_out = torch.stack([row0, row1])

        self.assertEqual(ref_out, out)
        self.assertEqual(ref_out, out2)


class TestEmbeddingNNDeviceType(NNTestCase):
    def test_embedding_dense_grad(self, device):
        with set_default_dtype(torch.double):
            embd = nn.Embedding(20, 20).to(device)
            weight = embd.weight

            def fn_wrapper(device):
                def fn(weight):
                    inp = torch.tensor(
                        [[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long
                    ).to(device)
                    return torch.nn.functional.embedding(inp, weight)

                return fn

            fn = fn_wrapper(device)
            _assertGradAndGradgradChecks(self, fn, (weight,))

    def test_embedding_scalar_weight_error(self, device):
        indices = torch.rand(2, 2, device=device).long()
        weights = [
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device).reshape(1, 1, 1),
        ]

        for weight in weights:
            with self.assertRaisesRegex(RuntimeError, "'weight' must be 2-D"):
                torch.nn.functional.embedding(indices, weight)

    @dtypesIfCUDA(torch.float16, torch.float64)
    @dtypes(torch.float64)
    def test_embedding_backward(self, device, dtype):
        embedding = nn.Embedding(10, 3, sparse=True)
        tensor = torch.tensor([[7, 1, 3]])
        ones = torch.tensor(1.0, dtype=dtype).expand(3, 3)
        tensorTwice = tensor.repeat(1, 2)
        onesTwice = torch.cat((ones, ones))

        embedding = embedding.to(dtype=dtype).to(device)
        tensor = tensor.to(device)
        ones = ones.to(device)
        tensorTwice = tensorTwice.to(device)
        onesTwice = onesTwice.to(device)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensor)
        self.assertEqual(embedding.weight.grad._values(), ones)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        tensor[0, 0] = 8
        embedding(tensor[0]).sum().backward()
        tensorTwice[0, 3] = 8
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypes(torch.float32)
    def test_embedding_max_norm_backward(self, device, dtype):
        # can't use gradcheck since in place renorm makes analytical gradients different from produced ones
        weight = torch.randn((4, 4), device=device, dtype=dtype) * 2
        weight.requires_grad_()
        inp_list = [0, 1, 2, 2]
        inp = torch.tensor(inp_list, device=device)
        out = nn.functional.embedding(inp, weight, max_norm=1.0).sum()
        out.backward()

        expected_grad = (
            torch.tensor([[1.0, 1.0, 2.0, 0.0]], device=device, dtype=dtype)
            .transpose(0, 1)
            .expand(4, 4)
        )
        self.assertEqual(weight.grad, expected_grad)

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypes(torch.float32)
    def test_embedding_max_norm_fwd_AD(self, device, dtype):
        if torch.device(device).type == "xla":
            self.skipTest("forward AD doesn't work on xla")

        # can't use gradcheck since in place renorm makes analytical gradients different from produced ones
        weight = torch.randn((4, 4), device=device, dtype=dtype) * 2
        tangent = torch.ones((4, 4), device=device, dtype=dtype)
        inp = torch.tensor([[0, 1], [2, 2]], device=device)
        with torch.autograd.forward_ad.dual_level():
            dual_weight = torch.autograd.forward_ad.make_dual(weight, tangent)
            out = nn.functional.embedding(inp, dual_weight, max_norm=1.0)
            jvp = torch.autograd.forward_ad.unpack_dual(out).tangent

        expected_grad = torch.ones((2, 2, 4), device=device, dtype=dtype)
        self.assertEqual(jvp, expected_grad)

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypes(torch.float32)
    def test_embedding_padding_idx(self, device, dtype):
        embedding = nn.Embedding(10, 20, padding_idx=0).to(device, dtype)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=0, sparse=True).to(device, dtype)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        # negative indexing check for padding_idx
        # padding_idx=-2, num_embeddings=10 ==> index 8 padded
        embedding = nn.Embedding(10, 20, padding_idx=-2).to(device, dtype)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=-2, sparse=True).to(device, dtype)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        # change padding vector
        padding_vector = torch.ones(20, dtype=dtype, device=device)
        embedding = nn.Embedding(10, 20, padding_idx=2, sparse=True).to(device, dtype)
        with torch.no_grad():
            embedding.weight[2] = padding_vector
        input = torch.tensor([0, 2], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[1], padding_vector)

        # out of bounds check for padding_idx
        self.assertRaises(
            AssertionError,
            nn.Embedding,
            num_embeddings=10,
            embedding_dim=20,
            padding_idx=25,
        )
        self.assertRaises(
            AssertionError,
            nn.Embedding,
            num_embeddings=10,
            embedding_dim=20,
            padding_idx=-25,
        )

        padding_idx = 0
        embedding = nn.Embedding(5, 2, padding_idx=padding_idx).to(device, dtype)
        for n in (
            1,
            2,
            1000,
        ):  # Need large N to trigger all the methods we have implemented
            for other_indices in ([], [1, 3], [2]):
                indices = torch.tensor(
                    other_indices + [padding_idx] * n, dtype=torch.long
                ).to(device)
                pre = embedding.weight[padding_idx].clone()
                embedding(indices).sum().backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

                # test double backward
                emb_sum = embedding(indices).sum()
                emb_grad = torch.autograd.grad(
                    outputs=emb_sum,
                    inputs=list(embedding.parameters()),
                    retain_graph=True,
                )
                scalar = emb_grad[0].sum() + emb_sum
                scalar.backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

    # Check correctness of torch.nn.functional.embedding_bag forward and
    # backward functions with padding_idx, given a 1D input separated into bags
    # with an offset array. Compare against an equivalent 2D input that uses
    # padding indices to fill in the gaps indicated by the offset array

    @skipIfTorchDynamo("see https://github.com/pytorch/pytorch/pull/95621")
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    def test_embedding_bag_1D_padding_idx(self, device, dtype):
        num_features = 3
        max_indices_per_bag = 10
        num_bags = 10
        num_words = 100

        def gen_1D_indices_offsets(include_last_offset, allpad):
            indices = []
            offsets = []
            cur_offset = 0

            # Make one bag full and one bag empty, for extra coverage
            empty_bag = random.randint(0, num_bags - 1)
            full_bag = empty_bag
            while full_bag == empty_bag:
                full_bag = random.randint(0, num_bags - 1)

            for bag in range(num_bags):
                offsets.append(cur_offset)
                if bag == full_bag:
                    bag_size = max_indices_per_bag
                elif bag == empty_bag:
                    bag_size = 0
                else:
                    bag_size = random.randint(1, max_indices_per_bag - 1)
                indices += [
                    1 if allpad else random.randint(0, num_words - 1)
                    for _ in range(bag_size)
                ]
                cur_offset += bag_size

            # embedding_bag requires first entry of offsets to be 0
            assert offsets[0] == 0

            indices = torch.tensor(indices, device=device)

            if include_last_offset:
                offsets.append(indices.size(0))

            offsets = torch.tensor(offsets, device=device)

            return indices, offsets

        # Convert a 1-D indices-offsets representation into 2-D. Fill any empty
        # indices with padding_idx
        def gen_2D_indices_from_1D(
            indices_1D, offsets, include_last_offset, padding_idx
        ):
            assert offsets[0] == 0
            if include_last_offset:
                offsets = offsets[:-1]
            indices_2D = torch.empty(
                num_bags, max_indices_per_bag, device=device, dtype=torch.long
            )
            for bag in range(num_bags):
                # Determine the start and end position of the bag within indices_1D
                start = offsets[bag]
                end = len(indices_1D) if bag + 1 == num_bags else offsets[bag + 1]
                end = min(len(indices_1D), end)

                # Pull out the bag's indices from indices_1D, and fill any
                # remaining space with padding indices
                indices_in_bag = []
                for item_pos in range(0, max_indices_per_bag):
                    if (start + item_pos) < end:
                        indices_in_bag.append(indices_1D[start + item_pos])
                    else:
                        indices_in_bag.append(padding_idx)
                indices_2D[bag] = torch.tensor(indices_in_bag, device=device)

            return indices_2D

        test_cases = product(
            ["max", "mean", "sum"], [False, True], [False, True], [False, True]
        )

        for mode, sparse, include_last_offset, allpad in test_cases:
            # Max sparse and bfloat16 are not supported
            if mode == "max":
                if sparse or (dtype == torch.bfloat16):
                    continue
            indices_1D, offsets = gen_1D_indices_offsets(include_last_offset, allpad)
            for padding_idx_1D in list(set(indices_1D.tolist())) + [None]:
                msg = (
                    f"mode: '{mode}', sparse: {sparse}, include_last_offset: {include_last_offset}, "
                    f"padding_idx_1D: {padding_idx_1D}"
                )

                # If 1D input does not use a padding index, we still need one for the 2D input,
                # so we can add one dummy word to the weights to act as the padded word
                padding_idx_2D = (
                    padding_idx_1D if padding_idx_1D is not None else num_words
                )
                num_words_with_padding = (
                    num_words if padding_idx_1D is not None else num_words + 1
                )

                indices_2D = gen_2D_indices_from_1D(
                    indices_1D, offsets, include_last_offset, padding_idx_2D
                )

                weights = torch.randn(
                    num_words_with_padding,
                    num_features,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                weights_check = weights.detach().clone().requires_grad_(True)

                bag = torch.nn.functional.embedding_bag(
                    indices_1D,
                    weights,
                    offsets,
                    padding_idx=padding_idx_1D,
                    mode=mode,
                    sparse=sparse,
                    include_last_offset=include_last_offset,
                )

                bag_check = torch.nn.functional.embedding_bag(
                    indices_2D,
                    weights_check,
                    padding_idx=padding_idx_2D,
                    mode=mode,
                    sparse=sparse,
                )
                self.assertEqual(bag, bag_check, msg=msg)

                bag.sum().backward()
                bag_check.sum().backward()

                # Sometimes, half dtype gradients mismatch by a greater amount
                # than other dtypes
                if dtype in [torch.half, torch.bfloat16]:
                    atol = 0.01
                    rtol = 0.01
                else:
                    atol = None
                    rtol = None
                self.assertEqual(
                    weights.grad, weights_check.grad, msg=msg, atol=atol, rtol=rtol
                )

    # Check correctness of torch.nn.functional.embedding_bag forward and
    # backward functions with padding_idx, given a 2D indices input. Compare
    # against torch.nn.functional.embedding followed by a reduction.
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    def test_embedding_bag_2D_padding_idx(self, device, dtype):
        # Use a Python implementation of embedding_bag with padding_idx support
        # to check torch.nn.functional.embedding_bag correctness
        def embedding_bag_check(indices, weights, mode, sparse, padding_idx):
            assert padding_idx is not None
            embedding = torch.nn.functional.embedding(
                indices, weights, padding_idx=padding_idx, sparse=sparse
            )

            reduction_dim = indices.dim() - 1

            if mode == "sum" or mode == "mean":
                # We must avoid including elements at padding_idx in the
                # sum/mean, so multiply those elements by 0, and multiply
                # all other elements by 1
                per_sample_weights = indices.ne(padding_idx).to(dtype).unsqueeze(-1)
                res = embedding.mul(per_sample_weights).sum(dim=reduction_dim)

                if mode == "mean":
                    weights_sum = per_sample_weights.sum(dim=reduction_dim)
                    res = res.div(weights_sum)

            elif mode == "max":
                # We must avoid allowing elements at padding_idx to be chosen
                # as the max, so set those elements to negative infinity
                res = embedding.masked_fill(
                    indices.unsqueeze(-1) == padding_idx, -float("inf")
                ).amax(dim=reduction_dim)

            else:
                raise RuntimeError(f"mode '{mode}' is not available")

            # If a row is all padding, set its corresponding result row to 0.
            # This is needed because the above mean and max mode
            # implementations set these elements to nan and -inf, respectively
            if mode in ["mean", "max"]:
                res = res.masked_fill(
                    indices.eq(padding_idx).all(dim=-1).unsqueeze(-1), 0
                )

            return res

        num_features = 3
        num_words = 10
        indices_dim1 = 10

        for mode, sparse, allpad, indices_dim0 in product(
            ["max", "mean", "sum"], [False, True], [False, True], [1, 10]
        ):
            # Max sparse and bfloat16 are not supported
            if mode == "max":
                if sparse or (dtype == torch.bfloat16):
                    continue

            if allpad:
                indices = torch.empty(
                    indices_dim0, indices_dim1, dtype=torch.long, device=device
                ).fill_(1)
            else:
                indices = torch.randint(
                    0, num_words, (indices_dim0, indices_dim1), device=device
                )

                if indices_dim0 > 1:
                    # Fill one row with duplicate index so we can test with a fully
                    # padded row
                    duplicate_row = random.randint(0, indices_dim0 - 1)
                    indices[duplicate_row] = indices[duplicate_row][0]

            for padding_idx in list(set(indices.flatten(0, -1).tolist())):
                weights = torch.randn(
                    num_words,
                    num_features,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                weights_check = weights.detach().clone().requires_grad_(True)

                msg = (
                    f"mode: '{mode}', sparse: {sparse}, padding_idx: {padding_idx}, "
                    f"allpad: {allpad}, indices.size(): {indices.size()}"
                )

                # Check forward with a Python implementation of padding_idx embedding_bag
                bag_check = embedding_bag_check(
                    indices, weights_check, mode, sparse, padding_idx
                )
                bag = torch.nn.functional.embedding_bag(
                    indices, weights, padding_idx=padding_idx, mode=mode, sparse=sparse
                )

                self.assertEqual(bag, bag_check, msg=msg)

                bag_check.sum().backward()
                grad_check = weights_check.grad

                bag.sum().backward()
                grad = weights.grad

                # Sometimes, half dtype gradients mismatch by a greater amount
                # than other dtypes
                if dtype in [torch.half, torch.bfloat16]:
                    atol = 0.01
                    rtol = 0.01
                else:
                    atol = None
                    rtol = None
                self.assertEqual(grad, grad_check, msg=msg, atol=atol, rtol=rtol)

    @onlyCUDA
    @dtypes(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    def test_embedding_max_norm_device(self, device, dtype):
        embedding = nn.Embedding(22, 5, max_norm=1.0).to(device, dtype=dtype)
        # nn.Embedding only takes LongTensor as input
        input = torch.tensor([2, 8, 8, 6], device=device, dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_embedding_bag_empty_input(self, device, dtypes):
        m = 4
        n = 3
        x = torch.tensor([], device=device, dtype=dtypes[0])
        for sparse in [True, False]:
            Embed = torch.nn.EmbeddingBag(m, n, sparse=sparse)
            Embed.to(device)

            output = Embed(
                input=x, offsets=torch.tensor([0], device=device, dtype=dtypes[1])
            )
            self.assertEqual(output, torch.zeros_like(output))

            output = Embed(
                input=x, offsets=torch.tensor([0, 0], device=device, dtype=dtypes[1])
            )
            self.assertEqual(output, torch.zeros_like(output))

    @skipCUDAIf(True, "no out-of-bounds check on CUDA for perf.")
    @dtypes(*itertools.product((torch.float, torch.double), (torch.int, torch.long)))
    @parametrize_test("padding_idx", [None, 0])
    @parametrize_test("mode", ["sum", "mean", "max"])
    def test_embedding_bag_out_of_bounds_idx(self, device, dtypes, padding_idx, mode):
        padding_idx = 0
        w_dtype, idx_dtype = dtypes
        # negative out-of-bound
        idx1 = torch.tensor([[-1, 1]], device=device, dtype=idx_dtype)
        # positive out-of-bound
        idx2 = torch.tensor([[11, 8]], device=device, dtype=idx_dtype)
        weight = torch.randn(10, 2, device=device, dtype=w_dtype)
        if mode == "sum":
            # Only `sum` supports per_sample_weight
            per_sample_weights = (
                None,
                torch.randn_like(idx1, device=device, dtype=w_dtype),
            )
        else:
            per_sample_weights = (None,)

        for p_s_weights, idx in itertools.product(per_sample_weights, (idx1, idx2)):
            msg = "Expected idx >= 0 && idx < num_embeddings"
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.nn.functional.embedding_bag(
                    idx,
                    weight,
                    per_sample_weights=p_s_weights,
                    padding_idx=padding_idx,
                    mode=mode,
                )

    def test_embedding_bag_dimension_errors(self, device):
        funcs = (
            lambda x, y, z: torch.nn.functional.embedding_bag(y, x, z),
            torch.embedding_bag,
            torch._embedding_bag,
            torch._embedding_bag_forward_only,
        )
        for i, f in enumerate(funcs):
            err_type = (ValueError, RuntimeError) if i == 0 else RuntimeError

            weight = torch.full(
                (
                    2,
                    6,
                ),
                0,
                dtype=torch.float64,
                device=device,
            )
            indices = torch.full(
                (
                    2,
                    0,
                    0,
                    6,
                    6,
                ),
                2,
                dtype=torch.int64,
                device=device,
            )
            offsets = torch.full((2, 0, 0, 6, 6), 0, dtype=torch.int64, device=device)

            if i == 0:
                error_msg = "input has to be 1D or 2D Tensor"
            else:
                error_msg = "input has to be a 1D or 2D Tensor"
            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type, error_msg, lambda: f(weight, indices, offsets)
            )

            weight = torch.full((2, 2), 0, dtype=torch.float64, device=device)
            indices = torch.full((2,), 1, dtype=torch.int64, device=device)

            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type,
                "offsets has to be a 1D Tensor",
                lambda: f(weight, indices, offsets),
            )

            weight = torch.full((2, 2, 2), 0, dtype=torch.float64, device=device)
            indices = torch.full((2,), 2, dtype=torch.int64, device=device)
            offsets = torch.full((2,), 0, dtype=torch.int64, device=device)

            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type,
                "weight has to be a 2D Tensor",
                lambda: f(weight, indices, offsets),
            )

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_EmbeddingBag_per_sample_weights_failures(self, device, dtypes):
        # Failure 1: mismatched embeddings / per_sample_weights dtype (only on CPU device)
        es = nn.EmbeddingBag(5, 2, mode="sum").to(dtype=torch.float, device=device)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        per_sample_weights = torch.randn_like(input, dtype=torch.double, device=device)
        if device == "cpu":
            with self.assertRaisesRegex(RuntimeError, "have the same type as"):
                es(input, offsets, per_sample_weights)

        # Failure 2.1: input/per_sample_weights have different sizes (1d input)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        per_sample_weights = torch.randn(5, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, "same shape as the input"):
            es(input, offsets, per_sample_weights)

        # Failure 2.2: input/per_sample_weights have different sizes (2d input)
        input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
        offsets = None
        per_sample_weights = torch.randn(7 * 3, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, "same shape as the input"):
            es(input, offsets, per_sample_weights)

        # Failure 3: Unsupported per_sample_weights and mode=('max', 'mean')
        for unsupported_mode in ("max", "mean"):
            es = nn.EmbeddingBag(5, 2, mode=unsupported_mode).to(
                dtype=torch.float, device=device
            )
            input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
            offsets = None
            per_sample_weights = torch.randn(7, 3, dtype=torch.float, device=device)
            with self.assertRaisesRegex(
                NotImplementedError, "only supported for mode='sum'"
            ):
                es(input, offsets, per_sample_weights)

    def _embedding_bag_reference_impl(
        self,
        input,
        weight,
        offsets=None,
        mode="sum",
        per_sample_weights=None,
        include_last_offset=False,
    ):
        assert mode == "sum" or per_sample_weights is None
        assert offsets is not None
        if per_sample_weights is None:
            per_sample_weights = torch.ones(input.size()).to(
                dtype=weight.dtype, device=weight.device
            )
        assert input.numel() == per_sample_weights.numel()

        bags = []
        long_input = input.to(torch.long)
        embeddings = weight.index_select(0, long_input) * per_sample_weights.unsqueeze(
            1
        )
        if include_last_offset:
            for index in range(len(offsets) - 1):
                offset = offsets[index]
                next_offset = offsets[index + 1]
                length = next_offset - offset
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    if mode == "sum":
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == "mean":
                        bags.append(
                            embeddings.narrow(0, offset, length).sum(0).div(length)
                        )
                    else:
                        assert mode == "max"
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        else:
            for index, offset in enumerate(offsets):
                if index + 1 < len(offsets):
                    next_offset = offsets[index + 1]
                else:
                    next_offset = len(long_input)
                length = next_offset - offset
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    if mode == "sum":
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == "mean":
                        bags.append(
                            embeddings.narrow(0, offset, length).sum(0).div(length)
                        )
                    else:
                        assert mode == "max"
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        return torch.stack(bags)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.half, torch.bfloat16, torch.float, torch.double),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    def test_EmbeddingBag_empty_per_sample_weights_and_offsets(self, device, dtypes):
        # Test empty input and per sample weight, and backward pass. There was a CUDA
        # invalid configuration bug (more context in #46572)
        def test_per_sample_weights(mode, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            input = torch.tensor([], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 0, 0, 0], device=device, dtype=dtypes[1])
            per_sample_weights = torch.randn_like(
                input, dtype=dtypes[2]
            ).requires_grad_(trainable_scale)
            ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                trainable_scale
            )
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights
            )
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            grad = torch.randn_like(expected)
            result.backward(grad)
            # the reference impl doesn't have grad fn for empty input; but the grad should
            # simply be a zero tensor
            ref_weights_grad = torch.zeros_like(es.weight)
            self.assertEqual(
                es.weight.grad,
                ref_weights_grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            if trainable_scale:
                ref_per_sample_weights_grad = torch.empty_like(per_sample_weights)
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights_grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        modes = ("sum",)
        trainable_scale = (True, False)
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    def test_EmbeddingBag_per_sample_weights_and_offsets(self, device, dtypes):
        def test_per_sample_weights(mode, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])
            per_sample_weights = torch.randn_like(
                input, dtype=dtypes[2]
            ).requires_grad_(trainable_scale)
            ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                trainable_scale
            )
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights
            )
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            grad = torch.randn_like(expected).to(dtype=dtypes[2], device=device)
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(
                es.weight.grad,
                reference_weights.grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            if trainable_scale:
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights.grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        modes = ("sum",)
        trainable_scale = (True, False)
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    def test_EmbeddingBag_per_sample_weights_and_new_offsets(self, device, dtypes):
        def test_per_sample_weights_new_offsets(
            mode, trainable_scale, include_last_offset, has_weight=True
        ):
            es = nn.EmbeddingBag(
                5, 2, mode=mode, include_last_offset=include_last_offset
            ).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])

            if include_last_offset:
                offsets = torch.cat(
                    (
                        offsets,
                        torch.tensor([input.size(0)], device=device, dtype=dtypes[1]),
                    ),
                    0,
                )

            if has_weight:
                per_sample_weights = torch.randn_like(
                    input, device=device, dtype=dtypes[2]
                ).requires_grad_(trainable_scale)
                ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                    trainable_scale
                )
            else:
                per_sample_weights = None
                ref_per_sample_weights = None

            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input,
                reference_weights,
                offsets,
                mode,
                ref_per_sample_weights,
                include_last_offset,
            )
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            grad = torch.randn_like(expected)
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(
                es.weight.grad,
                reference_weights.grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            if has_weight and trainable_scale:
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights.grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        trainable_scale = (True, False)
        include_last_offset_list = (True, False)
        modes = (("sum", False), ("sum", True), ("max", False), ("mean", False))
        for (mode, has_weight), trainable, include_last_offset in itertools.product(
            modes, trainable_scale, include_last_offset_list
        ):
            test_per_sample_weights_new_offsets(
                mode, trainable, include_last_offset, has_weight
            )

    def _test_EmbeddingBag_vs_Embedding(
        self,
        N,
        D,
        B,
        L,
        max_norm=None,
        mode="mean",
        device="cpu",
        wdtype=torch.float,
        dtype=torch.long,
        test_per_sample_weights=False,
        trainable_per_sample_weights=False,
        sparse=False,
        test_backward=True,
        backward_prec=None,
    ):
        es = nn.EmbeddingBag(N, D, mode=mode, sparse=sparse, max_norm=max_norm).to(
            device, wdtype
        )
        e = nn.Embedding(N, D, max_norm=max_norm).to(device, wdtype)
        e.weight.data.copy_(es.weight)
        input = torch.randint(N, (B, L), device=device, dtype=dtype)
        offsets = torch.arange(0, B, device=device, dtype=dtype).mul_(L)
        grad_output = torch.rand(B, D, device=device, dtype=wdtype)

        if test_per_sample_weights:
            # To prevent large gradients, weights should sum to 1 for each bag
            per_sample_weights = torch.randn(B, L, device=device, dtype=wdtype).softmax(
                dim=-1
            )
            per_sample_weights_reference = per_sample_weights.clone().requires_grad_(
                trainable_per_sample_weights
            )
            per_sample_weights.requires_grad_(trainable_per_sample_weights)
            output = es(input.view(-1), offsets, per_sample_weights.view(-1))
        else:
            output = es(input.view(-1), offsets)
            per_sample_weights = None
            per_sample_weights_reference = None

        if mode == "sum":
            if test_per_sample_weights:
                ref_output = (
                    e(input) * per_sample_weights_reference.unsqueeze(-1)
                ).sum(1)
            else:
                ref_output = e(input).sum(1)
        elif mode == "mean":
            assert not test_per_sample_weights
            ref_output = e(input).mean(1)
        elif mode == "max":
            assert not test_per_sample_weights
            ref_output = e(input).max(1)[0]

        self.assertEqual(output, ref_output, atol=dtype2prec_DONTUSE[wdtype], rtol=0)

        if not test_backward:
            return

        output.backward(grad_output)
        ref_output.backward(grad_output)
        es_weight_grad = es.weight.grad
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()

        # We have more floating point error here because we are dealing with larger numbers
        if backward_prec is None:
            needed_prec = dtype2prec_DONTUSE[wdtype] * 5
            rtol = 0.02 if wdtype == torch.half else 0
        else:
            needed_prec = backward_prec
            rtol = 0

        self.assertEqual(es_weight_grad, e.weight.grad, atol=needed_prec, rtol=rtol)

        if test_per_sample_weights and trainable_per_sample_weights:
            self.assertEqual(
                per_sample_weights.grad,
                per_sample_weights_reference.grad,
                atol=dtype2prec_DONTUSE[wdtype],
                rtol=0,
            )

    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long), (torch.half, torch.float, torch.double)
        )
    )
    @dtypes(*itertools.product((torch.int, torch.long), (torch.float, torch.double)))
    def test_EmbeddingBag_per_sample_weights_and_no_offsets(self, device, dtypes):
        def run_tests(mode, sparse, trainable_per_sample_weights):
            kwargs = dict(
                test_per_sample_weights=True,
                device=device,
                mode=mode,
                wdtype=dtypes[1],
                dtype=dtypes[0],
                sparse=sparse,
                trainable_per_sample_weights=trainable_per_sample_weights,
            )

            # Simple case
            self._test_EmbeddingBag_vs_Embedding(2, 3, 5, 7, **kwargs)

            # B * L > 1000
            self._test_EmbeddingBag_vs_Embedding(2, 5, 53, 23, **kwargs)

            # Large num_embedding
            self._test_EmbeddingBag_vs_Embedding(101, 5, 3, 7, **kwargs)

            # Large embedding_dim
            self._test_EmbeddingBag_vs_Embedding(2, 101, 3, 7, **kwargs)

        modes = ("sum",)
        sparsity = (True, False)
        trainable_scale = (True, False)
        for mode, sparse, trainable_per_sample_weights in itertools.product(
            modes, sparsity, trainable_scale
        ):
            run_tests(mode, sparse, trainable_per_sample_weights)

        # Test CUDA Dense on half precision
        if device == "cuda":
            modes = ("sum",)
            sparsity = (False,)
            trainable_scale = (True, False)
            for mode, sparse, trainable_per_sample_weights in itertools.product(
                modes, sparsity, trainable_scale
            ):
                run_tests(mode, sparse, trainable_per_sample_weights)

    def _test_EmbeddingBag(
        self,
        device,
        mode,
        sparse,
        wdtype=torch.double,
        dtype=torch.long,
        odtype=torch.long,
        test_backward=True,
    ):
        # check a known test example
        es = nn.EmbeddingBag(5, 2, mode=mode, sparse=sparse).to(device, wdtype)
        es.weight.data.copy_(
            torch.arange(1, 11, device=device).view_as(es.weight).to(wdtype)
        )
        input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtype)
        offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=odtype)

        grad_output = torch.tensor([1, 2, 3, 4], device=device, dtype=wdtype).view(2, 2)
        grad_output_with_empty = torch.tensor(
            [99, 99, 1, 2, 99, 99, 3, 4, 99, 99], device=device, dtype=wdtype
        ).view(5, 2)

        if mode == "sum" or mode == "mean":
            denominator = 1 if mode == "sum" else 3
            expected_output = (
                torch.tensor([[13, 16], [13, 16]], device=device, dtype=wdtype)
                / denominator
            )

            expected_output_with_empty = (
                torch.tensor(
                    [[0, 0], [13, 16], [0, 0], [13, 16], [0, 0]],
                    device=device,
                    dtype=wdtype,
                )
                / denominator
            )

            expected_grad_weight = (
                torch.tensor(
                    [[3, 4], [5, 8], [0, 0], [1, 2], [3, 4]],
                    device=device,
                    dtype=wdtype,
                )
                / denominator
            )
        elif mode == "max":
            expected_output = torch.tensor(
                [[7, 8], [9, 10]], device=device, dtype=wdtype
            )

            expected_output_with_empty = torch.tensor(
                [[0, 0], [7, 8], [0, 0], [9, 10], [0, 0]], device=device, dtype=wdtype
            )

            expected_grad_weight = torch.tensor(
                [[0, 0], [0, 0], [0, 0], [1, 2], [3, 4]], device=device, dtype=wdtype
            )
        output = es(input, offsets)
        output.backward(grad_output_with_empty)

        es_weight_grad = es.weight.grad
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()
        self.assertEqual(output, expected_output_with_empty)
        self.assertEqual(
            es_weight_grad,
            expected_grad_weight,
            atol=dtype2prec_DONTUSE[wdtype],
            rtol=0,
        )

        # check same example except as 2D (2 x 3)
        input = input.view(2, -1)
        es.zero_grad()
        output = es(input)
        output.backward(grad_output)

        es_weight_grad = es.weight.grad
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()
        self.assertEqual(output, expected_output)
        self.assertEqual(
            es_weight_grad,
            expected_grad_weight,
            atol=dtype2prec_DONTUSE[wdtype],
            rtol=0,
        )

        # test all empty bags
        es.zero_grad()
        inputs = torch.tensor([], dtype=dtype, device=device)
        offsets = torch.tensor([0, 0, 0, 0], dtype=odtype, device=device)
        es(inputs, offsets).sum().backward()
        dense_grad = es.weight.grad
        if dense_grad.is_sparse:
            dense_grad = dense_grad.to_dense()
        self.assertEqual(dense_grad, torch.zeros_like(es.weight))

        # now compare EmbeddingBag vs Embedding + Sum/Mean, for constant bag length
        N, D, B, L = (
            random.randint(1, 100),
            random.randint(1, 100),
            random.randint(1, 50),
            random.randint(1, 50),
        )
        kwargs = dict(
            mode=mode,
            sparse=sparse,
            device=device,
            wdtype=wdtype,
            dtype=dtype,
            test_backward=test_backward,
        )
        self._test_EmbeddingBag_vs_Embedding(N, D, B, L, **kwargs)
        for max_norm in (None, 3):
            for p in itertools.product([1, 2], repeat=4):
                self._test_EmbeddingBag_vs_Embedding(*p, max_norm=max_norm, **kwargs)

        # check that giving illegal input combos raises error
        es = nn.EmbeddingBag(10, 20, mode=mode, sparse=sparse)
        input = torch.ones(3, 4, dtype=dtype)
        offset = torch.arange(0, 3, dtype=odtype)
        torch._dynamo.disable(self.assertRaises)(ValueError, lambda: es(input, offset))
        torch._dynamo.disable(self.assertRaises)(ValueError, lambda: es(input.view(-1)))
        offset[0] = 1
        if self.device_type == "cpu":
            torch._dynamo.disable(self.assertRaises)(
                RuntimeError, lambda: es(input.view(-1), offset)
            )
            offset[0] = 0
            offset[-1] = 100
            torch._dynamo.disable(self.assertRaises)(
                RuntimeError, lambda: es(input.view(-1), offset)
            )

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    def test_embedding_bag_device(self, device, dtypes):
        if IS_JETSON and torch.bfloat16 in dtypes and device == "cpu":
            self.skipTest("bfloat16 not supported with Jetson cpu")
        with set_default_dtype(torch.double):
            self._test_EmbeddingBag(
                device,
                "sum",
                False,
                wdtype=dtypes[2],
                dtype=dtypes[0],
                odtype=dtypes[1],
            )
            self._test_EmbeddingBag(
                device,
                "mean",
                False,
                wdtype=dtypes[2],
                dtype=dtypes[0],
                odtype=dtypes[1],
            )
            self._test_EmbeddingBag(
                device,
                "max",
                False,
                wdtype=dtypes[2],
                dtype=dtypes[0],
                odtype=dtypes[1],
            )

            test_backward = False
            if self.device_type == "cuda":
                # see 'todo' in test_embedding_bag.
                test_backward = dtypes[2] is not torch.float16
            elif self.device_type == "cpu":
                # TODO: figure out why precision on sparse embeddings isn't the
                # same as for dense.
                test_backward = (
                    dtypes[2] is not torch.float and dtypes[2] is not torch.float16
                )

            self._test_EmbeddingBag(
                device,
                "sum",
                True,
                wdtype=dtypes[2],
                dtype=dtypes[0],
                odtype=dtypes[1],
                test_backward=test_backward,
            )
            self._test_EmbeddingBag(
                device,
                "mean",
                True,
                wdtype=dtypes[2],
                dtype=dtypes[0],
                odtype=dtypes[1],
                test_backward=test_backward,
            )

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    def test_embedding_bag_non_contiguous_weight(self, device, dtypes):
        weight_tensor = torch.randn(3, 4, dtype=dtypes[2], device=device)

        weight_tensor_non_contig = weight_tensor[
            :, :3
        ]  # This is non-contiguous strided.
        weight_tensor_contig = (
            weight_tensor_non_contig.clone().contiguous()
        )  # Contig-strided.

        index = torch.tensor([0, 1, 2], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 2], dtype=dtypes[1], device=device)
        for mode in ["sum", "mean", "max"]:
            output_non_contig = F.embedding_bag(
                input=index,
                weight=weight_tensor_non_contig,
                offsets=offsets,
                mode=mode,
            )
            output_contig = F.embedding_bag(
                input=index,
                weight=weight_tensor_contig,
                offsets=offsets,
                mode=mode,
            )
        self.assertEqual(output_non_contig, output_contig)

    @onlyNativeDeviceTypes  # currently fails on XLA
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_embedding_bag_bfloat16(self, device, dtypes):
        with set_default_dtype(torch.double):
            self._test_EmbeddingBag(
                device,
                "sum",
                True,
                wdtype=torch.bfloat16,
                dtype=dtypes[0],
                odtype=dtypes[1],
                test_backward=True,
            )
            self._test_EmbeddingBag(
                device,
                "mean",
                True,
                wdtype=torch.bfloat16,
                dtype=dtypes[0],
                odtype=dtypes[1],
                test_backward=True,
            )

    @onlyNativeDeviceTypes  # currently fails on XLA
    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_embedding_bag_half(self, device, dtypes):
        self._test_EmbeddingBag(
            device,
            "sum",
            True,
            wdtype=torch.float16,
            dtype=dtypes[0],
            odtype=dtypes[1],
            test_backward=True,
        )

    @parametrize_test(
        "bag_use_grad,per_sample_weights_use_grad",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_embedding_bag_per_sample_weights_grad(
        self, device, bag_use_grad: bool, per_sample_weights_use_grad: bool
    ):
        bag = torch.nn.EmbeddingBag(256, 256, mode="sum", device=device)
        bag.requires_grad_(bag_use_grad)
        x = torch.arange(1, 5, device=device).expand(3, -1)
        w = torch.rand(3, 4, device=device, requires_grad=per_sample_weights_use_grad)
        bag(x, per_sample_weights=F.softmax(w, dim=-1))


instantiate_device_type_tests(TestEmbeddingNNDeviceType, globals())
instantiate_parametrized_tests(TestEmbeddingNN)

if __name__ == "__main__":
    run_tests()
