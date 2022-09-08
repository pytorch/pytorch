# Owner(s): ["module: nn"]

import contextlib
import torch
import torch.nn.functional as F
import unittest


from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
    freeze_rng_state,
)
from torch.testing._internal.common_cuda import TEST_CUDA


@contextlib.contextmanager
def set_default_dtype(dtype):
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)

class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    # set up device, if 'cuda' is detected, test it in cuda, or else, in cpu 
    device_list = ['cuda' if TEST_CUDA else 'cpu']

    @unittest.skipIf(not TEST_CUDA, 'CUDA does not work with ASAN') 
    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    # freeze_rng_state() doesn't work in CUDA,dropout will make the resutls incomparable,thus,no dropout for CUDA unit test. 
    @parametrize("dropout_p", [0.0])
    @parametrize("device", device_list)
    def test_scaled_dot_product_attention(self, device, input_dim, attn_mask_dim, is_causal, dropout_p):
        # TODO: Support cross-device / dtype testing properly when instantiate_device_type_tests() is used.
        dtypes = [torch.double, torch.float]
        for dtype in dtypes:

            def rand_tensor(*shape):
                return torch.randn(shape, device=device, dtype=dtype)

            # This test compares python and C++ implementations of SDP.
            N, N_prime, L, S, E = 5, 2, 4, 3, 6
            if input_dim == 3:
                query = rand_tensor(N, L, E)
                key = rand_tensor(N, S, E)
                value = rand_tensor(N, S, E)
            elif input_dim == 4:
                query = rand_tensor(N, N_prime, L, E)
                key = rand_tensor(N, N_prime, S, E)
                value = rand_tensor(N, N_prime, S, E)
            else:
                self.fail(f'Invalid input_dim {input_dim} encountered in SDP test')

            attn_mask = None
            if attn_mask_dim is not None:
                assert attn_mask_dim in [2, input_dim]
                mask_size = (L, S) if attn_mask_dim == 2 else ((N, L, S) if input_dim == 3 else (N, N_prime, L, S))
                attn_mask = (torch.ones(mask_size, device=device, dtype=torch.bool).tril() if is_causal
                             else torch.randint(0, 2, size=mask_size, device=device, dtype=torch.bool))

            with freeze_rng_state():
                # Python impl only supports float mask and 3D inputs.
                attn_mask_float = attn_mask
                if attn_mask_float is not None:
                    attn_mask_float = torch.zeros_like(attn_mask, dtype=query.dtype)
                    attn_mask_float.masked_fill_(attn_mask.logical_not(), float("-inf"))
                q, k, v = query.view(-1, L, E), key.view(-1, S, E), value.view(-1, S, E)
                a = attn_mask_float
                if a is not None and attn_mask_dim > 3:
                    a = a.view(-1, L, S)
                expected = F._scaled_dot_product_attention(
                    q, k, v, attn_mask=a, dropout_p=dropout_p)
                if input_dim > 3:
                    expected = (expected[0].view(-1, N_prime, L, E), expected[1].view(-1, N_prime, L, S))

            need_attn_weights: bool = True
            with freeze_rng_state():
                if is_causal:
                    # NB: Don't pass attn_mask here
                    actual = torch.ops.aten._scaled_dot_product_attention(
                        query, key, value, None, dropout_p, need_attn_weights, is_causal)

                    # Error case: both explicit attn_mask and is_causal are set
                    with self.assertRaisesRegex(RuntimeError,
                                                "Explicit attn_mask should not be set when is_causal=True"):
                        torch.ops.aten._scaled_dot_product_attention(
                            query, key, value, attn_mask, dropout_p, need_attn_weights, is_causal)
                else:
                    actual = torch.ops.aten._scaled_dot_product_attention(
                        query, key, value, attn_mask, dropout_p, need_attn_weights, is_causal)

            self.assertEqual(actual, expected)

instantiate_parametrized_tests(TestTransformers)

if __name__ == '__main__':
    run_tests()
