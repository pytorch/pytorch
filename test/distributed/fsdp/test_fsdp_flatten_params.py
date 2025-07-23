# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._flat_param import (
    FlatParamHandle,
    FlatParamShardMetadata,
    HandleShardingStrategy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestFlattenParams(FSDPTest):
    """Tests parameter flattening and shard metadata logic."""

    @property
    def world_size(self) -> int:
        # Clamp the world size to 1 since these unit tests either exercise only
        # the flattening logic or check sharding subroutines directly without
        # requiring multiple ranks
        return 1

    def _get_default_config(self):
        return {
            "device": torch.device("cuda"),
            "sharding_strategy": HandleShardingStrategy.FULL_SHARD,
            "offload_params": False,
            "mp_param_dtype": None,
            "mp_reduce_dtype": None,
            "keep_low_precision_grads": False,
            "process_group": self.process_group,
            "use_orig_params": False,
            "fsdp_extension": None,
        }

    def _get_transformer(self, seed=0):
        torch.manual_seed(seed)  # keep everything deterministic
        module = torch.nn.Transformer(
            d_model=32,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )
        module.dummy_buffer = nn.Buffer(torch.tensor(1.0))

        def get_input(device, dtype):
            torch.manual_seed(1)  # keep everything deterministic
            src = torch.rand(20, 8, 32).to(device=device, dtype=dtype)  # T x B x C
            tgt = torch.rand(10, 8, 32).to(device=device, dtype=dtype)  # T x B x C
            return (src, tgt)

        module.get_input = get_input
        return module

    def _get_shared_params_transformer(self, seed=0):
        module = self._get_transformer(seed=seed)
        # share the FFNs
        for enc_layer, dec_layer in zip(module.encoder.layers, module.decoder.layers):
            dec_layer.linear1.weight = enc_layer.linear1.weight
            dec_layer.linear2.weight = enc_layer.linear2.weight
        return module

    @skip_if_lt_x_gpu(1)
    def test_partial_flattening(self):
        """Tests flattening some submodules but not others."""
        self.run_subtests(
            {"half": [False, True]},
            self._test_partial_flattening,
        )

    def _test_partial_flattening(self, half: bool):
        module = self._get_transformer()
        if half:
            module = module.half()
        numel = sum(p.numel() for p in module.parameters())

        encoder_1_params = list(module.encoder.layers[1].parameters())
        decoder_0_params = list(module.decoder.layers[0].parameters())
        params_to_flatten = encoder_1_params + decoder_0_params
        num_params = [len(encoder_1_params), len(decoder_0_params)]
        numel_to_flatten = sum(p.numel() for p in params_to_flatten)
        module.encoder.layers[1] = FSDP(module.encoder.layers[1])
        module.decoder.layers[0] = FSDP(module.decoder.layers[0])
        flat_params = [
            module.encoder.layers[1]._flat_param,
            module.decoder.layers[0]._flat_param,
        ]

        self.assertEqual(sum(fp.numel() for fp in flat_params), numel_to_flatten)
        self.assertEqual(sum(p.numel() for p in module.parameters()), numel)

        # Check that flattened parameters have been replaced with a single
        # `FlatParameter`
        self.assertEqual(len(list(module.encoder.layers[1].parameters())), 1)
        self.assertEqual(len(list(module.decoder.layers[0].parameters())), 1)

        # Check that non-flattened parameters remain
        self.assertEqual(
            len(list(module.encoder.layers[0].parameters())), num_params[0]
        )
        self.assertEqual(
            len(list(module.decoder.layers[1].parameters())), num_params[1]
        )

        # Check that calling `module.to()` affects the `FlatParameter`s
        orig_dtype = params_to_flatten[0].dtype
        new_dtype = torch.float32 if orig_dtype == torch.float16 else torch.float16
        for flat_param in flat_params:
            self.assertEqual(flat_param.dtype, orig_dtype)
        self.assertTrue(
            all(p.dtype == orig_dtype for p in module.encoder.layers[0].parameters())
        )
        module = module.to(dtype=new_dtype)
        for flat_param in flat_params:
            self.assertEqual(flat_param.dtype, new_dtype)
        self.assertTrue(
            all(p.dtype == new_dtype for p in module.encoder.layers[0].parameters())
        )

    def test_flatten_nothing(self):
        """
        Tests that constructing a ``FlatParamHandle`` with no parameters
        raises an error.
        """
        self.run_subtests(
            {"half": [False, True]},
            self._test_flatten_nothing,
        )

    def _test_flatten_nothing(self, half: bool):
        module = self._get_transformer()
        if half:
            module = module.half()
        with self.assertRaisesRegex(
            ValueError,
            "Cannot construct a FlatParamHandle with an empty parameter list",
        ):
            FlatParamHandle(
                [],
                module,
                **self._get_default_config(),
            )

    @skip_if_lt_x_gpu(1)
    def test_empty_module(self):
        """
        Tests flattening an empty module (i.e. one without any parameters).
        """
        module = self._get_empty_module()
        in_data = torch.rand(1)
        ref_out = module(in_data)
        fsdp_module = FSDP(module)
        self.assertEqual(len(list(fsdp_module.parameters())), 0)
        self.assertIsNone(fsdp_module._flat_param)
        fsdp_out = fsdp_module(in_data)
        self.assertEqual(ref_out, fsdp_out)

    def _get_empty_module(self):
        """Returns a module with no parameters."""
        torch.manual_seed(0)  # keep everything deterministic

        class EmptyModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

            def get_input(self, device, dtype):
                torch.manual_seed(1)  # keep everything deterministic
                return torch.rand(1).to(device=device, dtype=dtype)

        return EmptyModule()

    def test_numel_without_shared_params(self):
        """
        Tests that numel is preserved after flattening when there are no shared
        parameters in the module.
        """
        self.run_subtests(
            {"half": [False, True]},
            self._test_numel_without_shared_params,
        )

    def _test_numel_without_shared_params(self, half: bool):
        module = self._get_transformer()
        if half:
            module = module.half()
        self._test_numel(module)

    def test_numel_with_shared_params(self):
        """
        Tests that numel is preserved after flattening when there are shared
        parameters in the module.
        """
        self.run_subtests(
            {"half": [False, True]},
            self._test_numel_with_shared_params,
        )

    def _test_numel_with_shared_params(self, half: bool):
        module = self._get_shared_params_transformer()
        if half:
            module = module.half()
        self._test_numel(module)

    def _test_numel(self, module):
        ref_numel = sum(p.numel() for p in module.parameters())
        params_to_flatten = list(module.parameters())
        flat_param_handle = FlatParamHandle(
            params_to_flatten,
            module,
            **self._get_default_config(),
        )
        self.assertEqual(ref_numel, flat_param_handle.flat_param.numel())

    @skip_if_lt_x_gpu(1)
    def test_output_without_shared_params(self):
        """
        Tests a forward pass after flattening when there are no shared
        parameters in the module.
        """
        self.run_subtests(
            {"half": [False, True]},
            self._test_output_without_shared_params,
        )

    def _test_output_without_shared_params(self, half: bool):
        module = self._get_transformer()
        if half:
            module = module.half()
        self._test_output(module)

    @skip_if_lt_x_gpu(1)
    def test_output_with_shared_params(self):
        """
        Tests a forward pass after flattening when there are shared parameters
        in the module.
        """
        self.run_subtests(
            {"half": [False, True]},
            self._test_output_with_shared_params,
        )

    def _test_output_with_shared_params(self, half: bool):
        module = self._get_shared_params_transformer()
        if half:
            module = module.half()
        self._test_output(module)

    def _test_output(self, module: nn.Module):
        module = module.to(self.rank)
        ref_output = self._get_output(module)
        fsdp_module = FSDP(module)
        fsdp_output = self._get_output(fsdp_module)
        self.assertEqual(ref_output, fsdp_output)

    def _get_output(self, module):
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        input = module.get_input(device, dtype)
        return module(*input)

    @skip_if_lt_x_gpu(1)
    def test_pnorm_after_step_with_shared_params(self):
        """
        Tests for parameter Frobenius norm parity after an optimizer step when
        there are shared parameters in the module. If the parameter sharing is
        handled incorrectly, then an optimizer step should reveal that.
        """
        self.run_subtests(
            {"half": [False, True]},
            self._test_pnorm_after_step_with_shared_params,
        )

    def _test_pnorm_after_step_with_shared_params(self, half: bool):
        module = self._get_shared_params_transformer().to(self.rank)
        if half:
            module = module.half()
        ref_pnorm_after_step = self._get_pnorm_after_step(module)
        module = self._get_shared_params_transformer().to(self.rank)  # recreate
        if half:
            module = module.half()
        fsdp_module = FSDP(module)
        fsdp_pnorm_after_step = self._get_pnorm_after_step(fsdp_module)
        self.assertEqual(ref_pnorm_after_step, fsdp_pnorm_after_step)

    def _get_pnorm_after_step(self, module):
        optim = torch.optim.SGD(module.parameters(), lr=0.01)
        loss = self._get_output(module).sum()
        loss.backward()
        optim.step()
        return torch.norm(torch.stack([p.detach().norm() for p in module.parameters()]))

    def test_flat_param_shard_metadata_unaligned(self):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected
        without any explicit alignment padding.
        """
        module = torch.nn.Sequential(
            torch.nn.Linear(10, 10, bias=False),
            nn.ReLU(),
            torch.nn.Linear(10, 10, bias=False),
            nn.ReLU(),
            torch.nn.Linear(10, 10, bias=False),
            nn.ReLU(),
        )
        params_to_flatten = list(module.parameters())
        handle = FlatParamHandle(
            params_to_flatten,
            module,
            **self._get_default_config(),
        )

        self._test_flat_param_shard_metadata(
            handle,
            start=0,
            end=0,
            expected=FlatParamShardMetadata(
                param_names=["0.weight"],
                param_shapes=[(10, 10)],
                param_strides=[(10, 1)],
                param_contiguities=[True],
                param_numels=[100],
                param_offsets=[(0, 0)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=0,
            end=50,
            expected=FlatParamShardMetadata(
                param_names=["0.weight"],
                param_shapes=[(10, 10)],
                param_strides=[(10, 1)],
                param_contiguities=[True],
                param_numels=[100],
                param_offsets=[(0, 50)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=0,
            end=99,
            expected=FlatParamShardMetadata(
                param_names=["0.weight"],
                param_shapes=[(10, 10)],
                param_strides=[(10, 1)],
                param_contiguities=[True],
                param_numels=[100],
                param_offsets=[(0, 99)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=50,
            end=149,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "2.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_strides=[(10, 1), (10, 1)],
                param_contiguities=[True, True],
                param_numels=[100, 100],
                param_offsets=[(50, 99), (0, 49)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=50,
            end=199,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "2.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_strides=[(10, 1), (10, 1)],
                param_contiguities=[True, True],
                param_numels=[100, 100],
                param_offsets=[(50, 99), (0, 99)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=99,
            end=199,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "2.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_strides=[(10, 1), (10, 1)],
                param_contiguities=[True, True],
                param_numels=[100, 100],
                param_offsets=[(99, 99), (0, 99)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=100,
            end=199,
            expected=FlatParamShardMetadata(
                param_names=["2.weight"],
                param_shapes=[(10, 10)],
                param_strides=[(10, 1)],
                param_contiguities=[True],
                param_numels=[100],
                param_offsets=[(0, 99)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=100,
            end=299,
            expected=FlatParamShardMetadata(
                param_names=["2.weight", "4.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_strides=[(10, 1), (10, 1)],
                param_contiguities=[True, True],
                param_numels=[100, 100],
                param_offsets=[(0, 99), (0, 99)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=100,
            end=1000,
            expected=FlatParamShardMetadata(
                param_names=["2.weight", "4.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_strides=[(10, 1), (10, 1)],
                param_contiguities=[True, True],
                param_numels=[100, 100],
                param_offsets=[(0, 99), (0, 99)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            start=299,
            end=299,
            expected=FlatParamShardMetadata(
                param_names=["4.weight"],
                param_shapes=[(10, 10)],
                param_strides=[(10, 1)],
                param_contiguities=[True],
                param_numels=[100],
                param_offsets=[(99, 99)],
            ),
        )

    def test_flat_param_shard_metadata_aligned_full_precision(self):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected
        with alignment padding and parameter full precision.
        """
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 7, bias=False),  # 0.weight
            torch.nn.Linear(7, 5, bias=False),  # 1.weight
            torch.nn.Linear(5, 5, bias=False),  # 2.weight
        )
        params_to_flatten = list(module.parameters())
        handle_kwargs = self._get_default_config()
        handle_kwargs["use_orig_params"] = True
        handle = FlatParamHandle(params_to_flatten, module, **handle_kwargs)
        # For 32-bit full precision, FSDP pads up to 3 numel after each
        # original parameter to achieve 0 mod 4 numel (i.e. 0 mod 16 bytes).
        # Thus, the unsharded `FlatParameter` layout looks like:
        #   21 + (3) + 35 + (1) + 25
        # where (x) means x numel of padding. This gives a total of 85 numel.

        # The `FlatParamShardMetadata` do not include alignment padding but do
        # account for them
        self._test_flat_param_shard_metadata(
            handle,
            # Emulate rank 0 of 2 ranks
            start=0,
            end=42,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "1.weight"],
                param_shapes=[(7, 3), (5, 7)],
                param_strides=[(3, 1), (7, 1)],
                param_contiguities=[True, True],
                param_numels=[21, 35],
                # 21 + (3) + 19 = 43
                param_offsets=[(0, 20), (0, 18)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            # Emulate rank 1 of 2 ranks
            start=43,
            end=85,
            expected=FlatParamShardMetadata(
                param_names=["1.weight", "2.weight"],
                param_shapes=[(5, 7), (5, 5)],
                param_strides=[(7, 1), (5, 1)],
                param_contiguities=[True, True],
                param_numels=[35, 25],
                # 16 + (1) + 25 = 42
                param_offsets=[(19, 34), (0, 24)],
            ),
        )

    def test_flat_param_shard_metadata_aligned_mixed_precision(self):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected
        with alignment padding and parameter mixed precision.
        """
        module = torch.nn.Sequential(
            torch.nn.Linear(2, 5, bias=False),  # 0.weight
            torch.nn.Linear(5, 5, bias=False),  # 1.weight
            torch.nn.Linear(5, 3, bias=False),  # 2.weight
        )
        params_to_flatten = list(module.parameters())
        handle_kwargs = self._get_default_config()
        handle_kwargs["use_orig_params"] = True
        handle_kwargs["mp_param_dtype"] = torch.float16
        handle = FlatParamHandle(params_to_flatten, module, **handle_kwargs)
        # For 16-bit mixed precision, FSDP pads up to 7 numel after each
        # original parameter to achieve 0 mod 8 numel (i.e. 0 mod 16 bytes).
        # Thus, the unsharded `FlatParameter` layout looks like:
        #   10 + (6) + 25 + (7) + 15
        # where (x) means x numel of padding. This gives a total of 63 numel.

        # The `FlatParamShardMetadata` do not include alignment padding but do
        # account for them
        self._test_flat_param_shard_metadata(
            handle,
            # Emulate rank 0 of 2 ranks
            start=0,
            end=31,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "1.weight"],
                param_shapes=[(5, 2), (5, 5)],
                param_strides=[(2, 1), (5, 1)],
                param_contiguities=[True, True],
                param_numels=[10, 25],
                # 10 + (6) + 16 = 32
                param_offsets=[(0, 9), (0, 15)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            # Emulate rank 1 of 2 ranks
            start=32,
            end=63,
            expected=FlatParamShardMetadata(
                param_names=["1.weight", "2.weight"],
                param_shapes=[(5, 5), (3, 5)],
                param_strides=[(5, 1), (5, 1)],
                param_contiguities=[True, True],
                param_numels=[25, 15],
                # 9 + (7) + 15 = 31
                param_offsets=[(16, 24), (0, 14)],
            ),
        )

    def _test_flat_param_shard_metadata(
        self,
        handle: FlatParamHandle,
        start: int,
        end: int,
        expected: FlatParamShardMetadata,
    ):
        """
        Tests the subroutine ``_get_shard_metadata()`` that computes shard
        metadata based on start and end indices in the unsharded flat
        parameter, where both indices are inclusive.

        We manually set the relevant attributes on the flat parameter to be
        able to check the effect of ``_get_shard_metadata()`` via
        ``shard_metadata()`` since normally the attributes are set in
        ``_init_shard_metadata()`` with the start and end indices fixed based
        on rank and world size.
        """
        flat_param = handle.flat_param
        flat_param._shard_param_infos = handle._get_shard_metadata(start, end)
        shard_metadata = handle.shard_metadata()
        self.assertEqual(
            shard_metadata,
            expected,
            msg=f"{handle.shard_metadata()}, {expected}",
        )

    @parametrize("memory_format", [torch.contiguous_format, torch.channels_last])
    def test_flat_param_shard_metadata_with_memory_format(self, memory_format):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected
        with alignment padding and parameter full precision.
        """
        module = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, 3, bias=False),  # 0.weight, 1800 params
            torch.nn.Conv2d(20, 10, 5, bias=False),  # 1.weight, 5000 params
            torch.nn.Conv2d(10, 10, 1, bias=False),  # 2.weight, 100 params
        ).to(memory_format=memory_format)
        params_to_flatten = list(module.parameters())
        handle_kwargs = self._get_default_config()
        handle_kwargs["use_orig_params"] = True
        handle = FlatParamHandle(params_to_flatten, module, **handle_kwargs)
        contiguous_tensors = memory_format == torch.contiguous_format
        self._test_flat_param_shard_metadata(
            handle,
            # Emulate rank 0 of 2 ranks
            start=0,
            end=2999,
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "1.weight"],
                param_shapes=[(20, 10, 3, 3), (10, 20, 5, 5)],
                param_strides=[(90, 9, 3, 1), (500, 25, 5, 1)]
                if contiguous_tensors
                else [(90, 1, 30, 10), (500, 1, 100, 20)],
                param_contiguities=[contiguous_tensors, contiguous_tensors],
                param_numels=[1800, 5000],
                param_offsets=[(0, 1799), (0, 1199)],
            ),
        )
        self._test_flat_param_shard_metadata(
            handle,
            # Emulate rank 1 of 2 ranks
            start=3000,
            end=6899,
            expected=FlatParamShardMetadata(
                param_names=["1.weight", "2.weight"],
                param_shapes=[(10, 20, 5, 5), (10, 10, 1, 1)],
                param_strides=[(500, 25, 5, 1), (10, 1, 1, 1)]
                if contiguous_tensors
                else [(500, 1, 100, 20), (10, 1, 10, 10)],
                param_contiguities=[contiguous_tensors, contiguous_tensors],
                param_numels=[5000, 100],
                param_offsets=[(1200, 4999), (0, 99)],
            ),
        )

    @skip_if_lt_x_gpu(1)
    def test_writeback_orig_params_no_shard(self):
        class EmbeddingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(5, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.emb(x).sum()

        model = EmbeddingModel().half().to(self.rank)
        fsdp_model = FSDP(
            model,
            sharding_strategy=HandleShardingStrategy.NO_SHARD,
            use_orig_params=True,
        )

        # Copied from https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py#L1679-1719
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            if not fsdp_module._has_params:
                continue
            param = fsdp_module._flat_param
            param.data = param.data.float()
            fsdp_module._handle._orig_param_dtype = torch.float32

        x = torch.randint(0, 5, (20,), device=self.rank)
        with torch.no_grad():
            out = fsdp_model(x)
        self.assertEqual(out.shape, torch.Size([]))


instantiate_parametrized_tests(TestFlattenParams)

if __name__ == "__main__":
    run_tests()
