# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.flat_param import (
    FlatParamHandle,
    FlatParamShardMetadata,
    HandleConfig,
    HandleShardingStrategy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

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
        return HandleConfig(HandleShardingStrategy.FULL_SHARD, False, None, None)

    def _get_transformer(self, seed=0):
        torch.manual_seed(seed)  # keep everything deterministic
        module = torch.nn.Transformer(
            d_model=32,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )
        module.register_buffer("dummy_buffer", torch.tensor(1.0))

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
            "Cannot initialize a `FlatParameter` from an empty parameter list",
        ):
            FlatParamHandle(
                [],
                module,
                torch.device("cuda"),
                self._get_default_config(),
                self.process_group,
                False,
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
            torch.device("cuda"),
            self._get_default_config(),
            self.process_group,
            False,
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

    def test_flat_param_shard_metadata(self):
        """
        Tests that ``FlatParameter`` shard metadata are computed as expected.
        """
        module = torch.nn.Sequential(
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
        )
        params_to_flatten = list(module.parameters())
        flat_param_handle = FlatParamHandle(
            params_to_flatten,
            module,
            torch.device("cuda"),
            self._get_default_config(),
            self.process_group,
            False,
        )

        def _test(kwargs, expected):
            """
            Tests the subroutine ``_get_shard_metadata()`` that computes shard
            metadata based on start and end indices in the unsharded flattened
            parameter.

            We manually set the relevant attributes on the flattened parameter
            to be able to check the effect of ``_get_shard_metadata()`` via
            ``shard_metadata()`` since normally the attributes are set in
            ``init_shard_info()`` with the start and end indices fixed based on
            rank and world size.
            """
            flat_param = flat_param_handle.flat_param
            (
                flat_param._shard_param_offsets,
                flat_param._shard_indices,
            ) = flat_param_handle._get_shard_metadata(kwargs["start"], kwargs["end"])
            self.assertEqual(
                flat_param_handle.shard_metadata(),
                expected,
                msg=f"{flat_param_handle.shard_metadata()}, {expected}",
            )

        _test(
            kwargs={"start": 0, "end": 0},
            expected=FlatParamShardMetadata(
                param_names=["0.weight"],
                param_shapes=[(10, 10)],
                param_numels=[100],
                param_offsets=[(0, 0)],
            ),
        )
        _test(
            kwargs={"start": 0, "end": 50},
            expected=FlatParamShardMetadata(
                param_names=["0.weight"],
                param_shapes=[(10, 10)],
                param_numels=[100],
                param_offsets=[(0, 50)],
            ),
        )
        _test(
            kwargs={"start": 0, "end": 99},
            expected=FlatParamShardMetadata(
                param_names=["0.weight"],
                param_shapes=[(10, 10)],
                param_numels=[100],
                param_offsets=[(0, 99)],
            ),
        )
        _test(
            kwargs={"start": 50, "end": 149},
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "2.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_numels=[100, 100],
                param_offsets=[(50, 99), (0, 49)],
            ),
        )
        _test(
            kwargs={"start": 50, "end": 199},
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "2.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_numels=[100, 100],
                param_offsets=[(50, 99), (0, 99)],
            ),
        )
        _test(
            kwargs={"start": 99, "end": 199},
            expected=FlatParamShardMetadata(
                param_names=["0.weight", "2.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_numels=[100, 100],
                param_offsets=[(99, 99), (0, 99)],
            ),
        )
        _test(
            kwargs={"start": 100, "end": 199},
            expected=FlatParamShardMetadata(
                param_names=["2.weight"],
                param_shapes=[(10, 10)],
                param_numels=[100],
                param_offsets=[(0, 99)],
            ),
        )
        _test(
            kwargs={"start": 100, "end": 299},
            expected=FlatParamShardMetadata(
                param_names=["2.weight", "4.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_numels=[100, 100],
                param_offsets=[(0, 99), (0, 99)],
            ),
        )
        _test(
            kwargs={"start": 100, "end": 1000},
            expected=FlatParamShardMetadata(
                param_names=["2.weight", "4.weight"],
                param_shapes=[(10, 10), (10, 10)],
                param_numels=[100, 100],
                param_offsets=[(0, 99), (0, 99)],
            ),
        )
        _test(
            kwargs={"start": 299, "end": 299},
            expected=FlatParamShardMetadata(
                param_names=["4.weight"],
                param_shapes=[(10, 10)],
                param_numels=[100],
                param_offsets=[(99, 99)],
            ),
        )


if __name__ == "__main__":
    run_tests()
