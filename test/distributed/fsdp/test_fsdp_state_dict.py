# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import unittest
import pytest

import torch
from torch import nn

from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_fsdp import (
    dist_init,
    objects_are_equal,
    teardown,
    temp_files_ctx,
    CONFIG_OPTIONS,
    DistributedTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
    spawn_and_init,
)


class TestLocalStateDict(DistributedTest):
    def test_load_local_state_dict(self):
        for flatten_params, mixed_precision in [[True, True], [False, False]]:
            test_fn = functools.partial(
                self._load_local_and_train, {"flatten_parameters": flatten_params, "mixed_precision": mixed_precision}
            )
            spawn_and_init(test_fn)

    @classmethod
    def _load_local_and_train(cls, config, rank, group, d_model=16, d_vocab=23):
        """Check that local_state_dict can be saved and loaded for a given worker, and that training updates it"""
        model = cls.get_wrapped_model(
            group, cuda_first=False, config=config, d_vocab=d_vocab, d_model=d_model, add_bn=False
        )  # Set bn=True here to show that BN doesn't get updated
        state_1 = model.local_state_dict()
        state_before_training = {k: v.cpu().clone() for k, v in state_1.items()}
        assert len(state_1) > 0
        model.load_local_state_dict(state_1)
        weight_key = "flat_param_0" if model.flatten_parameters else "embed_tokens.weight"

        state_1_weight = state_1[weight_key]
        assert state_1_weight.dtype == torch.float32, f"got dtype {state_1_weight.dtype} expected torch.float32"
        if not model.flatten_parameters:
            # The weight will be sharded since we access module.state_dict directly
            state_1_module_weight = model.module.state_dict()[weight_key]
            torch.testing.assert_allclose(state_1_weight, state_1_module_weight)
            torch.testing.assert_allclose(state_1_weight, model.module.embed_tokens.weight)
        cls._train_for_several_steps(model, 1, model.mixed_precision)

        state_2 = model.local_state_dict()
        state_after_training = {k: v.cpu().clone() for k, v in state_2.items()}
        model.load_local_state_dict(state_2)

        assert state_1.keys() == state_2.keys()

        # Assert that parameters were updated since before training
        unchanged = []
        unwrapped_model = model.module.module
        buffers = {name for name, _ in unwrapped_model.named_buffers()}
        for k in state_1:
            if (state_before_training[k] == state_after_training[k]).all() and (k not in buffers):
                unchanged.append(k)
        if unchanged:
            raise AssertionError(f"params {unchanged} not changed after training")


class TestSaveLoadStateDict(DistributedTest):
    def test_calling_state_dict_twice_mixed_precision(self):
        for mixed_precision in [False, True]:
            test_fn = functools.partial(
                self._test_calling_state_dict_twice, {"flatten_parameters": False, "mixed_precision": mixed_precision}
            )
            spawn_and_init(test_fn)

    @classmethod
    def _test_calling_state_dict_twice(cls, config, rank, group, **model_kwargs):
        ddp_model = cls.get_wrapped_model(group, cuda_first=False, config=config, **model_kwargs)
        autocast = ddp_model.mixed_precision
        cls._train_for_several_steps(ddp_model, 1, autocast)
        ddp_model.state_dict()
        ddp_model.state_dict()  # second call

    def test_state_dict_after_forward(self):
        for config in CONFIG_OPTIONS:
            test_fn = functools.partial(self._test_module_state_dict, config)
            spawn_and_init(test_fn)

    def test_state_dict_before_forward(self):
        for mixed_precision in [False, True]:
            test_fn = functools.partial(
                self._test_state_dict_before_forward, {"flatten_parameters": False, "mixed_precision": mixed_precision}
            )
            spawn_and_init(test_fn)

    @classmethod
    def _test_state_dict_before_forward(cls, config, rank, group):
        ddp_model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        sd = ddp_model.state_dict()
        for param_name in ("embed_tokens.weight", "vocab_bias"):
            wt = sd[param_name]
            assert wt.dtype == torch.float32, f"got dtype {wt.dtype} for {param_name}, expected torch.float32"
        cls._train_for_several_steps(ddp_model, 1, ddp_model.mixed_precision)

    @classmethod
    def _test_module_state_dict(cls, config, rank, group):
        ddp_model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        autocast = ddp_model.mixed_precision
        cls._train_for_several_steps(ddp_model, 2, autocast)
        state_1 = ddp_model.state_dict()
        # You must make a new FSDP instance to use module.load_state_dict
        unwrapped_model = TransformerWithSharedParams(group)
        unwrapped_model.load_state_dict(state_1)
        new_ddp_model = FSDP(unwrapped_model, group, **config).cuda()
        cls._train_for_several_steps(new_ddp_model, 2, autocast)
        try:
            ddp_model.load_state_dict(new_ddp_model.state_dict())
            AssertionError("ddp_model.load_state_dict(new_ddp_model.state_dict()) succeeded")
        except Exception:
            pass

    def test_nested_wrapped_model(self):
        for config in CONFIG_OPTIONS:
            test_fn = functools.partial(self._test_nested_wrapped_model, config=config)
            spawn_and_init(test_fn)

    def test_nested_wrapped_model_local_state_dict(self):
        for config in CONFIG_OPTIONS:
            test_fn = functools.partial(self._test_nested_wrapped_model_local_state_dict, config=config)
            spawn_and_init(test_fn)

    @classmethod
    def _test_nested_wrapped_model(cls, rank, group, config=None):
        # Get reference state dict without any nested FSDP instances.
        model = NestedWrappedModule(group, None).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, process_group=group)
        cls._train_for_several_steps(model, 2, autocast=config["mixed_precision"])
        ref_state_dict = {k: v.clone() for k, v in model.module.state_dict().items()}

        # Create a nested FSDP-wrapped instance.
        if config["mixed_precision"]:
            config["compute_dtype"] = torch.float32
        model = NestedWrappedModule(group, config)
        model = FSDP(model, group, **config).cuda()
        cls._train_for_several_steps(model, 2, autocast=config["mixed_precision"])

        # Round-trip state dict save/load/save.
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state_dict)
        state_dict = model.state_dict()

        assert ref_state_dict.keys() == state_dict.keys(), f"{ref_state_dict.keys()} != {state_dict.keys()}"
        for key in ref_state_dict.keys():
            assert objects_are_equal(
                ref_state_dict[key], state_dict[key], raise_exception=False
            ), f"{key}, {ref_state_dict[key]} != {state_dict[key]}"

    @classmethod
    def _test_nested_wrapped_model_local_state_dict(cls, rank, group, config=None, local=None):
        # Create a nested FSDP-wrapped instance.
        model = NestedWrappedModule(group, config)
        model = FSDP(model, group, **config).cuda()
        cls._train_for_several_steps(model, 2, autocast=config["mixed_precision"])

        # Round trip state dict save/load/save.
        ref_state_dict = {k: v.clone() for k, v in model.local_state_dict().items()}
        model.load_local_state_dict(ref_state_dict)
        state_dict = model.local_state_dict()

        assert ref_state_dict.keys() == state_dict.keys(), f"{ref_state_dict.keys()} != {state_dict.keys()}"
        for key in ref_state_dict.keys():
            assert objects_are_equal(
                ref_state_dict[key], state_dict[key], raise_exception=False
            ), f"{key}, {ref_state_dict[key]} != {state_dict[key]}"


class TestStateDictDeviceDtype(DistributedTest):
    def test_state_dict_device(self):
        for mixed_precision, cpu_offload in [[False, False], [True, False], [True, True]]:
            test_fn = functools.partial(
                self._test_state_dict_device, {"cpu_offload": cpu_offload, "mixed_precision": mixed_precision}
            )
            spawn_and_init(test_fn)

    def test_state_dict_device_cuda(self):
        for mixed_precision, cpu_offload in [[False, False], [True, False], [True, True]]:
            test_fn = functools.partial(
                self._test_state_dict_device,
                {"cpu_offload": cpu_offload, "mixed_precision": mixed_precision, "state_dict_device": torch.device("cuda")},
            )
            spawn_and_init(test_fn)

    def test_state_dict_device_cpu(self):
        for mixed_precision, cpu_offload in [[False, False], [True, False], [True, True]]:
            test_fn = functools.partial(
                self._test_state_dict_device,
                {"cpu_offload": cpu_offload, "mixed_precision": mixed_precision, "state_dict_device": torch.device("cpu")},
            )
            spawn_and_init(test_fn)

    def test_state_dict_device_pure_fp16(self):
        test_fn = functools.partial(
            self._test_state_dict_device,
            {"cpu_offload": False, "mixed_precision": False, "compute_dtype": torch.float16},
            # pure_fp16 is similar to the --memory-efficient-fp16 option in fairseq
            pure_fp16=True,
        )
        spawn_and_init(test_fn)

    @classmethod
    def _test_state_dict_device(cls, config, rank, group, pure_fp16=False, **model_kwargs):
        model = TransformerWithSharedParams(group, **model_kwargs)
        if pure_fp16:
            assert not config["mixed_precision"]
            model = model.half()
        fsdp_model = FSDP(model, group, **config)
        if not config["cpu_offload"]:
            fsdp_model = fsdp_model.cuda()
        autocast = fsdp_model.mixed_precision or pure_fp16
        cls._train_for_several_steps(fsdp_model, 1, autocast)

        sd = fsdp_model.state_dict()

        sd_device = config.get("state_dict_device")
        for k, v in sd.items():
            if config["cpu_offload"] or (sd_device is not None and sd_device.type == "cpu"):
                assert v.device.type == "cpu", v.device.type
            else:
                assert v.device.type == "cuda", v.device.type

        expected_dtype = torch.float16 if pure_fp16 else torch.float32
        for k, v in sd.items():
            if not torch.is_floating_point(v):
                continue
            assert v.dtype == expected_dtype, f"{v.dtype} != {expected_dtype}"


@pytest.mark.skipif(torch.cuda.is_available(), reason="Testing only on CPUs to save time")
def test_local_state_dict_calls_state_dict_recursion():
    """Testing the case of infinite recursive when FSDP is subclassed"""

    class TestModule(FSDP):
        def __init__(self):
            super().__init__(module=nn.Linear(100, 100))

        def state_dict(self, *args, **kwargs):
            return self.local_state_dict(*args, **kwargs)

    rank = 0
    world_size = 1
    with temp_files_ctx(2) as temp_files:
        result = dist_init(rank, world_size, temp_files[0], temp_files[1])
        assert result, "Dist init failed"

        m = TestModule()
        d = m.state_dict()

        teardown()


if __name__ == "__main__":
    unittest.main()
