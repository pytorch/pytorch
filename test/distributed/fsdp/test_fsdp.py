# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
from math import inf
import pickle
from typing import Dict
import unittest
from unittest import mock

import torch
from torch import nn
import torch.distributed

from torch.distributed._fsdp import FullyShardedDataParallel
from torch.testing._internal.common_fsdp import (
    DeviceAndTypeCheckModule,
    DummyProcessGroup,
    get_cycles_per_ms,
    objects_are_equal,
    TransformerWithSharedParams,
    NestedWrappedModule,
    CONFIG_OPTIONS,
    DistributedTest,
    spawn_and_init,
    MixtureOfExperts,
)

# How to use remote-pdb: https://gist.github.com/sshleifer/9d43351957179c13606e015b072927d4
# All helper functions called by spawn must be either @classmethod, @staticmethod

class TestMixedPrecision(DistributedTest):
    def test_all_fp32(self):
        self._spawn_test_case(
            {"mixed_precision": False},
            False,  # autocast enabled
            torch.float32,  # expected_input_dtype
            torch.float32,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_mixed_precision(self):
        self._spawn_test_case(
            {"mixed_precision": True},
            False,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float16,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
        )

    def test_mixed_precision_autocast(self):
        """If autocast enabled, loss should be fp32."""
        self._spawn_test_case(
            {"mixed_precision": True},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
        )

    def test_mixed_precision_autocast_buffer_type_fp32(self):
        """If autocast enabled, loss should be fp32."""
        self._spawn_test_case(
            {"mixed_precision": True, "buffer_dtype": torch.float32},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
            expected_buffer_type=torch.float32,
        )

    def test_mixed_precision_autocast_fp32_compute(self):
        self._spawn_test_case(
            {"mixed_precision": True, "compute_dtype": torch.float32},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float32,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
            expected_buffer_type=torch.float32,
        )

    def test_fp32_reduce_scatter(self):
        self._spawn_test_case(
            {"mixed_precision": True, "fp32_reduce_scatter": True},
            False,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float16,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
            expected_buffer_type=torch.float16,
        )

    def test_fp32_reduce_scatter_autocast(self):
        self._spawn_test_case(
            {"mixed_precision": True, "fp32_reduce_scatter": True},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def _spawn_test_case(
        self,
        cfg,
        autocast_enabled,
        in_dtype,
        p_dtype,
        loss_dtype,
        reduce_dtype,
        expected_buffer_type=None,
        world_size=2,
    ):
        """Call test_dtypes inside of torch.multiprocessing.spawn"""
        fn = functools.partial(
            self._test_dtypes,
            cfg,
            autocast_enabled,
            in_dtype,
            p_dtype,
            loss_dtype,
            reduce_dtype,
            expected_buffer_type=expected_buffer_type,
        )
        spawn_and_init(fn, world_sizes=[world_size])

    @staticmethod
    def _test_dtypes(
        cfg: Dict, autocast, in_dtype, p_dtype, loss_dtype, reduce_dtype, rank, group, expected_buffer_type=None
    ):
        # Patch torch.distributed.reduce_scatter to check the dtype of the reduction
        orig_reduce_scatter = torch.distributed.reduce_scatter

        model: nn.Module = DeviceAndTypeCheckModule(
            expected_input_dtype=in_dtype,
            expected_param_dtype=p_dtype,
            expected_loss_dtype=loss_dtype,
            expected_buffer_dtype=expected_buffer_type,
        )

        def _reduce_scatter(output, input_list, **kwargs):
            for tensor in input_list:
                model._check("reduce_scatter.dtype", tensor.dtype, expected=reduce_dtype)
            return orig_reduce_scatter(output, input_list, **kwargs)

        with mock.patch("torch.distributed.reduce_scatter", new=_reduce_scatter):
            model = FullyShardedDataParallel(model, group, **cfg).cuda()
            device = next(model.parameters()).device
            x = torch.rand(2, 5).to(device)
            with torch.cuda.amp.autocast(enabled=autocast):
                loss = model(x)
            loss.backward()


class TestComparisonToPyTorchDDP(DistributedTest):
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. FullyShardedDataParallel.
    """

    def test_nested_wrapped_model(self):
        for config in CONFIG_OPTIONS:
            test_fn = functools.partial(self._test_identical_outputs, NestedWrappedModule, config)
            spawn_and_init(test_fn)

    def test_nested_all_wrapped_model(self):
        for config in CONFIG_OPTIONS:
            model_fn = functools.partial(NestedWrappedModule, wrap_everything=True)
            test_fn = functools.partial(self._test_identical_outputs, model_fn, config)
            spawn_and_init(test_fn)

    # def test_nested_all_wrapped_model_checkpoint(self):
    #    for config in CONFIG_OPTIONS:
    #        model_fn = functools.partial(NestedWrappedModule, wrap_everything=True, checkpoint=True)
    #        test_fn = functools.partial(self._test_identical_outputs, model_fn, config)
    #        spawn_and_init(test_fn)

    def test_transformer_parameterized(self):
        # Test every combination of these options:
        for config in CONFIG_OPTIONS:
            spawn_and_init(functools.partial(self._test_identical_outputs, TransformerWithSharedParams, config))

    def test_cpu_offload_and_cpu_grads(self):
        # We don't test the False condition because that requires the optimizer to internally do
        # the device transfer and PyTorch optimizers don't support this.
        config = {"mixed_precision": True, "cpu_offload": True, "move_grads_to_cpu": True}
        test_fn = functools.partial(
            self._test_identical_outputs, TransformerWithSharedParams, config, use_cuda=False, lr=0.01
        )
        spawn_and_init(test_fn)

    def test_cpu_offload_and_cuda_grads_breaks(self):
        # If grads are on gpu, but model and optimizer are on cpu, backward breaks.
        config = {"mixed_precision": True, "cpu_offload": True, "move_grads_to_cpu": False}
        with self.assertRaises(Exception):  # RuntimeError inside spawn
            test_fn = functools.partial(
                self._test_identical_outputs, TransformerWithSharedParams, config, use_cuda=False
            )
            spawn_and_init(test_fn)

    def test_delayed_optim_step(self):
        # We use a model with a long CUDA delay right before the optimizer step.
        # This tests our streams logic, and that we don't start the FP32 -> FP16
        # transfer until after the optimization step completes.
        config = {"mixed_precision": True}
        model_fn = functools.partial(NestedWrappedModuleWithDelay, delay_after_loss_ms=250)
        test_fn = functools.partial(self._test_identical_outputs, model_fn, config)
        spawn_and_init(test_fn)

    def test_delayed_reduce_scatter(self):
        # We insert a delay in the torch.distributed.reduce_scatter op, so that
        # the post_backward_stream takes much longer than the backward pass.
        # This tests that we properly block at the end of the backward pass for
        # the reductions to finish.
        config = {"mixed_precision": True}
        model_fn = functools.partial(NestedWrappedModuleWithDelay, delay_before_reduction_ms=250)
        test_fn = functools.partial(self._test_identical_outputs, model_fn, config)
        spawn_and_init(test_fn)

    # @parameterized.expand([[{"checkpoint_act": False}], [{"checkpoint_act": True}]], name_func=rename_test)
    def test_mixture_of_experts(self):
        fsdp_config = {"mixed_precision": True}
        test_fn = functools.partial(
            self._test_identical_outputs,
            functools.partial(MixtureOfExperts),
            fsdp_config,
            # MixtureOfExperts implements custom reduce logic, so the reference
            # behavior should use that logic instead of PyTorch DDP.
            ref_ddp_fn=self._dummy_ddp_fn,
            norm_type=None,
        )
        spawn_and_init(test_fn)

    # @parameterized.expand([[{"checkpoint_act": False}], [{"checkpoint_act": True}]], name_func=rename_test)
    def test_mixture_of_experts_with_delay_before_free(self):
        fsdp_config = {"mixed_precision": True}
        test_fn = functools.partial(
            self._test_identical_outputs,
            functools.partial(MixtureOfExperts, delay_before_free_ms=250),
            fsdp_config,
            # MixtureOfExperts implements custom reduce logic, so the reference
            # behavior should use that logic instead of PyTorch DDP.
            ref_ddp_fn=self._dummy_ddp_fn,
            norm_type=None,
        )
        spawn_and_init(test_fn)

    def test_mixture_of_experts_grad_clip_breaks(self):
        config = {"mixed_precision": True}
        test_fn = functools.partial(
            self._test_identical_outputs, MixtureOfExperts, config, ref_ddp_fn=self._dummy_ddp_fn, norm_type=2,
        )
        with self.assertRaises(Exception):
            spawn_and_init(test_fn)

    @classmethod
    def _dummy_ddp_fn(cls, model, group):
        return DummyDDP(model)

    def test_clip_norm_transformer(self):
        config = {"mixed_precision": True}
        for norm_type in [1, inf]:
            test_fn = functools.partial(
                self._test_identical_outputs, TransformerWithSharedParams, config, norm_type=norm_type,
            )
        spawn_and_init(test_fn)


class TestParamInit(DistributedTest):
    def test_param_change_after_init(self):
        test_fn = functools.partial(self._test_param_change_after_init, config={"mixed_precision": True})
        spawn_and_init(test_fn)

    @classmethod
    def _test_param_change_after_init(cls, rank, group, config):
        # Establish reference behavior.
        model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        model.eval()  # no dropout for this test
        input = model.module.get_input(torch.device("cuda"))
        ref_output = model(*input)

        # Change the weights in place.
        model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        model.eval()  # no dropout for this test
        first_param = next(model.parameters())
        nn.init.normal_(first_param.data)
        new_output = model(*input)

        assert not objects_are_equal(ref_output, new_output), "new_output did not reflect change to param after init"


class TestSerialization(DistributedTest):
    def test_pickle(self):
        """Ensure that wrapped modules can be pickled/unpickled."""
        for mixed_precision, cpu_offload in [(False, False), (True, False), (True, True)]:
            config = {"mixed_precision": mixed_precision, "cpu_offload": cpu_offload}
            test_fn = functools.partial(self._test_pickle, config=config)
            spawn_and_init(test_fn, world_sizes=[2])

    def test_multiprocessing(self):
        """Ensure that wrapped modules can be sent via multiprocessing."""
        for mixed_precision, cpu_offload in [(False, False), (True, False), (True, True)]:
            config = {"mixed_precision": mixed_precision, "cpu_offload": cpu_offload}
            test_fn = functools.partial(self._test_multiprocessing, config=config)
            spawn_and_init(test_fn, world_sizes=[2])

    @classmethod
    def _test_pickle(cls, rank, group, config):
        model = cls._get_model(group, config)
        model = pickle.loads(pickle.dumps(model))
        if not config["cpu_offload"]:
            model = model.cuda()
        cls._one_step(model, group)

    @classmethod
    def _test_multiprocessing(cls, rank, group, config):
        mp = torch.multiprocessing.Pool(1)
        dummy_group = DummyProcessGroup(rank=group.rank(), size=group.size())
        model = mp.apply(cls._get_model, (dummy_group, config))
        if not config["cpu_offload"]:
            model = model.cuda()
        cls._one_step(model, group)

    @classmethod
    def _get_model(cls, group, config):
        with torch.no_grad():  # required for multiprocessing
            model = NestedWrappedModule(group, wrapper_config=config)
            return FullyShardedDataParallel(model, group, **config)

    @classmethod
    def _one_step(cls, model, group):
        # reset the process group (required after unpickling)
        for m in model.modules():
            if isinstance(m, FullyShardedDataParallel):
                m.process_group = group
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        input = model.module.get_input(torch.device("cuda"))
        output = model(*input)
        loss = model.module.get_loss(input, output)
        model.module.run_backward(loss)
        optim.step()


class TestHooks(DistributedTest):
    # Feel free to modify these tests as the implementation changes.
    # They aspire to make sure that backward hooks are registered and used

    def test_output_backward_hooks(self):
        for cuda_first in [True, False]:
            fn = functools.partial(self._test_output_backward_hooks, cuda_first=cuda_first)
            spawn_and_init(fn)

    def test_backward_hooks_after_save(self):
        fn = functools.partial(self._test_backward_hooks_after_save, cuda_first=False)
        spawn_and_init(fn)

    @classmethod
    def _test_backward_hooks_after_save(cls, rank, group, cuda_first=False):
        model = cls.get_wrapped_model(group, cuda_first=cuda_first)
        cls._train_for_several_steps(model, 2, model.mixed_precision)
        state_1 = model.local_state_dict()
        model.load_local_state_dict(state_1)
        cls._test_output_backward_hooks(rank, group, cuda_first=cuda_first, model=model)

    @classmethod
    def _test_output_backward_hooks(cls, rank, group, cuda_first=False, model=None):
        if model is None:
            model = cls.get_wrapped_model(group, cuda_first=cuda_first)
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optim.zero_grad()
        # Inputs always cuda regardless of move_grads_cpu, or model.device
        input = model.module.get_input(torch.device("cuda"))
        output = model(*input)
        assert len(output._backward_hooks) == 1  # this is pre-bwd hook
        loss = model.module.get_loss(input, output).cuda()
        loss.backward()
        assert len(output._backward_hooks) == 1  # It doesn't get removed
        optim.step()
        assert len(output._backward_hooks) == 1

    def test_register_functions_called(self):
        for cuda_first in [True, False]:
            fn = functools.partial(self._test_register_functions_called, cuda_first=cuda_first)
            spawn_and_init(fn)

    @classmethod
    def _test_register_functions_called(cls, rank, group, cuda_first=False):
        """Tests that _register_{pre|post}_backward_hooks called during forward."""
        model = cls.get_wrapped_model(group, cuda_first=cuda_first)
        input = model.module.get_input(torch.device("cuda"))
        model._register_post_backward_hooks = mock.MagicMock(return_value=None)
        model._register_pre_backward_hooks = mock.MagicMock(return_value=None)
        assert not model._register_post_backward_hooks.called
        assert not model._register_pre_backward_hooks.called
        model(*input)
        assert model._register_post_backward_hooks.called
        assert model._register_pre_backward_hooks.called


class TestNoGrad(DistributedTest):
    def test_transformer_parameterized(self):
        for config in CONFIG_OPTIONS:
            test_fn = functools.partial(self._test_transformer, config=config)
            spawn_and_init(test_fn)

    @classmethod
    def _test_transformer(cls, rank, group, config):
        autocast = config["mixed_precision"]

        # Train model for a step
        model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        cls._train_for_several_steps(model, 1, autocast)

        model.eval()  # no dropout for this test

        # Eval in standard mode (i.e., without no_grad)
        input = model.module.get_input(torch.device("cuda"))
        ref_output = model(*input)

        # Eval with no_grad and compare
        with torch.no_grad():
            no_grad_output = model(*input)

        assert objects_are_equal(ref_output, no_grad_output, raise_exception=True)


class TestModuleProperties(DistributedTest):
    def test_named_parameters(self):
        for config in [{"flatten_parameters": False}, {"flatten_parameters": True}]:
            test_fn = functools.partial(self._test_named_params, config=config)
            spawn_and_init(test_fn)

    @classmethod
    def _test_named_params(cls, rank, group, config):
        # Get the named parameters before wrapping.
        before_wrap_model = TransformerWithSharedParams(group)
        before_wrap_params = before_wrap_model.named_parameters()

        # Train the model for 1 step.
        model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        cls._train_for_several_steps(model, 1, autocast=False)

        # Get the named parameters after wrapping to compare.
        after_wrap_params = model.named_parameters()

        if not config["flatten_parameters"]:
            for before_nm, after_nm in zip(before_wrap_params, after_wrap_params):
                assert before_nm[0] == after_nm[0]
        else:
            named_params_flat = list(p for p in after_wrap_params)[0][0]
            assert "flat_param_0" in named_params_flat

        # Compare name and size under the `summon_full_params` context.
        with model.summon_full_params():
            after_wrap_params = model.named_parameters()

            for before_nm, after_nm_original in zip(before_wrap_params, after_wrap_params):
                assert before_nm[0] == after_nm_original[0]
                torch.testing.assert_allclose(before_nm[1].shape, after_nm_original[1].cpu().shape)


class DummyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ModuleWithDelay(nn.Module):
    def __init__(self, module, delay_after_loss_ms=0, delay_before_reduction_ms=0):
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms
        self.delay_before_reduction_ms = delay_before_reduction_ms
        self.module = module

    def get_input(self, device):
        return self.module.get_input(device)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed.reduce_scatter

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(int(self.delay_before_reduction_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)

        with mock.patch("torch.distributed.reduce_scatter", _delayed_reduce_scatter):
            self.module.run_backward(loss)


class NestedWrappedModuleWithDelay(ModuleWithDelay):
    def __init__(self, group, wrapper_config, **kwargs):
        super().__init__(NestedWrappedModule(group, wrapper_config), **kwargs)


if __name__ == "__main__":
    unittest.main()
