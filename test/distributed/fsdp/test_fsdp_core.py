# Owner(s): ["oncall: distributed"]

import functools
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    DummyDDP,
    FSDPTest,
    MixtureOfExperts,
    NestedWrappedModule,
    NestedWrappedModuleWithDelay,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class TestParityWithDDP(FSDPTest):
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. FullyShardedDataParallel.
    """

    @skip_if_lt_x_gpu(2)
    def test_nested_wrapped_model(self):
        self._test_identical_outputs(NestedWrappedModule)

    @skip_if_lt_x_gpu(2)
    def test_nested_all_wrapped_model(self):
        model_fn = functools.partial(NestedWrappedModule, wrap_everything=True)
        self._test_identical_outputs(model_fn)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parameterized(self):
        self._test_identical_outputs(TransformerWithSharedParams)

    @skip_if_lt_x_gpu(2)
    def test_delayed_optim_step(self):
        # We use a model with a long CUDA delay right before the optimizer step.
        # This tests our streams logic, and that we don't start the allgather
        # until after the optimization step completes.
        model_fn = functools.partial(
            NestedWrappedModuleWithDelay, delay_after_loss_ms=250
        )
        self._test_identical_outputs(model_fn)

    @skip_if_lt_x_gpu(2)
    def test_delayed_reduce_scatter(self):
        # We insert a delay in the torch.distributed._reduce_scatter_base op, so that
        # the post_backward_stream takes much longer than the backward pass.
        # This tests that we properly block at the end of the backward pass for
        # the reductions to finish.
        model_fn = functools.partial(
            NestedWrappedModuleWithDelay, delay_before_reduction_ms=250
        )
        self._test_identical_outputs(model_fn)

    def _dummy_ddp_fn(self, model):
        return DummyDDP(model)

    @skip_if_lt_x_gpu(2)
    def test_mixture_of_experts(self):
        self._test_identical_outputs(
            MixtureOfExperts,
            # MixtureOfExperts implements custom reduce logic, so the reference
            # behavior should use that logic instead of PyTorch DDP.
            ref_ddp_fn=self._dummy_ddp_fn,
        )

    @skip_if_lt_x_gpu(2)
    def test_mixture_of_experts_with_delay_before_free(self):
        model_fn = functools.partial(MixtureOfExperts, delay_before_free_ms=250)
        self._test_identical_outputs(
            model_fn,
            ref_ddp_fn=self._dummy_ddp_fn,
        )


class TestParamInit(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_param_change_after_init(self):
        group = dist.distributed_c10d._get_default_group()
        # Establish reference behavior.
        model = self._get_wrapped_model(group, cuda_first=False)
        model.eval()  # no dropout for this test
        input = model.module.get_input(torch.device("cuda"))
        ref_output = model(*input)

        # Change the weights in place.
        model = self._get_wrapped_model(group, cuda_first=False)
        model.eval()  # no dropout for this test
        first_param = next(model.parameters())
        nn.init.normal_(first_param.data)
        new_output = model(*input)

        self.assertNotEqual(
            ref_output,
            new_output,
            msg="new_output did not reflect change to param after init",
        )


class TestHooks(FSDPTest):
    # They aspire to make sure that backward hooks are registered and used
    @skip_if_lt_x_gpu(2)
    @parametrize("cuda_first", [False, True])
    def test_output_backward_hooks(self, cuda_first):
        group = dist.distributed_c10d._get_default_group()
        model = self._get_wrapped_model(group, cuda_first=cuda_first)
        self._test_output_backward_hooks(model=model)

    @skip_if_lt_x_gpu(2)
    def test_backward_hooks_after_save(self):
        group = dist.distributed_c10d._get_default_group()
        model = self._get_wrapped_model(group, cuda_first=False)
        self._train_for_several_steps(model, num_steps=2, autocast=False)
        state_1 = model.state_dict()
        model.load_state_dict(state_1)
        self._test_output_backward_hooks(model=model)

    def _test_output_backward_hooks(self, model):
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optim.zero_grad()
        # Inputs always cuda, as computation happes on CUDA device only
        input = model.module.get_input(torch.device("cuda"))
        output = model(*input)
        # this is pre-bwd hook
        self.assertEqual(len(output._backward_hooks), 1)
        loss = model.module.get_loss(input, output).cuda()
        loss.backward()
        # It doesn't get removed
        self.assertEqual(len(output._backward_hooks), 1)
        optim.step()
        self.assertEqual(len(output._backward_hooks), 1)

    @skip_if_lt_x_gpu(2)
    @parametrize("cuda_first", [False, True])
    def test_register_functions_called(self, cuda_first):
        """Tests that _register_{pre|post}_backward_hooks called during forward."""
        group = dist.distributed_c10d._get_default_group()
        model = self._get_wrapped_model(group, cuda_first=cuda_first)
        input = model.module.get_input(torch.device("cuda"))
        model._register_post_backward_hooks = mock.MagicMock(return_value=None)
        model._register_pre_backward_hooks = mock.MagicMock(return_value=None)
        self.assertFalse(model._register_post_backward_hooks.called)
        self.assertFalse(model._register_pre_backward_hooks.called)
        model(*input)
        self.assertTrue(model._register_post_backward_hooks.called)
        self.assertTrue(model._register_pre_backward_hooks.called)


class TestNoGrad(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_transformer_no_grad(self):
        group = dist.distributed_c10d._get_default_group()
        model = self._get_wrapped_model(group, cuda_first=False)
        # Train model for a step
        self._train_for_several_steps(model, num_steps=1, autocast=False)

        model.eval()  # no dropout for this test

        # Eval in standard mode (i.e., without no_grad)
        input = model.module.get_input(torch.device("cuda"))
        ref_output = model(*input)

        # Eval with no_grad and compare
        with torch.no_grad():
            no_grad_output = model(*input)

        self.assertEqual(ref_output, no_grad_output)


instantiate_parametrized_tests(TestHooks)

if __name__ == "__main__":
    run_tests()
