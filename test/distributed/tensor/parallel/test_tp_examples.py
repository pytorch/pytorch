# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    NUM_DEVICES,
    Transformer,
    with_comms,
)


class DistTensorParallelExampleTest(DTensorTestBase):
    def _check_module(self, m1, m2, check_grad=False):
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            self.assertEqual(param_m2, param_m1)

    def _all_reduce_grad(self, m):
        for param in m.parameters():
            dist.all_reduce(param.grad)

    def _test_mlp_training_e2e(self, is_seq_parallel=False, recompute_activation=False):
        inp_size = [8, 10]
        # Ensure all tp ranks have same input.
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self._check_module(model, model_tp)

        # Shard module and initialize optimizer.
        LR = 0.25
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel(),
            "net2": RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel(),
        }
        model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)
        if recompute_activation:
            model_tp = input_reshard(
                checkpoint_wrapper(
                    model_tp, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                device_mesh,
                None if is_seq_parallel else 0,
            )
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()

        if is_seq_parallel:
            # Sum gradients from different ranks, since input
            # are different across ranks for sequence parallel.
            self._all_reduce_grad(model)

        # Ensure gradients are same.
        self._check_module(model, model_tp, check_grad=True)

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        # Due to the trick we use for Partial aggregation, we only check the weight when local_rank = 0.
        self._check_module(model, model_tp)

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    def _test_mlp_inference(self, device_mesh):
        inp_size = [8, 10]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self._check_module(model, model_tp)

        # Shard module and initialize optimizer.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    def _test_transformer_training_e2e(self, is_seq_parallel=False):
        # Model initialization.
        n_layers, vocab_size, dim, n_heads = 1, 26, 16, 4  # arguments for transformer
        model = Transformer(n_layers, vocab_size, dim, n_heads, 1).to(self.device_type)
        model_tp = Transformer(n_layers, vocab_size, dim, n_heads, self.world_size).to(self.device_type)

        # Ensure model are initialized the same way.
        self._check_module(model, model_tp)

        # Set up the parallelize plan.
        parallelize_plan = {}
        # Parallelize the embedding submodule.
        parallelize_plan["tok_embeddings"] = ColwiseParallel(
            input_layouts=Shard(0),
            output_layouts=Shard(0),
        ) if is_seq_parallel else ColwiseParallel(output_layouts=Replicate())
        # Parallelize attention and feed forward submodules.
        for i in range(n_layers):
            if is_seq_parallel:
                parallelize_plan[f"layers.{i}.attention"] = PrepareModuleInput(
                    input_layouts=[Shard(0)], desired_input_layouts=[Replicate()]
                )
            parallelize_plan[f"layers.{i}.attention.wq"] = ColwiseParallel()
            parallelize_plan[f"layers.{i}.attention.wk"] = ColwiseParallel()
            parallelize_plan[f"layers.{i}.attention.wv"] = ColwiseParallel()
            parallelize_plan[f"layers.{i}.attention.wo"] = RowwiseParallel(
                output_layouts=Shard(0)
            ) if is_seq_parallel else RowwiseParallel()

            parallelize_plan[f"layers.{i}.feed_forward.w1"] = ColwiseParallel(
                input_layouts=Shard(0)
            ) if is_seq_parallel else ColwiseParallel()
            parallelize_plan[f"layers.{i}.feed_forward.w2"] = RowwiseParallel(
                output_layouts=Shard(0)
            ) if is_seq_parallel else RowwiseParallel()
        # Parallelize the output submodule.
        parallelize_plan["output"] = ColwiseParallel(
            input_layouts=Shard(0),
            output_layouts=Shard(0),
        ) if is_seq_parallel else ColwiseParallel(output_layouts=Replicate())

        device_mesh = DeviceMesh(self.device_type, torch.arange(0, NUM_DEVICES))
        model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)

        # Initialize optimizer.
        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        # Initialize input.
        inp_size = [8, 10]  # [batch_size, seq_len]
        # Ensure all tp ranks have same input.
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.randint(vocab_size, inp_size, device=self.device_type)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()

        if is_seq_parallel:
            # Sum gradients from different ranks, since input
            # are different across ranks for sequence parallel.
            self._all_reduce_grad(model)

        # Ensure gradients are same.
        self._check_module(model, model_tp, check_grad=True)

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        # Due to the trick we use for Partial aggregation, we only check the weight when local_rank = 0.
        self._check_module(model, model_tp)

        # TODO: attention, layernorm when is_seq_parallel=True

        # TODO: the following 2nd input fail with small deviations -- need to investigate
        # inp = torch.randint(vocab_size, inp_size, device=self.device_type)
        # output = model(inp)
        # output_tp = model_tp(inp)
        # self.assertEqual(output, output_tp)

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    # TODO: need to revisit input_reshard API about why it failed multi-gpu tests.
    # @parametrize("recompute_activation", [True, False])
    @parametrize("recompute_activation", [False])
    def test_mlp_training(self, is_seq_parallel, recompute_activation):
        self._test_mlp_training_e2e(
            is_seq_parallel=is_seq_parallel, recompute_activation=recompute_activation
        )

    @with_comms
    def test_mlp_inference(self):
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        with torch.inference_mode():
            self._test_mlp_inference(device_mesh)

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    def test_transformer_training(self, is_seq_parallel):
        self._test_transformer_training_e2e(is_seq_parallel=is_seq_parallel)

instantiate_parametrized_tests(DistTensorParallelExampleTest)

if __name__ == "__main__":
    run_tests()
