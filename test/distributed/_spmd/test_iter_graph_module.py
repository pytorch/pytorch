# Owner(s): ["oncall: distributed"]

import torch.nn as nn
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Sequential(
            nn.Linear(20, 20),
            nn.Softmax(),
        )

    def forward(self, input):
        return self.ln(input)


class NestedBoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(20, 20)
        self.ln2 = nn.Linear(20, 20)
        self.inner = BoringModel()

    def forward(self, input):
        return self.inner(self.ln2(self.ln1(input)))


class IterGraphModuleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    def test_basic_movement(self) -> None:
        return
        # TODO: the following UT is broken after 4/1/2023.
        # Since the UT is still using the legacy way to trace and expand the
        # graph, it does not worth to fix it. Will migrate the UT to the latest
        # torch.distributed._spmd.compile in the next few PRs (after compile
        # supports graph optimization)
        """
        class FakeOptimization:
            def __init__(self) -> None:
                self.all_reduce_counter = 0
                self.wait_counter = 0

            def fake_all_reduce(self, gradients: List[torch.Tensor]) -> torch.Tensor:
                self.all_reduce_counter += 1
                return torch.concat(gradients)

            def fake_wait(self, wait_tensor: torch.Tensor) -> torch.Tensor:
                self.wait_counter += 1
                return torch.clone(wait_tensor)

            def fake_step(self, step) -> None:
                return

            def fake_optimizer(self, gradient) -> None:
                return

            def fake_comm_schedule(self, gm: IterGraphModule, move: bool):
                step_placeholder = None
                for node in gm.graph.nodes:
                    if str(node.op) == "placeholder":
                        step_placeholder = node
                        break
                for node in gm.graph.nodes:
                    if node.name == "addmm_2":
                        break
                with gm.graph.inserting_after(node):
                    all_reduce_node = gm.graph.call_function(
                        self.fake_all_reduce, ([node],)
                    )
                with gm.graph.inserting_after(all_reduce_node):
                    wait_node = gm.graph.call_function(
                        self.fake_wait, (all_reduce_node,)
                    )
                with gm.graph.inserting_after(wait_node):
                    step_node = gm.graph.call_function(
                        self.fake_step, (step_placeholder,)
                    )
                with gm.graph.inserting_after(step_node):
                    optimizer_node = gm.graph.call_function(
                        self.fake_optimizer, (wait_node,)
                    )
                # mimic the real use case when tracing the whole graph
                optimizer_node.name = "_fused_adam_"
                step_node.name = "_foreach_add_"
                gm.graph.functionalize_optim()

                gm.graph.keep_unused_nodes()
                for target_node in gm.graph.nodes:
                    if target_node.name == "addmm_1":
                        break
                if move:
                    gm.graph.move_to_next_iter_before(
                        [all_reduce_node, wait_node, step_node, optimizer_node],
                        target_node,
                    )
                gm.graph.eliminate_dead_code()
                gm.recompile()
                gm.graph.defunctionalize_optim()

        def _compile_bwd(
            gm: fx.GraphModule, inps: List[torch.Tensor]
        ) -> fx.GraphModule:
            return make_boxed_func(gm)

        def _compile_fwd(
            optimization: FakeOptimization,
            num_iters: int,
            move: bool,
            gm: fx.GraphModule,
            inps: List[torch.Tensor],
        ) -> fx.GraphModule:
            igm = IterGraphModule(gm)
            igm.setup(num_iters)
            optimization.fake_comm_schedule(igm, move)
            return make_boxed_func(igm)

        num_iters = 5
        model = NestedBoringModel().to("cuda")
        model_wo_wrapped = copy.deepcopy(model)
        optim_wo_moved = FakeOptimization()
        model_wo_moved = aot_module(
            copy.deepcopy(model),
            partial(_compile_fwd, optim_wo_moved, num_iters, False),
            _compile_bwd,
        )
        optim_wi_moved = FakeOptimization()
        model_wi_moved = aot_module(
            copy.deepcopy(model),
            partial(_compile_fwd, optim_wi_moved, num_iters, True),
            _compile_bwd,
        )
        all_models = [model_wo_wrapped, model_wo_moved, model_wi_moved]

        for curr_iter in range(num_iters):
            input_ = torch.randn(128, 20, device="cuda")
            outputs = [model(input_) for model in all_models]

            # All the model outputs must be the same even if IterGraphModule is
            # applied and optimized.
            for output in outputs:
                self.assertEqual(output, outputs[0])

            if curr_iter == 0:
                self.assertEqual(optim_wo_moved.all_reduce_counter, 1)
                self.assertEqual(optim_wi_moved.all_reduce_counter, 0)
                self.assertEqual(optim_wo_moved.wait_counter, 1)
                self.assertEqual(optim_wi_moved.wait_counter, 0)
            elif curr_iter == num_iters - 1:
                self.assertEqual(optim_wo_moved.all_reduce_counter, num_iters)
                self.assertEqual(optim_wi_moved.all_reduce_counter, num_iters)
                self.assertEqual(optim_wo_moved.wait_counter, num_iters)
                self.assertEqual(optim_wi_moved.wait_counter, num_iters)
            else:
                self.assertEqual(optim_wo_moved.all_reduce_counter, curr_iter + 1)
                self.assertEqual(optim_wi_moved.all_reduce_counter, curr_iter)
                self.assertEqual(optim_wo_moved.wait_counter, curr_iter + 1)
                self.assertEqual(optim_wi_moved.wait_counter, curr_iter)
        """


if __name__ == "__main__":
    run_tests()
