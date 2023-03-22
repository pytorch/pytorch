# Owner(s): ["oncall: distributed"]

import copy
from functools import partial
from typing import List

import torch
import torch.fx as fx
import torch.nn as nn
from torch._functorch.aot_autograd import aot_module, make_boxed_func
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Sequential(
            nn.Linear(20, 20),
            nn.Softmax(),
        )

    def forward(self, input):
        return self.ln(input)


class NestedBoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.Linear(20, 20)
        self.ln2 = torch.nn.Linear(20, 20)
        self.inner = BoringModel()

    def forward(self, input):
        return self.inner(self.ln2(self.ln1(input)))


class IterGraphModuleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    def test_basic_movement(self) -> None:
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

            def fake_comm_schedule(self, gm: IterGraphModule, move: bool):
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
                for target_node in gm.graph.nodes:
                    if target_node.name == "addmm_1":
                        break
                if move:
                    gm.graph.move_to_next_iter_before([wait_node], target_node)
                # Not calling eliminate_dead_code ensures that nodes won't be
                # removed from the graph.
                gm.graph.lint()
                gm.recompile()

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
                self.assertEqual(optim_wi_moved.all_reduce_counter, 1)
                self.assertEqual(optim_wo_moved.wait_counter, 1)
                self.assertEqual(optim_wi_moved.wait_counter, 0)
            elif curr_iter == num_iters - 1:
                self.assertEqual(optim_wo_moved.all_reduce_counter, num_iters)
                self.assertEqual(optim_wi_moved.all_reduce_counter, num_iters)
                self.assertEqual(optim_wo_moved.wait_counter, num_iters)
                self.assertEqual(optim_wi_moved.wait_counter, num_iters)
            else:
                self.assertEqual(optim_wo_moved.all_reduce_counter, curr_iter + 1)
                self.assertEqual(optim_wi_moved.all_reduce_counter, curr_iter + 1)
                self.assertEqual(optim_wo_moved.wait_counter, curr_iter + 1)
                self.assertEqual(optim_wi_moved.wait_counter, curr_iter)


if __name__ == "__main__":
    run_tests()
