"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 comm_mode_features_example.py -e MLP_operation_tracing
"""
import argparse
import os
from typing import Callable, Dict, Union

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    MLPModule,
    MLPStacked,
    ModelArgs,
    NUM_DEVICES,
    Transformer,
)
from torch.utils.checkpoint import checkpoint


def get_device_type() -> str:
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )


c10d_functional = torch.ops.c10d_functional

aten = torch.ops.aten
supported_ops = [aten.view.default, aten._to_copy.default]


class CommDebugModeExample:
    """
    Checks if the set of keys in ground truth dictionary and the set
    produced in advanced_module_tracker are in the same order
    """

    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        self.device_type = get_device_type()

    def _MLP_model_setup(
        self, model_type: type, parallelize_plan: Union[None, dict] = None
    ) -> tuple[nn.Module, torch.Tensor]:
        """
        Creates MLP or MLPStacked model for examples
        """

        if parallelize_plan is None:
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }

        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        inp_size = [8, 10]
        inp = torch.rand(*inp_size, device=self.device_type)

        model = model_type(self.device_type)
        model = parallelize_module(model, device_mesh, parallelize_plan)
        return model, inp

    def _transformer_model_setup(
        self, is_seq_parallel: bool = False
    ) -> tuple[nn.Module, torch.Tensor]:
        """
        Creates transformer model for examples
        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        model_args = ModelArgs()
        model = Transformer(model_args).to(device=self.device_type)
        model = Transformer.parallelize(model, device_mesh, is_seq_parallel)
        inp_size = [8, 8]

        inp = torch.randint(model_args.vocab_size, inp_size, device=self.device_type)

        return model, inp

    def example_MLP_distributed_sharding_display(self) -> None:
        """
        Example of obtaining all module's FQN and parameters for a given distributed model and printing the sharding info

        Expected output:
        MLPModule.net1.weight: (Shard(dim=0),)
        MLPModule.net1.bias: (Shard(dim=0),)
        MLPModule.net2.weight: (Shard(dim=1),)
        MLPModule.net2.bias: (Replicate(),)
        """

        torch.manual_seed(0)
        model, inp = self._MLP_model_setup(model_type=MLPModule)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        comm_mode.print_sharding_info()

    def example_MLPStacked_distributed_sharding_display(self) -> None:
        """
        Example of obtaining all module's FQN and parameters for a given
        distributed model with nested modules and printing the sharding info

        Expected output:
        MLPStacked.layers.0.net1.weight: (Shard(dim=0),)
        MLPStacked.layers.0.net1.bias: (Shard(dim=0),)
        MLPStacked.layers.0.net2.weight: (Shard(dim=1),)
        MLPStacked.layers.0.net2.bias: (Replicate(),)
        MLPStacked.layers.1.net1.weight: (Shard(dim=0),)
        MLPStacked.layers.1.net1.bias: (Shard(dim=0),)
        MLPStacked.layers.1.net2.weight: (Shard(dim=1),)
        MLPStacked.layers.1.net2.bias: (Replicate(),)
        """

        torch.manual_seed(0)

        parallelize_plan = {
            "MLPStacked.layers.0.net1": ColwiseParallel(),
            "MLPStacked.layers.0.net2": RowwiseParallel(),
            "MLPStacked.layers.1.net1": ColwiseParallel(),
            "MLPStacked.layers.1.net2": RowwiseParallel(),
        }

        model, inp = self._MLP_model_setup(
            model_type=MLPStacked, parallelize_plan=parallelize_plan
        )

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        comm_mode.print_sharding_info()

    def example_MLP_module_tracing(self) -> None:
        """
        Example code to demonstrate CommModeDebug's module level tracing using a MLP model.
        Prints a table of module level collective tracing information and logs table to comm_mode_log.txt

        Expected Output:
        Global
          FORWARD PASS
            *c10d_functional.all_reduce: 1
            MLPModule
              FORWARD PASS
                *c10d_functional.all_reduce: 1
                MLPModule.net1
                MLPModule.relu
                MLPModule.net2
                  FORWARD PASS
                    *c10d_functional.all_reduce: 1
        """

        torch.manual_seed(0)

        model, inp = self._MLP_model_setup(model_type=MLPModule)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        # print the module level collective tracing information
        print(comm_mode.generate_comm_debug_tracing_table(noise_level=0))
        comm_mode.log_comm_debug_tracing_table_to_file(noise_level=0)

    def example_transformer_module_tracing(self) -> None:
        """
        Example code to demonstrate CommModeDebug's module level tracing using a distributed Transformer model.
        Prints a table of module level collective tracing information and logs table to comm_mode_log.txt

        Expected output:
        Global
          FORWARD PASS
            *c10d_functional.all_reduce: 6
            *c10d_functional.all_gather_into_tensor: 1
            Transformer
              FORWARD PASS
                *c10d_functional.all_reduce: 6
                *c10d_functional.all_gather_into_tensor: 1
                Transformer.tok_embeddings
                  FORWARD PASS
                    *c10d_functional.all_reduce: 1
                Transformer.pos_embeddings
                  FORWARD PASS
                    *c10d_functional.all_reduce: 1
                Transformer.dropout
                Transformer.layers.0
                  FORWARD PASS
                    *c10d_functional.all_reduce: 2
                    Transformer.layers.0.attention_norm
                    Transformer.layers.0.attention
                      FORWARD PASS
                        *c10d_functional.all_reduce: 1
                        Transformer.layers.0.attention.wq
                        Transformer.layers.0.attention.wk
                        Transformer.layers.0.attention.wv
                        Transformer.layers.0.attention.wo
                          FORWARD PASS
                            *c10d_functional.all_reduce: 1
                        Transformer.layers.0.attention.resid_dropout
                    Transformer.layers.0.ffn_norm
                    Transformer.layers.0.feed_forward
                      FORWARD PASS
                        *c10d_functional.all_reduce: 1
                        Transformer.layers.0.feed_forward.w1
                        Transformer.layers.0.feed_forward.gelu
                        Transformer.layers.0.feed_forward.w2
                          FORWARD PASS
                            *c10d_functional.all_reduce: 1
                        Transformer.layers.0.feed_forward.resid_dropout
                Transformer.layers.1
                  FORWARD PASS
                    *c10d_functional.all_reduce: 2
                    Transformer.layers.1.attention_norm
                    Transformer.layers.1.attention
                      FORWARD PASS
                        *c10d_functional.all_reduce: 1
                        Transformer.layers.1.attention.wq
                        Transformer.layers.1.attention.wk
                        Transformer.layers.1.attention.wv
                        Transformer.layers.1.attention.wo
                          FORWARD PASS
                            *c10d_functional.all_reduce: 1
                        Transformer.layers.1.attention.resid_dropout
                    Transformer.layers.1.ffn_norm
                    Transformer.layers.1.feed_forward
                      FORWARD PASS
                        *c10d_functional.all_reduce: 1
                        Transformer.layers.1.feed_forward.w1
                        Transformer.layers.1.feed_forward.gelu
                        Transformer.layers.1.feed_forward.w2
                          FORWARD PASS
                            *c10d_functional.all_reduce: 1
                        Transformer.layers.1.feed_forward.resid_dropout
                Transformer.norm
                Transformer.output
                  FORWARD PASS
                    *c10d_functional.all_gather_into_tensor: 1

        """

        torch.manual_seed(0)

        model, inp = self._transformer_model_setup()

        comm_mode = CommDebugMode()
        with comm_mode:
            model(inp)

        # print the module level collective tracing information
        print(comm_mode.generate_comm_debug_tracing_table(noise_level=0))
        comm_mode.log_comm_debug_tracing_table_to_file(noise_level=0)

    def example_MLP_operation_tracing(self) -> None:
        """
        Example code to demonstrate CommModeDebug's module operation level tracing using a distributed MLP model.
        Prints a table of module opoeration level collective tracing information and logs table to comm_mode_log.txt

        Expected output:
        Global
          FORWARD PASS
            *c10d_functional.all_reduce: 1
            **aten.view.default
            **aten.sum.default
            **aten.ones_like.default
          BACKWARD PASS
            **aten.expand.default
            MLPModule
            *module type: class 'torch.testing._internal.distributed._tensor.common_dtensor.MLPModule'
              FORWARD PASS
                *c10d_functional.all_reduce: 1
                **aten.view.default
                **aten.view.default
                **aten.view.default
                MLPModule.net1
                *module type: class 'torch.nn.modules.linear.Linear'
                *Parameter List
                *weight: (Shard(dim=0),)
                *bias: (Shard(dim=0),)
                  FORWARD PASS
                    **aten.detach.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.view.default
                    **aten.t.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.addmm.default
                      shape: [torch.Size([16]), torch.Size([8, 10]), torch.Size([10, 16])]
                      sharding: [(Shard(dim=0),), (Replicate(),), (Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.addmm.default
                    **aten.view.default
                  BACKWARD PASS
                    **aten.t.default
                      shape: [torch.Size([8, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.mm.default
                      shape: [torch.Size([16, 8]), torch.Size([8, 10])]
                      sharding: [(Shard(dim=0),), (Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.mm.default
                    **aten.t.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.sum.dim_IntList
                      shape: [torch.Size([8, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.sum.dim_IntList
                    **aten.view.default
                      shape: [torch.Size([1, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.view.default
                    **aten.detach.default
                      shape: [torch.Size([16])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.t.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.detach.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                MLPModule.relu
                *module type: class 'torch.nn.modules.activation.ReLU'
                  FORWARD PASS
                    **aten.view.default
                    **aten.relu.default
                    **aten.detach.default
                  BACKWARD PASS
                    **aten.detach.default
                    **aten.threshold_backward.default
                MLPModule.net2
                *module type: class 'torch.nn.modules.linear.Linear'
                *Parameter List
                *weight: (Shard(dim=1),)
                *bias: (Replicate(),)
                  FORWARD PASS
                    *c10d_functional.all_reduce: 1
                    **aten.detach.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.view.default
                    **aten.view.default
                      shape: [torch.Size([8, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.view.default
                    **aten.t.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.addmm.default
                      shape: [torch.Size([10]), torch.Size([8, 16]), torch.Size([16, 10])]
                      sharding: [(Replicate(),), (Shard(dim=1),), (Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.div.Tensor
                    **aten.addmm.default
                    **_c10d_functional.all_reduce.default
                    **aten.view.default
                  BACKWARD PASS
                    **aten.t.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.mm.default
                      shape: [torch.Size([8, 10]), torch.Size([10, 16])]
                      sharding: [(Replicate(),), (Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.mm.default
                    **aten.t.default
                      shape: [torch.Size([8, 10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.mm.default
                      shape: [torch.Size([10, 8]), torch.Size([8, 16])]
                      sharding: [(Replicate(),), (Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.mm.default
                    **aten.t.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.sum.dim_IntList
                      shape: [torch.Size([8, 10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.sum.dim_IntList
                    **aten.view.default
                      shape: [torch.Size([1, 10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.view.default
                    **aten.detach.default
                      shape: [torch.Size([10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10])]
                      sharding: [(Replicate(),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                    **aten.t.default
                      shape: [torch.Size([16, 10])]
                      sharding: [(Shard(dim=0),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.t.default
                    **aten.detach.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default
                      shape: [torch.Size([10, 16])]
                      sharding: [(Shard(dim=1),)]
                      device mesh: DeviceMesh([0, 1, 2, 3])
                    **aten.detach.default
                    **aten.detach.default

        """
        torch.manual_seed(0)

        model, inp = self._MLP_model_setup(model_type=MLPModule)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        # print the operation level collective tracing information
        print(comm_mode.generate_comm_debug_tracing_table(noise_level=3))
        comm_mode.log_comm_debug_tracing_table_to_file(noise_level=3)

    def example_transformer_operation_tracing(
        self, is_seq_parallel: bool = False
    ) -> None:
        """
        Example code to demonstrate CommModeDebug's module operation level tracing using a distributed transformer model.
        Prints a table of module opoeration level collective tracing information, excluding trivial operations and logs
        table to transformer_operation_log.txt
        """

        torch.manual_seed(0)

        model, inp = self._transformer_model_setup()

        comm_mode = CommDebugMode()
        with comm_mode:
            model(inp)

        # print the operation level collective tracing information
        print(comm_mode.generate_comm_debug_tracing_table(noise_level=2))
        comm_mode.log_comm_debug_tracing_table_to_file(
            noise_level=1, file_name="transformer_operation_log.txt"
        )

    def example_MLP_json_dump(self) -> None:
        """
        Example code to demonstrate CommModeDebug's json dump using a MLP model. Sends the information to default
        comm_mode_log.json file
        """
        torch.manual_seed(0)

        model, inp = self._MLP_model_setup(model_type=MLPModule)

        comm_mode = CommDebugMode()
        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        comm_mode.generate_json_dump()

    def example_transformer_json_dump(self, is_seq_parallel: bool = False) -> None:
        """
        Example code to demonstrate CommModeDebug's json dump using a transformer model, excluding the trivial
        operations. Sends the information to user-passed transformer_log.json file
        """

        torch.manual_seed(0)

        model, inp = self._transformer_model_setup()

        comm_mode = CommDebugMode()
        with comm_mode:
            model(inp)

        comm_mode.generate_json_dump(file_name="transformer_log.json", noise_level=1)
        comm_mode.generate_json_dump(file_name="transformer_log_2.json", noise_level=2)

    def example_activation_checkpointing(self) -> None:
        """
        Example code showing that CommDebugMode is able to differentiate between backward passes
        and activation checkpointing. Sends the information to default comm_mode_log.json file.
        The output for the example output is shown below:

        Global
          FORWARD PASS
            **aten.sum.default
            **aten.ones_like.default
          BACKWARD PASS
            **aten.expand.default
            Foo
            *module type: class '__main__.CommDebugModeExample.example_activation_checkpointing.locals.Foo'
              FORWARD PASS
                **aten.relu.default
                **aten.empty.memory_format
                **aten.empty.memory_format
                **aten.relu.default
              BACKWARD PASS
                **aten.threshold_backward.default
                Foo.linears.0
                *module type: class 'torch.nn.modules.linear.Linear'
                  FORWARD PASS
                    **aten.addmm.default
                  BACKWARD PASS
                    **aten.mm.default
                    **aten.sum.dim_IntList
                Foo.linears.1
                *module type: class 'torch.nn.modules.linear.Linear'
                  FORWARD PASS
                    **aten.addmm.default
                  ACTIVATION CHECKPOINTING
                    **aten.mm.default
                    **aten.mm.default
                    **aten.sum.dim_IntList
                    **aten.threshold_backward.default
        """

        class Foo(torch.nn.Module):
            def __init__(self, n_layers: int, dim: int, use_ac: bool = False):
                super().__init__()
                self.linears = torch.nn.ModuleList()
                self.use_ac = use_ac
                for _ in range(n_layers):
                    self.linears.append(torch.nn.Linear(dim, dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for i, block in enumerate(self.linears):
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                    assert x is not None
                    x = torch.nn.functional.relu(x)
                return x

        bsz = 2
        dim = 8
        n_layers = 2

        model = Foo(n_layers, dim, True)
        x = torch.randn(bsz, dim)

        comm_mode = CommDebugMode()
        with comm_mode:
            model(x).sum().backward()

        print(comm_mode.generate_comm_debug_tracing_table(noise_level=2))
        comm_mode.log_comm_debug_tracing_table_to_file(noise_level=2)
        comm_mode.generate_json_dump(noise_level=2)


def run_example(world_size: int, rank: int, example_name: str) -> None:
    # set manual seed
    # intializing class with all of the functions
    instantiated_example = CommDebugModeExample(world_size, rank)
    # dict that stores example code function names
    name_to_example_code: Dict[str, Callable[[], None]] = {
        "MLP_distributed_sharding_display": instantiated_example.example_MLP_distributed_sharding_display,
        "MLPStacked_distributed_sharding_display": instantiated_example.example_MLPStacked_distributed_sharding_display,
        "MLP_module_tracing": instantiated_example.example_MLP_module_tracing,
        "transformer_module_tracing": instantiated_example.example_transformer_module_tracing,
        "MLP_operation_tracing": instantiated_example.example_MLP_operation_tracing,
        "transformer_operation_tracing": instantiated_example.example_transformer_operation_tracing,
        "MLP_json_dump": instantiated_example.example_MLP_json_dump,
        "transformer_json_dump": instantiated_example.example_transformer_json_dump,
        "activation_checkpointing": instantiated_example.example_activation_checkpointing,
    }

    name_to_example_code[example_name]()


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4  # our example uses 4 worker ranks

    parser = argparse.ArgumentParser(
        description="comm_mode_feature examples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    example_prompt = (
        "choose one comm_mode_feature example from below:\n"
        "\t1. MLP_distributed_sharding_display\n"
        "\t2. MLPStacked_distributed_sharding_display\n"
        "\t3. MLP_module_tracing\n"
        "\t4. transformer_module_tracing\n"
        "\t5. MLP_operation_tracing\n"
        "\t6. transformer_operation_tracing\n"
        "\t7. MLP_json_dump\n"
        "\t8. transformer_json_dump\n"
        "\t9. activation_checkpointing\n"
        "e.g. you want to try the MLPModule sharding display example, please input 'MLP_distributed_sharding_display'\n"
    )
    parser.add_argument("-e", "--example", help=example_prompt, required=True)
    example = parser.parse_args().example

    run_example(world_size, rank, example)
