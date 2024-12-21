# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from typing import Any, Dict

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.api import distribute_tensor, DTensor
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    MLPStacked,
    ModelArgs,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    Transformer,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


class TestCommModeFeatures(DTensorTestBase):
    # checks if parameter / sharding info is the same as ground truth
    def check_same_set_of_keys(self, dict1, dict2):
        """
        Used to ensure the comm_mode parameter/sharding dictionaries contain the same information produced by the
        ground truth
        """
        dict1_keys = []
        dict2_keys = []

        for key in dict1:
            for nested_key in dict1[key]:
                dict1_keys.append((key, nested_key))

        for key in dict2:
            for nested_key in dict2[key]:
                dict2_keys.append((key, nested_key))

        self.assertEqual(len(dict1_keys), len(dict2_keys))

        for i in range(len(dict1_keys)):
            self.assertEqual(dict1_keys[i], dict2_keys[i])

    # generates the ground truth parameter and sharding info
    def ground_truth(self, model):
        """
        Used to generate the ground-truth parameter and sharding info for a given distributed model to
        verify comm_mode correctness
        """
        module_parameters_dict: Dict[str, Any] = {}
        module_sharding_dict: Dict[str, Any] = {}

        for name, parameters in model.named_parameters():
            # splits name into module name to create FQN and parameter name
            module_name = model.__class__.__name__ + "." + name.rsplit(".", 1)[0]
            parameter_name = name.rsplit(".", 1)[1]

            if module_name not in module_parameters_dict:
                module_parameters_dict[module_name] = {}

            module_parameters_dict[module_name][parameter_name] = parameters.data

            if isinstance(parameters.data, DTensor):
                key_name = module_name + "." + parameter_name
                module_sharding_dict[key_name] = parameters.data.placements

        return module_parameters_dict, module_sharding_dict

    @with_comms
    def test_MLP_distributed_sharding_display(self):
        """
        tests parameters and sharding on a module level
        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        inp_size = [8, 10]
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        module_parameters_dict, module_sharding_dict = self.ground_truth(model)

        # checks if parameter / sharding info is the same as ground truth
        self.check_same_set_of_keys(
            module_parameters_dict, comm_mode.get_parameter_info()
        )
        self.check_same_set_of_keys(module_sharding_dict, comm_mode.get_sharding_info())

    @with_comms
    def test_MLPStacked_distributed_sharding_display(self):
        """
        tests model with nested modules and makes sure comm_mode correctly resets parameter and sharding information
        """

        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        inp_size = [8, 10]
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        model2 = MLPStacked(self.device_type)

        parallelize_plan = {
            "MLPStacked.layers.0.net1": ColwiseParallel(),
            "MLPStacked.layers.0.net2": RowwiseParallel(),
            "MLPStacked.layers.1.net1": ColwiseParallel(),
            "MLPStacked.layers.1.net2": RowwiseParallel(),
        }

        model2 = parallelize_module(model2, device_mesh, parallelize_plan)

        with comm_mode:
            # ensures that comm_mode is resetting properly
            self.assertEqual(comm_mode.get_parameter_info(), {})
            self.assertEqual(comm_mode.get_sharding_info(), {})

            output_tp = model2(inp)

        module_parameters_dict, module_sharding_dict = self.ground_truth(model2)

        self.check_same_set_of_keys(
            module_parameters_dict, comm_mode.get_parameter_info()
        )
        self.check_same_set_of_keys(module_sharding_dict, comm_mode.get_sharding_info())
        self.assertEqual(len(comm_mode.get_sharding_info()), 8)

    @with_comms
    def test_MLP_module_tracing(self):
        """
        tests module-level tracing for MLP module
        """

        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        # checks to see if all sub-modules make it into the module_depth_dictionary
        self.assertEqual(len(comm_mode.advanced_module_tracker.module_helper_dict), 5)

        # checks to see if all collectives were correctly traced at the module-level

        self.assertEqual(
            comm_mode.comm_module_counts["Global"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["MLPModule"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["MLPModule.net2"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )

    @skip_unless_torch_gpu
    @with_comms
    def test_transformer_module_tracing(self, is_seq_parallel=False):
        """
        tests module-level tracing for more complicated transformer module and
        ensures that comm_module depth and tracing dictionaries correctly reset
        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()
        with comm_mode:
            self.assertEqual(
                len(comm_mode.advanced_module_tracker.module_helper_dict), 1
            )
            self.assertEqual(
                comm_mode.comm_module_counts,
                {"Global": {"forward": {}, "backward": {}}},
            )
            model(inp)

        model_args = ModelArgs(dropout_p=0.0)
        model2 = Transformer(model_args).to(device=self.device_type)
        model2 = Transformer.parallelize(model2, device_mesh, is_seq_parallel)

        inp_size = [8, 8]

        inp = torch.randint(model_args.vocab_size, inp_size, device=self.device_type)
        inp = distribute_tensor(inp, device_mesh=device_mesh)

        comm_mode = CommDebugMode()
        with comm_mode:
            model2(inp)

        # checks to see if all collectives were correctly traced at the module-level
        self.assertEqual(
            comm_mode.comm_module_counts["Global"]["forward"][
                c10d_functional.all_reduce
            ],
            6,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Global"]["forward"][
                c10d_functional.all_gather_into_tensor
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer"]["forward"][
                c10d_functional.all_reduce
            ],
            6,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer"]["forward"][
                c10d_functional.all_gather_into_tensor
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.tok_embeddings"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.pos_embeddings"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.0"]["forward"][
                c10d_functional.all_reduce
            ],
            2,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.0.attention"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.0.attention.wo"][
                "forward"
            ][c10d_functional.all_reduce],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.0.feed_forward"][
                "forward"
            ][c10d_functional.all_reduce],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.0.feed_forward.w2"][
                "forward"
            ][c10d_functional.all_reduce],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.1"]["forward"][
                c10d_functional.all_reduce
            ],
            2,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.1.attention"]["forward"][
                c10d_functional.all_reduce
            ],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.1.attention.wo"][
                "forward"
            ][c10d_functional.all_reduce],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.1.feed_forward"][
                "forward"
            ][c10d_functional.all_reduce],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.layers.1.feed_forward.w2"][
                "forward"
            ][c10d_functional.all_reduce],
            1,
        )
        self.assertEqual(
            comm_mode.comm_module_counts["Transformer.output"]["forward"][
                c10d_functional.all_gather_into_tensor
            ],
            1,
        )


if __name__ == "__main__":
    run_tests()
