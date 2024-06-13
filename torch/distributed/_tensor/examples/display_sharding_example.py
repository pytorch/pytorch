# mypy: allow-untyped-defs
from typing import Any, Dict

import torch

from torch.distributed._tensor import DeviceMesh, Shard
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.debug.comm_mode import ModuleParamaterShardingTracker

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    MLPModule,
    MLPStacked,
    NUM_DEVICES,
)


def get_device_type():
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )


c10d_functional = torch.ops.c10d_functional

aten = torch.ops.aten
supported_ops = [aten.view.default, aten._to_copy.default]


class DisplayShardingExample:
    """
    Checks if the set of keys in ground truth dictionary and the set
    produced in advanced_module_tracker are in the same order
    """

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.device_type = get_device_type()

    def same_set_of_keys(self, dict1, dict2):
        dict1_keys = []
        dict2_keys = []

        for key in dict1:
            for nested_key in dict1[key]:
                dict1_keys.append((key, nested_key))

        for key in dict2:
            for nested_key in dict2[key]:
                dict2_keys.append((key, nested_key))

        if len(dict1_keys) != len(dict2_keys):
            return False

        for i in range(len(dict1_keys)):
            if dict1_keys[i] != dict2_keys[i]:
                return False

        return True

    def ground_truth(self, model):
        module_parameters_dict: Dict[str, Any] = {}

        for name, parameters in model.named_parameters():
            module_name = model.__class__.__name__ + "." + name.rsplit(".", 1)[0]
            parameter_name = name.rsplit(".", 1)[1]

            if module_name not in module_parameters_dict:
                module_parameters_dict[module_name] = {}

            module_parameters_dict[module_name][parameter_name] = parameters.data

        return module_parameters_dict

    def test_display_parameters_MLP(self):
        """Example of obtaining all module's FQN and parameters for a given model"""

        inp_size = [8, 10]

        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size)
        model = MLPModule(None)

        LR = 0.25

        comm_mode = CommDebugMode()
        module_tracker = ModuleParamaterShardingTracker()

        with comm_mode, module_tracker:
            output = model(inp)
            output.sum().backward()

        print(
            self.same_set_of_keys(
                self.ground_truth(model), module_tracker.module_parameters_dict
            )
        )

        model2 = MLPStacked(None)
        with comm_mode, module_tracker:
            output = model2(inp)

        print(
            self.same_set_of_keys(
                self.ground_truth(model2), module_tracker.module_parameters_dict
            )
        )

    def test_display_parameters_MLP_distributed(
        self, is_seq_parallel=False, recompute_activation=False
    ):
        "Example of obtaining all module's FQN and parameters for a given distributed model and printing the sharding info"
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        LR = 0.25

        parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel(),
            "net2": RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        print(
            self.same_set_of_keys(
                self.ground_truth(model), comm_mode.get_parameter_info()
            )
        )

        comm_mode.print_sharding_info()


def run_example(world_size, rank):
    # set manual seed
    torch.manual_seed(0)

    # run the example
    instantiated_test = DisplayShardingExample(world_size, rank)
    instantiated_test.test_display_parameters_MLP_distributed()


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    import os

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4  # our example uses 4 worker ranks

    run_example(world_size, rank)
