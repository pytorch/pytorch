from typing import Any, Dict

import torch

from torch.distributed._tensor.debug import CommDebugMode

from torch.distributed._tensor.debug.comm_mode import ModuleParamaterShardingTracker

from torch.testing._internal.distributed._tensor.common_dtensor import (
    MLPModule,
    MLPStacked,
)


class DisplayShardingExample:
    """
    Checks if the set of keys in ground truth dictionary and the set
    produced in advanced_module_tracker are in the same order
    """

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
        """
        Example of using obtaining all module's FQN and parameters for a given model
        """

        inp_size = [8, 10]

        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size)
        model = MLPModule(None)

        LR = 0.25

        optim = torch.optim.SGD(model.parameters(), lr=LR)
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


if __name__ == "__main__":
    instantiated_test = DisplayShardingExample()
    instantiated_test.test_display_parameters_MLP()
