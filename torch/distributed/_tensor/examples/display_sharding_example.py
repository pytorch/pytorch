from typing import Dict, List

import torch
from torch.distributed._tensor.debug import CommDebugMode

from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.utils.module_tracker import ModuleTracker


class DisplayShardingExampleTest:
    def test_mlp_training_e2e(self):
        inp_size = [8, 10]

        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=None)
        model = MLPModule(None)

        LR = 0.25

        optim = torch.optim.SGD(model.parameters(), lr=LR)
        comm_mode = CommDebugMode()
        module_tracker = ModuleTracker()

        with comm_mode, module_tracker:
            output = model(inp)
            output.sum().backward()

        # gets all parameters in all modules in the model
        module_parameters: Dict[str, List[str]] = {}

        for name, module in model.named_modules():
            if module._parameters:
                module_parameters[name] = []
                for param_name, param in module.named_parameters():
                    module_parameters[name].append(param_name)

        # gets all modules that were actually seen during operation
        seen_module_parameters: Dict[str, List[str]] = {}
        keys = list(module_tracker._known_modules.keys())[1:]
        for key in keys:
            last_period_index = module_tracker._known_modules[key].rfind(".")
            result = (
                module_tracker._known_modules[key][last_period_index + 1 :]
                if last_period_index != -1
                else module_tracker._known_modules[key]
            )
            if result in module_parameters:
                seen_module_parameters[result] = module_parameters[result]

        print(seen_module_parameters)


if __name__ == "__main__":
    instantiated_test = DisplayShardingExampleTest()
    instantiated_test.test_mlp_training_e2e()
