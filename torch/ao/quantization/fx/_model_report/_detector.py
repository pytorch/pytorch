from typing import Any, Dict, Set, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig import QConfig
from torch.ao.sparsity.sparsifier.utils import module_to_fqn
from torch.fx import GraphModule
from torch.nn.qat.modules.conv import _ConvNd as QatConvNd
from torch.nn.qat.modules.linear import Linear as QatLinear

# Default map for representing supported per channel quantization modules for different backends
DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: Dict[str, Set[Any]] = {
    "fbgemm": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, QatLinear, QatConvNd]),
    "qnnpack": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, QatLinear, QatConvNd]),
    "onednn": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, QatLinear, QatConvNd]),
}


def _detect_per_channel(model: GraphModule) -> Tuple[str, Dict[str, Any]]:
    """Checks if any Linear or Conv layers in the model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        Looks at q_config format and backend to determine if per_channel can be utilized.
        Uses the DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES structure to determine support

    Args:
        model: The prepared and calibrated model we want to check if using per_channel

    Returns a tuple with two elements:
        String report of potential actions to improve model (if per_channel quantization is available in backend)
        Dictionary mapping per_channel quantizable elements to:
            whether per_channel quantization is supported by the backend
            if it is being utilized in the current model
    """
    backend_chosen = torch.backends.quantized.engine
    supported_modules = set([])
    if backend_chosen in DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
        supported_modules = DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[
            backend_chosen
        ]
    else:
        raise ValueError(
            "Not configured to work with {}. Try a different default backend".format(
                backend_chosen
            )
        )

    # store information on submodules and if per_channel quantization is supported and used as well as qconfig information
    per_channel_info = {"backend": backend_chosen, "per_channel_status": {}}

    def _detect_per_channel_helper(module: nn.Module):
        """
        Recursive operation to determine if per_channel quantization is supported in modules and submodules.

        Populates a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            module: The current module that is being checked to see if it is per_channel qunatizable
        """
        # get the fully qualified name and check if in list of modules to include and list of modules to ignore
        fqn = module_to_fqn(model, module)
        is_in_include_list = (
            True
            if sum(list(map(lambda x: isinstance(module, x), supported_modules))) > 0
            else False
        )

        # check if the module per_channel is supported
        # based on backend
        per_channel_supported = False

        if is_in_include_list:
            per_channel_supported = True

            # assert statement for MyPy
            q_config_file = module.qconfig
            assert isinstance(q_config_file, QConfig)

            # this object should either be fake quant or observer
            q_or_s_obj = module.qconfig.weight.p.func()
            assert isinstance(q_or_s_obj, FakeQuantize) or isinstance(
                q_or_s_obj, ObserverBase
            )

            per_channel_used = False  # will be true if found in qconfig

            if hasattr(
                q_or_s_obj, "ch_axis"
            ):  # then we know that per_channel quantization used

                # all fake quants have channel axis so need to check is_per_channel
                if isinstance(q_or_s_obj, FakeQuantize):
                    if (
                        hasattr(q_or_s_obj, "is_per_channel")
                        and q_or_s_obj.is_per_channel
                    ):
                        per_channel_used = True
                elif isinstance(q_or_s_obj, ObserverBase):
                    # should be an observer otherwise
                    per_channel_used = True
                else:
                    raise ValueError("Should be either observer or fake quant")

            per_channel_info["per_channel_status"][fqn] = {
                "per_channel_supported": per_channel_supported,
                "per_channel_used": per_channel_used,
            }

        # recurse on children
        for child in module.children():
            _detect_per_channel_helper(child)

    # run the helper function to populate the dictionary
    _detect_per_channel_helper(model)

    # String to let the user know of further optimizations
    further_optims_str = "Further Optimizations for backend {}: \n".format(
        backend_chosen
    )

    # assert for MyPy check
    assert isinstance(per_channel_info["per_channel_status"], dict)

    optimizations_possible = False
    for fqn in per_channel_info["per_channel_status"]:
        fqn_dict = per_channel_info["per_channel_status"][fqn]
        if fqn_dict["per_channel_supported"] and not fqn_dict["per_channel_used"]:
            optimizations_possible = True
            further_optims_str += "Module {module_fqn} can be configured to use per_channel quantization.\n".format(
                module_fqn=fqn
            )

    if optimizations_possible:
        further_optims_str += "To use per_channel quantization, make sure the qconfig has a per_channel weight observer."
    else:
        further_optims_str += "No further per_channel optimizations possible."

    # return the string and the dictionary form of same information
    return (further_optims_str, per_channel_info)
