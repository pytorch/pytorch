from typing import Any, Dict, Set, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig import QConfig
from torch.nn.qat.modules.conv import _ConvNd as QatConvNd
from torch.nn.qat.modules.linear import Linear as QatLinear


# Default map for representing supported per channel quantization modules for different backends
DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: Dict[str, Set[Any]] = {
    "fbgemm": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, QatLinear, QatConvNd]),
    "qnnpack": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, QatLinear, QatConvNd]),
    "onednn": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, QatLinear, QatConvNd]),
}


def _detect_per_channel(model: nn.Module) -> Tuple[str, Dict[str, Any]]:
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
        supported_modules = DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[backend_chosen]
    else:
        raise ValueError("Not configured to work with {}. Try a different default backend".format(backend_chosen))

    # store information on submodules and if per_channel quantization is supported and used as well as qconfig information
    per_channel_info = {"backend": backend_chosen, "per_channel_status": {}}

    def _detect_per_channel_helper(model: nn.Module):
        """
        determines if per_channel quantization is supported in modules and submodules.

        Populates a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            module: The current module that is being checked to see if it is per_channel qunatizable
        """
        for named_mod in model.named_modules():

            # get the fully qualified name and check if in list of modules to include and list of modules to ignore
            fqn, module = named_mod

            # asserts for MyPy
            assert isinstance(fqn, str) and isinstance(per_channel_info["per_channel_status"], dict)

            is_in_include_list = (
                True if sum(list(map(lambda x: isinstance(module, x), supported_modules))) > 0 else False
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
                assert isinstance(q_or_s_obj, FakeQuantize) or isinstance(q_or_s_obj, ObserverBase)

                per_channel_used = False  # will be true if found in qconfig

                if hasattr(q_or_s_obj, "ch_axis"):  # then we know that per_channel quantization used

                    # all fake quants have channel axis so need to check is_per_channel
                    if isinstance(q_or_s_obj, FakeQuantize):
                        if hasattr(q_or_s_obj, "is_per_channel") and q_or_s_obj.is_per_channel:
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

    # run the helper function to populate the dictionary
    _detect_per_channel_helper(model)

    # String to let the user know of further optimizations
    further_optims_str = "Further Optimizations for backend {}: \n".format(backend_chosen)

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
        further_optims_str += (
            "To use per_channel quantization, make sure the qconfig has a per_channel weight observer."
        )
    else:
        further_optims_str += "No further per_channel optimizations possible."

    # return the string and the dictionary form of same information
    return (further_optims_str, per_channel_info)


# names for the pre and post observers that are inserted
DEFAULT_PRE_OBSERVER_NAME = "model_report_pre_observer"
DEFAULT_POST_OBSERVER_NAME = "model_report_post_observer"

# naming conventions for stationary vs non-stationary data
DEFAULT_STATIONARY = "stationary"
DEFAULT_NON_STATIONARY = "non-stationary"


def _detect_dynamic_vs_static(model: GraphModule, tolerance=0.5) -> Tuple[str, Dict[str, Any]]:
    """
    Determines whether dynamic or static quantization is more appropriate for a given module.

    Takes advantage of the ModelReportObserver that records range information.
    Stationary distribution of data are strictly above tolerance level for the comparison statistic:

        S = average_batch_activation_range/epoch_activation_range

    Nonstationary distributions are below or at the tolerance level for this metric.

    If the distribution of data right after the module is non-stationary, recommend dynamic quantization
        Otherwise recommend static quantization

    This will then generate suggestions for dynamic vs static quantization focused around Linear.

    Args:
        model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers around layers of interest
        tolerance (float, optional): The threshold where S metric is stationary above and non-stationary otherwise. Default: 0.5
    """

    # store modules dynamic vs static information
    module_dynamic_static_info = {}

    # loop through all submodules included nested ones
    for name, module in model.named_modules():
        # if module has the ModelReportObserver attached to it
        if hasattr(module, DEFAULT_PRE_OBSERVER_NAME) and hasattr(module, DEFAULT_POST_OBSERVER_NAME):
            # get pre and post observers for the module
            pre_obs = getattr(module, DEFAULT_PRE_OBSERVER_NAME)
            post_obs = getattr(module, DEFAULT_POST_OBSERVER_NAME)

            # get the statistics for each module
            pre_stat = pre_obs.get_batch_to_epoch_ratio()
            post_stat = post_obs.get_batch_to_epoch_ratio()

            # record module, pre and post stat, and whether to do dynamic or static based off it
            # true if post observer data distribution is non-stationary, false if it's stationary
            dynamic_recommended = True if post_stat <= tolerance else False

            # specify the classifications for whether data distributions considered stationary or non-stationary
            pre_obs_dist_classification = DEFAULT_STATIONARY if pre_stat > tolerance else DEFAULT_NON_STATIONARY
            post_obs_dist_classification = DEFAULT_STATIONARY if post_stat > tolerance else DEFAULT_NON_STATIONARY

            # store the set of important information for this module
            module_info = {
                "tolerance": tolerance,
                "dynamic_recommended": dynamic_recommended,
                "pre_observer_comp_stat": pre_stat,
                "pre_observer_data_dist": pre_obs_dist_classification,
                "post_observer_comp_stat": post_stat,
                "post_observer_data_dist": post_obs_dist_classification,
            }

            module_dynamic_static_info[name] = module_info

    dynamic_vs_static_string = "Dynamic vs. Static Quantization suggestions: \n"

    for module_name in module_dynamic_static_info.keys():

        module_info = module_dynamic_static_info[module_name]
        suggestion_string_template = "For module {} it is suggested to use {} quantization because {}.\n"

        # decide what string formatting values will be
        quantization_type = ""

        quantization_reasoning = "the distribution of data before {} is {} and the distribution after is {}."
        dynamic_benefit = " You will get more accurate results if you use dynamic quantization"
        static_benefit = " You can increase model efficiency if you use static quantization"

        # strings for if dynamic quantized per tensor is needed
        recommend_per_tensor = " We recommend to add a {} before this module if it is static."
        rec_lay_to_add = "dynamic quantize per tensor layer"
        dynamic_per_tensor_string = recommend_per_tensor.format(rec_lay_to_add)
        dynamic_per_tensor_reasoning_string = " This is because the input to this module has a non-stationary distribution."

        # start composing explanation
        if module_info["dynamic_recommended"]:
            quantization_type = "dynamic"
            quantization_reasoning = (
                quantization_reasoning.format(
                    module_name, module_info["pre_observer_data_dist"], module_info["post_observer_data_dist"]
                )
                + dynamic_benefit
            )
        else:
            quantization_type = "static"
            quantization_reasoning = (
                quantization_reasoning.format(
                    module_name, module_info["pre_observer_data_dist"], module_info["post_observer_data_dist"]
                )
                + static_benefit
            )

        # if we have a non-stationary input -> linear -> stationary we suggested static
        # however, we want to also recommend they add a dynamic quantize per tensor right if this change is made
        if (
            module_info["pre_observer_data_dist"] == DEFAULT_NON_STATIONARY
            and module_info["post_observer_data_dist"] == DEFAULT_STATIONARY
        ):
            quantization_reasoning = (
                quantization_reasoning + dynamic_per_tensor_string + dynamic_per_tensor_reasoning_string
            )

        # format the overall suggestion string with the specific inputs
        module_suggestion_string = suggestion_string_template.format(
            module_name, quantization_type, quantization_reasoning
        )

        # append to overall suggestion
        dynamic_vs_static_string += module_suggestion_string

    # return the string as well as the dictionary of information
    return (dynamic_vs_static_string, module_dynamic_static_info)
