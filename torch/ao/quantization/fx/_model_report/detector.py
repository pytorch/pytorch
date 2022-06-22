from typing import Any, Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import QConfig

# Adding base class for detectors
class DetectorBase(ABC):
    r""" Base Detector Module
    Any detector class should derive from this class.

    Concrete detectors should follow the same general API, which includes:
    - A method to calculate and return observer insertion points
        - Should return both the fqns and the Observer class to insert
    - A method to return a report based on the the detector
        - Should return a str-based report and dict info in Tuple[str,Dict] format
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def determine_observer_insert_points(self, model) -> Dict:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Tuple of two elements:
            Set[str] of observer fqns denoting where to insert observers
            ObserverBase (or subclass): the class (not an instance) of the observer to insert
        """
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        r""" Returns the name of the current detector """
        pass

    @abstractmethod
    def generate_detector_report(self, model) -> Tuple[str, Dict[str, Any]]:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Tuple of two elements:
            Str: string report of the suggested improvements
            Dict: contains useful data collected by the observer pertinent to this report
        """
        pass

class PerChannelDetector(DetectorBase):
    r""" This class is used to detect if any Linear or Conv layers in a model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        per_channel quantization can lead to major benefits in the form of accuracy.
        Therefore, if the backend used by the user supports it, it is recommended to use

        Args:
            backend (str, optional): the backend the user wishes to use in production
                Default value is current torch.backends.quantized.engine
    """

    # Default map for representing supported per channel quantization modules for different backends
    DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: Dict[str, Set[Any]] = {
        "fbgemm": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d]),
        "qnnpack": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d]),
        "onednn": set([nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d]),
    }

    def __init__(self, backend: str = torch.backends.quantized.engine):
        super().__init__()

        # store the backend information
        self.backend_chosen = backend
        self.supported_modules = set([])
        if self.backend_chosen in self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
            self.supported_modules = self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[self.backend_chosen]
        else:
            raise ValueError("Not configured to work with {}. Try a different default backend".format(self.backend_chosen))

    def get_detector_name(self) -> str:
        r""" returns the string name of this detector"""
        return "per_channel_detector"

    def determine_observer_insert_points(self, model: nn.Module) -> Dict:
        r"""
        There is no observers inserted for the PerChannelDetector.

        Returns an empty dictionary since no observers are added or needed
        """
        return {}


    def _detect_per_channel_helper(self, model: nn.Module, per_channel_info: Dict):
        r"""
        determines if per_channel quantization is supported in modules and submodules.

        Returns a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            module: The current module that is being checked to see if it is per_channel qunatizable

        Returns dictionary mapping fqns to if per_channel quantization is possible
        """
        for named_mod in model.named_modules():

            # get the fully qualified name and check if in list of modules to include and list of modules to ignore
            fqn, module = named_mod

            # asserts for MyPy
            assert isinstance(fqn, str) and isinstance(per_channel_info["per_channel_status"], dict)

            is_in_include_list = sum(list(map(lambda x: isinstance(module, x), self.supported_modules))) > 0

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

        return per_channel_info

    def generate_detector_report(self, model: nn.Module) -> Tuple[str, Dict[str, Any]]:
        r"""Checks if any Linear or Conv layers in the model utilize per_channel quantization.
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

        # store information on submodules and if per_channel quantization is supported and used as well as qconfig information
        per_channel_info = {"backend": self.backend_chosen, "per_channel_status": {}}

        # run the helper function to populate the dictionary
        per_channel_info = self._detect_per_channel_helper(model, per_channel_info)

        # String to let the user know of further optimizations
        further_optims_str = "Further Optimizations for backend {}: \n".format(self.backend_chosen)

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


class DynamicStaticDetector(DetectorBase):
    r"""
    Determines whether dynamic or static quantization is more appropriate for a given module.

    Takes advantage of the ModelReportObserver that records range information.
    Stationary distribution of data are strictly above tolerance level for the comparison statistic:

        S = average_batch_activation_range/epoch_activation_range

    Nonstationary distributions are below or at the tolerance level for this metric.

    If the distribution of data right after the module is non-stationary, recommend dynamic quantization
        Otherwise recommend static quantization

    Args:
        tolerance (float, optional): The threshold where S metric is stationary above and non-stationary otherwise. Default: 0.5
    """
    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME = "model_report_pre_observer"
    DEFAULT_POST_OBSERVER_NAME = "model_report_post_observer"

    # naming conventions for stationary vs non-stationary data
    DEFAULT_STATIONARY = "stationary"
    DEFAULT_NON_STATIONARY = "non-stationary"

    # naming conventions for the keys of the return module info
    DEFAULT_TOLERANCE_KEY = "tolerance"
    DEFAULT_DYNAMIC_REC_KEY = "dynamic_recommended"
    DEFAULT_PRE_OBS_COMP_STAT_KEY = "pre_observer_comp_stat"
    DEFAULT_POST_OBS_COMP_STAT_KEY = "post_observer_comp_stat"
    DEFAULT_PRE_OBS_DATA_DIST_KEY = "pre_observer_data_dist"
    DEFAULT_POST_OBS_DATA_DIST_KEY = "post_observer_data_dist"

    # modules that are supported both dynamic and static for this report function
    DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED = set([nn.Linear])

    def __init__(self, tolerance=0.5):
        super().__init__()

        # set tolerance level and initialize a set to keep track of useful fqn locations
        self.tolerance = tolerance
        self.useful_observer_fqns: Set[str] = set([])

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        r"""
        Determines where observers need to be inserted for the Dynamic vs Static detector.
        For this detector, we want to place observers on either side of linear layers in the model.

        Currently inserts observers for:
            linear layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "insert_observer" -> the observer we wish to insert (ObserverBase)
            key "insert_post" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # check to see if module is of a supported type
            is_supported_type = sum(list(map(lambda x: isinstance(module, x), self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED))) > 0

            if is_supported_type:
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    "target_node": targeted_node,
                    "insert_observer": obs_ctr,
                    "insert_post": False,
                    "observer_args": targeted_node.args
                }

                # add entry for post-observer
                post_obs_fqn = fqn + "." + self.DEFAULT_POST_OBSERVER_NAME

                obs_fqn_to_info[post_obs_fqn] = {
                    "target_node": targeted_node,
                    "insert_observer": obs_ctr,
                    "insert_post": True,
                    "observer_args": (targeted_node,)
                }

        return obs_fqn_to_info


    def _get_targeting_node(self, prepared_fx_model: GraphModule, target_fqn: str) -> torch.fx.node.Node:
        r"""
        Takes in a GraphModule and the target_fqn and finds the node object that targets this fqn

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            target_fqn (str): The fqn of the layer we are trying to target

        Returns the node object we are trying to add observers around
        """
        for node in prepared_fx_model.graph.nodes:
            # if the node's target is our target, return it
            if node.target == target_fqn:
                return node

        raise ValueError("passed in target_fqn not found in graph's targets.")

    def get_detector_name(self) -> str:
        r""" returns the string name of this detector"""
        return "dynamic_vs_static_detector"

    def _generate_dict_info(self, model: GraphModule) -> Dict[str, Any]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
        """
        # store modules dynamic vs static information
        module_dynamic_static_info = {}

        # This for loop goes through the modules, and extracts all relavent information into module_dynamic_static_info
        #   This information primary includes whether the data distributions around a supported module is stationary or not
        #   Based on this, it is recorded whether dynamic or static quantization is recommended

        # loop through all submodules included nested ones
        for fqn, module in model.named_modules():

            # check to see if module is of a supported type
            is_supported_type = sum(list(map(lambda x: isinstance(module, x), self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED))) > 0

            # if module is Linear has the ModelReportObserver attached to it
            if (
                is_supported_type
                and hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                and hasattr(module, self.DEFAULT_POST_OBSERVER_NAME)
            ):
                # get pre and post observers for the module
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                post_obs = getattr(module, self.DEFAULT_POST_OBSERVER_NAME)

                # get the statistics for each module
                pre_stat = pre_obs.get_batch_to_epoch_ratio()
                post_stat = post_obs.get_batch_to_epoch_ratio()

                # record module, pre and post stat, and whether to do dynamic or static based off it
                # true if post observer data distribution is non-stationary, false if it's stationary
                dynamic_recommended = post_stat <= self.tolerance

                # specify the classifications for whether data distributions considered stationary or non-stationary
                pre_obs_dist_classif = self.DEFAULT_STATIONARY if pre_stat > self.tolerance else self.DEFAULT_NON_STATIONARY
                post_obs_dist_classif = self.DEFAULT_STATIONARY if post_stat > self.tolerance else self.DEFAULT_NON_STATIONARY

                # store the set of important information for this module
                module_info = {
                    self.DEFAULT_TOLERANCE_KEY: self.tolerance,
                    self.DEFAULT_DYNAMIC_REC_KEY: dynamic_recommended,
                    self.DEFAULT_PRE_OBS_COMP_STAT_KEY: pre_stat,
                    self.DEFAULT_PRE_OBS_DATA_DIST_KEY: pre_obs_dist_classif,
                    self.DEFAULT_POST_OBS_COMP_STAT_KEY: post_stat,
                    self.DEFAULT_POST_OBS_DATA_DIST_KEY: post_obs_dist_classif,
                }

                module_dynamic_static_info[fqn] = module_info

        return module_dynamic_static_info

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        r"""
        Determines whether dynamic or static quantization is more appropriate for a given module.

        Takes advantage of the ModelReportObserver that records range information.
        Stationary distribution of data are strictly above tolerance level for the comparison statistic:

            S = average_batch_activation_range/epoch_activation_range

        Nonstationary distributions are below or at the tolerance level for this metric.

        If the distribution of data right after the module is non-stationary, recommend dynamic quantization
            Otherwise recommend static quantization

        This will then generate suggestions for dynamic vs static quantization focused around Linear.

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of of whether dynamic or static quantization is recommended for certain modules
            Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
        """

        # get the dictionary of the information to format the string report
        module_dynamic_static_info = self._generate_dict_info(model)

        dynamic_vs_static_string = "Dynamic vs. Static Quantization suggestions: \n"

        # This for loop goes through the information collected in module_dynamic_static_info and:
        #   Populates the string based report with the information from module_dynamic_static_info
        #   Compiles the complete report by appending relavent formatted strings

        for module_fqn in module_dynamic_static_info.keys():

            module_info = module_dynamic_static_info[module_fqn]
            suggestion_string_template = "For module {} it is suggested to use {} quantization because {}.\n"

            # decide what string formatting values will be
            quantization_type = ""

            quantization_reasoning = "the distribution of data before {} is {} and the distribution after is {}."
            dynamic_benefit = " You will get more accurate results if you use dynamic quantization"
            static_benefit = " You can increase model efficiency if you use static quantization"
            benefit_str = ""

            # strings for if dynamic quantized per tensor is needed
            recommend_per_tensor = " We recommend to add a {} before this module if it is static."
            rec_lay_to_add = "dynamic quantize per tensor layer"
            dynamic_per_tensor_string = recommend_per_tensor.format(rec_lay_to_add)
            dynamic_per_tensor_reasoning_string = (
                " This is because the input to this module has a non-stationary distribution."
            )

            # start composing explanation
            if module_info[self.DEFAULT_DYNAMIC_REC_KEY]:
                quantization_type = "dynamic"
                benefit_str = dynamic_benefit
            else:
                quantization_type = "static"
                benefit_str = static_benefit

            # now set the quantization explanation string
            quantization_reasoning = (
                quantization_reasoning.format(
                    module_fqn, module_info[self.DEFAULT_PRE_OBS_DATA_DIST_KEY], module_info[self.DEFAULT_POST_OBS_DATA_DIST_KEY]
                )
                + benefit_str
            )

            # if we have a non-stationary input -> linear -> stationary we suggested static
            # however, we want to also recommend they add a dynamic quantize per tensor right if this change is made
            if (
                module_info[self.DEFAULT_PRE_OBS_DATA_DIST_KEY] == self.DEFAULT_NON_STATIONARY
                and module_info[self.DEFAULT_POST_OBS_DATA_DIST_KEY] == self.DEFAULT_STATIONARY
            ):
                quantization_reasoning = (
                    quantization_reasoning + dynamic_per_tensor_string + dynamic_per_tensor_reasoning_string
                )

            # format the overall suggestion string with the specific inputs
            module_suggestion_string = suggestion_string_template.format(
                module_fqn, quantization_type, quantization_reasoning
            )

            # append to overall suggestion
            dynamic_vs_static_string += module_suggestion_string

        # return the string as well as the dictionary of information
        return (dynamic_vs_static_string, module_dynamic_static_info)
