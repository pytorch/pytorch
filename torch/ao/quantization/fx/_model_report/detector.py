from typing import Any, Dict, Set, Tuple, Callable

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

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict.
            This dict maps string keys to detector specific information
        """
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        r""" Returns the name of the current detector """
        pass

    def _get_targeting_node(self, prepared_fx_model: GraphModule, target_fqn: str) -> torch.fx.node.Node:
        r"""
        Takes in a GraphModule and the target_fqn and finds the node object that targets this fqn

        If it's not found, it means it is most likely inside a fused layer
            We just go one layer up in terms of the fqn we are searching for until we find parent node
            If we get to empty string, then we know that it doesn't exist

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            target_fqn (str): The fqn of the layer we are trying to target

        Returns the node object we are trying to add observers around
        """
        for node in prepared_fx_model.graph.nodes:
            # if the node's target is our target, return it
            if node.target == target_fqn:
                return node

        # getting here means node not found
        # if no "." we are already at base and failed
        parent_fqn_sep_index = target_fqn.rfind(".")
        if parent_fqn_sep_index == -1:
            raise ValueError("passed in target_fqn not found in graph's targets.")
        else:
            # recursively call it with parent fqn
            return self._get_targeting_node(prepared_fx_model, target_fqn[:parent_fqn_sep_index])

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
            is_supported_type: bool = (
                sum(list(map(lambda x: isinstance(module, x), self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED))) > 0
            )

            if is_supported_type:
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    "target_node": targeted_node,
                    "insert_observer": obs_ctr(),
                    "insert_post": False,
                    "observer_args": targeted_node.args
                }

                # add entry for post-observer
                post_obs_fqn = fqn + "." + self.DEFAULT_POST_OBSERVER_NAME

                obs_fqn_to_info[post_obs_fqn] = {
                    "target_node": targeted_node,
                    "insert_observer": obs_ctr(),
                    "insert_post": True,
                    "observer_args": (targeted_node,)
                }

        return obs_fqn_to_info

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


class InputWeightEqualizationDetector(DetectorBase):
    r"""
    Determines whether input-weight equalization can help improve quantization for certain modules.

    Specifically, this list of modules includes:
        linear
        conv

    Determines whether input-weight equalization is recommended based on the comp stat:
        s_c = sqrt(w_c/W)/sqrt(i_c/I)
        where:
            w_c is range of weight for channel c, W is range of weight over all channels
            i_c is range of input for channel c, I is range of input over all channels

        if s_c >= threshold or <= 1 / threshold, recommends input-weight equalization

    Args:
        ratio_threshold (float): The threshold for s_c to determine if input-weight equalization is sugggested
            Should be between 0 and 1 (both non-inclusive)
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for s_c to determine if input-weight equalization is sugggested
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine input weight equalization

    * :attr:`SUPPORTED_MODULES`: This specifies the modules that are supported for input-weight equalization

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """

    SUPPORTED_MODULES: Set[Callable] = set(
        [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d]
    )

    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME: str = "model_report_pre_observer"
    DEFAULT_POST_OBSERVER_NAME: str = "model_report_post_observer"

    # string names for keys of info dictionaries
    PER_CHANNEL_MAX_KEY = "per_channel_max"
    PER_CHANNEL_MIN_KEY = "per_channel_min"
    GLOBAL_MAX_KEY = "global_max"
    GLOBAL_MIN_KEY = "global_min"

    # keys for return dict of recommendations
    RECOMMENDED_KEY = "input_weight_equalization_recommended"
    COMP_METRIC_KEY = "channel_comparison_metrics"
    THRESHOLD_KEY = "threshold"
    CHANNEL_KEY = "channel_axis_selected"
    INPUT_INFO_KEY = "input_range_info"
    WEIGHT_INFO_KEY = "weight_range_info"

    def __init__(self, ratio_threshold: float, ch_axis: int = 1):
        # ensure passed in inputs are valid
        if ratio_threshold <= 0 or ratio_threshold >= 1:
            raise ValueError("Make sure threshold is > 0 and < 1")

        # intialize attributes based on args
        self.ratio_threshold: float = ratio_threshold
        self.ch_axis: int = ch_axis

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        r"""Determines where observers need to be inserted for the Input Weight Equalization Detector.
        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            linear layers
            conv layers

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
            is_supported_type: bool = (
                sum(list(map(lambda x: type(module) is x, self.SUPPORTED_MODULES))) > 0
            )

            if is_supported_type:
                # if it's a supported type, we want to get node and add observer insert locations

                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    "target_node": targeted_node,
                    "insert_observer": obs_ctr(ch_axis=self.ch_axis),
                    "insert_post": False,
                    "observer_args": targeted_node.args,
                }

                post_obs_fqn = fqn + "." + self.DEFAULT_POST_OBSERVER_NAME
                # add entry for post observer
                obs_fqn_to_info[post_obs_fqn] = {
                    "target_node": targeted_node,
                    "insert_observer": obs_ctr(ch_axis=self.ch_axis),
                    "insert_post": True,
                    "observer_args": (targeted_node,),
                }

        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        r"""Returns the name of this detector"""
        return "input_weight_equalization_detector"

    def _extract_input_info(self, model: GraphModule) -> Dict[str, Dict]:
        r"""
        Takes in a callibrated GraphModule and then finds the relavent observers.
        It then extracts the input information for each observer returns it

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relavent module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """

        # return dictionary mapping observer fqns to desired info
        input_info: Dict[str, Dict] = {}

        for fqn, module in model.named_modules():
            # check to see if module is of a supported type
            is_supported_type: bool = (
                sum(list(map(lambda x: type(module) is x, self.SUPPORTED_MODULES))) > 0
            )

            # if module is supported and it has a pre-observer
            if is_supported_type and hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME):
                # get pre observer for the module
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

                input_info[fqn] = {
                    self.PER_CHANNEL_MAX_KEY: pre_obs.max_val,
                    self.PER_CHANNEL_MIN_KEY: pre_obs.min_val,
                    self.GLOBAL_MAX_KEY: max(pre_obs.max_val),
                    self.GLOBAL_MIN_KEY: min(pre_obs.min_val),
                }

        return input_info

    def _extract_weight_info(self, model: GraphModule) -> Dict[str, Dict]:
        r"""
        Takes in a callibrated GraphModule and then finds the relavent observers.
        It then extracts the weight information for each layer an observer is attached to.

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """
        # return dictionary mapping observer fqns to desired info
        weight_info: Dict[str, Dict] = {}

        for fqn, module in model.named_modules():
            # check to see if module is of a supported type
            is_supported_type: bool = (
                sum(list(map(lambda x: type(module) is x, self.SUPPORTED_MODULES))) > 0
            )

            # if module is supported and it has a pre-observer
            if is_supported_type and hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME):
                # we don't need actual observer, just the module weights
                # calculate min and max vals
                min_val, max_val = torch.aminmax(module.weight, dim=self.ch_axis)

                # flatten entries since conv can have multiple dimensions
                min_val = torch.flatten(min_val)
                max_val = torch.flatten(max_val)

                weight_info[fqn] = {
                    self.PER_CHANNEL_MAX_KEY: max_val,
                    self.PER_CHANNEL_MIN_KEY: min_val,
                    self.GLOBAL_MAX_KEY: max(max_val),
                    self.GLOBAL_MIN_KEY: min(min_val),
                }

        return weight_info

    def _calculate_range_ratio(self, info_dict: Dict) -> torch.Tensor:
        r"""
        Takes in an info dict and calculates the s_c matrix.

        Args:
            info_dict (dict): A dictionary of either input or weight range info

        Returns a tensor of values, where each value is the s_c stat for a different channel
        """
        # calculate the ratios of the info
        per_channel_range = info_dict[self.PER_CHANNEL_MAX_KEY] - info_dict[self.PER_CHANNEL_MIN_KEY]
        global_range = info_dict[self.GLOBAL_MAX_KEY] - info_dict[self.GLOBAL_MIN_KEY]

        # if global range is 0, throw error
        if global_range == 0:
            raise ValueError("The range of the info dict is 0")

        ratio = per_channel_range / global_range

        return ratio

    def _generate_comparision_values(self, input_info: Dict, weight_info: Dict) -> Dict[str, torch.Tensor]:
        r"""
        Takes in the information on the min and max values of the inputs and weights and:
            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)

        Args:
            input_info (dict): A dict mapping each observer to input range information
            weight_info (dict): A dict mapping each observer to weight range information

        Returns a dict mapping relavent observer fqns (str) to a 1-D tensor.
            Each value is a different s_c value for a different channel
        """
        # create return dictionary for each observer
        module_fqn_to_channel: Dict[str, torch.Tensor] = {}

        # for each module (both passed in dicts should have same keys)
        for module_fqn in input_info:

            # raise error if not in weight info
            if module_fqn not in weight_info:
                raise KeyError("Both input_info and weight_info should have same keys")

            # calculate the ratios of the weight info and input info
            weight_ratio = self._calculate_range_ratio(weight_info[module_fqn])
            input_ratio = self._calculate_range_ratio(input_info[module_fqn])

            # calculate the s metric per channel
            s = torch.sqrt(weight_ratio) / torch.sqrt(input_ratio)

            # add to dictionary
            module_fqn_to_channel[module_fqn] = s

        # return compiled observer ratios
        return module_fqn_to_channel

    def _generate_dict_info(self, input_info: Dict, weight_info: Dict, comp_stats: Dict) -> Dict[str, Dict]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            input_info (dict): A dict mapping each module to input range information
            weight_info (dict): A dict mapping each module to weight range information
            comp_stats (dict): A dict mapping each module to its corresponding comp stat

        Returns a dictionary mapping each module with relavent ModelReportObservers around them to:
            whether input weight equalization is recommended
            their s_c metric compared to the threshold
            the threshold used to make the recommendation
            the channel used for recording data
            the input channel range info
            the weight channel range info
        """
        # store modules input weight equalization info
        input_weight_equalization_info: Dict[str, Dict] = {}

        # for each module we add seperate set of suggestions
        for module_fqn in input_info:

            # get relavent info for this module
            mod_input_info: Dict = input_info[module_fqn]
            mod_weight_info: Dict = weight_info[module_fqn]
            mod_comp_stat: Dict = comp_stats[module_fqn]

            # decide if each channel should have input weight equalization or not
            channel_rec_vals: list = []

            for val in mod_comp_stat:
                float_rep: float = val.item()

                # decide if recommending input weight equalization
                recommended: bool = float_rep >= self.ratio_threshold and float_rep <= 1 / self.ratio_threshold
                channel_rec_vals.append(recommended)

            # build the return dict input
            input_weight_equalization_info[module_fqn] = {
                self.RECOMMENDED_KEY: channel_rec_vals,
                self.COMP_METRIC_KEY: mod_comp_stat,
                self.THRESHOLD_KEY: self.ratio_threshold,
                self.CHANNEL_KEY: self.ch_axis,
                self.INPUT_INFO_KEY: mod_input_info,
                self.WEIGHT_INFO_KEY: mod_weight_info,
            }

        # return our compiled info for each module
        return input_weight_equalization_info

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        r"""
        Determines whether input weight equalization is appropriate for a given module.

        Takes advantage of the ModelReport Observer which records per channel information of input range
        It then uses the passed in weight info inconjunction to compute the desired ratio
        Finally, it gives suggestions based on this information for each module of interest

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers
            weight_info (Dict): Maps modules of interest to information on their weights to be analyzed

        Returns a tuple with two elements:
            String report of of whether input weight equalization is recommended for certain modules
            Dictionary mapping modules of interst to:
                whether input weight equalization is recommended
                their s_c metric compared to the threshold
                the threshold used to make the recommendation
                the channel used for recording data
                the input channel range info
                the weight channel range info
        """

        # find the range of inputs
        input_values: Dict[str, Dict] = self._extract_input_info(model)

        # find the range of weights
        weight_values: Dict[str, Dict] = self._extract_weight_info(model)

        # calculate per_channel comparision statistic s_c
        comp_stats: Dict[str, torch.Tensor] = self._generate_comparision_values(input_values, weight_values)

        # generate the return dictionary
        input_weight_equalization_info: Dict[str, Dict] = self._generate_dict_info(input_values, weight_values, comp_stats)

        # now we can generate report based on this information
        input_weight_string = "Input-Weight Equalization suggestions: \n"

        # some strings to be formatted depending on module we are adding
        module_suggestion_str = "For Module {} looked at with axis {} we suggest: \n"
        channel_suggestion_str = "\tFor channel {}, we suggest {} input weight equalization because {}\n"
        use_str = "to use"
        no_use_str = "to not use"
        input_weight_benefit_str = "we expect significant reduction in quantization error."
        input_weight_non_benefit_reasoning = "the scales of the input vs. weight with regards to their ranges."
        input_weight_non_benefit_str = "we don't expect much improvement from input-weight equalization based on {}"

        # compile the suggestion string
        for module_fqn in input_weight_equalization_info:
            # add the module level description
            input_weight_string += module_suggestion_str.format(module_fqn, self.ch_axis)

            mod_info: Dict[str, Any] = input_weight_equalization_info[module_fqn]

            # look at each individual channel and add a suggestion
            for index, channel_suggested in enumerate(mod_info[self.RECOMMENDED_KEY]):
                if channel_suggested:
                    channel_str = channel_suggestion_str.format(index, use_str, input_weight_benefit_str)
                    input_weight_string += channel_str
                else:
                    non_benefit_str = input_weight_non_benefit_str.format(input_weight_non_benefit_reasoning)
                    channel_str = channel_suggestion_str.format(index, no_use_str, non_benefit_str)
                    input_weight_string += channel_str

        # return a tuple with the string explanation and the compiled dict info
        return (input_weight_string, input_weight_equalization_info)
