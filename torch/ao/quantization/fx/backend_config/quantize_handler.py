import torch
from typing import Dict, Callable, Any, Optional
from .observation_type import ObservationType
from ..quantization_patterns import QuantizeHandler
from ..quantization_types import Pattern, NodePattern
from ...utils import (
    activation_dtype,
)
from torch.fx import Node

def get_quantize_handler_cls(
        observation_type,
        dtype_configs,
        num_tensor_args_to_observation_type,
        overwrite_output_fake_quantizer,
        overwrite_output_observer):

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(
                self,
                node_pattern: NodePattern,
                modules: Dict[str, torch.nn.Module],
                root_node_getter: Callable = None):
            super().__init__(node_pattern, modules, root_node_getter)
            if num_tensor_args_to_observation_type:
                assert self.num_tensor_args in num_tensor_args_to_observation_type, \
                    f"Must provide observation_type config for tensor number {self.num_tensor_args}" \
                    f" in num_tensor_args_to_observation_type for {node_pattern}"
                self.observation_type = num_tensor_args_to_observation_type[self.num_tensor_args]
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs
            self.overwrite_output_fake_quantizer = overwrite_output_fake_quantizer
            self.overwrite_output_observer = overwrite_output_observer

        def is_general_tensor_value_op(self) -> bool:
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

        # TODO: change this to output activation
        def get_activation_ctr(
                self,
                qconfig: Any,
                pattern: Pattern,
                is_training: bool,
        ) -> Optional[Callable]:
            """
            Returns the constructor for the activation observer which should be
            used for the pattern matched to this handler. Some handlers override
            this to a different value than what is specified in the qconfig.
            """
            act_dtype = activation_dtype(qconfig)
            # TODO: change to is_qat
            if is_training:
                if act_dtype == torch.quint8 and self.overwrite_output_fake_quantizer is not None:
                    return self.overwrite_output_fake_quantizer
            else:
                if act_dtype == torch.quint8 and self.overwrite_output_observer is not None:
                    return self.overwrite_output_observer
            return qconfig.activation


    return ConfigurableQuantizeHandler
