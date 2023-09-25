import warnings

import torch
import torch.cuda


# This class is used to contain the provided stream interface for dynamo to capture
class StreamInterface:
    def __init__(self) -> None:
        self.current_stream_method = {}
        self.create_stream_context_method = {}
        self.create_event_method = {}
        self.set_stream_method = {}
        self.set_stream_by_id_method = {}

    def __init_subclass__(cls):
        raise TypeError("class StreamInterface can not be inherit from")

    def __get(self, container: str, device: str):
        assert device in getattr(self, container).keys(), "unknown device {device}"
        if getattr(self, container)[device] is None:
            warnings.warn(
                "no stream interface is found for " + container + " and " + device
            )
        return getattr(self, container)[device]

    def __get_all(self, container: str) -> list:
        ret_interface = []
        for device_key in getattr(self, container).keys():
            ret_interface.append(getattr(self, container)[device_key])
        return ret_interface

    def register_method(self, container: str, device: str, interface):
        getattr(self, container)[device] = interface

    def get_all_methods(self, container: str):
        if container in [
            "set_stream",
            "set_stream_by_id",
            "current_stream",
            "create_stream_context",
            "create_event",
        ]:
            return self.__get_all(container + "_method")
        else:
            raise RuntimeError(f"Unknown stream interface '{container}'")

    def get_method_by_device(self, container: str, device: str):
        return self.__get(container + "_method", device)


# the global instance to contain the stream methods
StreamInterfaceObject = StreamInterface()


# Here are some APIs for developers to resgiter their stream methods, which are needed to
# align with the specific semantics.
def register_stream_interface(device: str, method_args_dict: dict):
    for key in method_args_dict.keys():
        StreamInterfaceObject.register_method(
            key + "_method", device, method_args_dict[key]
        )


# A dict with specific semantics and associated method is required for register.
# The key in the dict represents the fixed semantics and cannot be changed, the value paired
# with the key is the associated method or class for this semantics
#
# * device_stream_interface = {'current_stream': method_1,
# *                             'set_stream': method_2,
# *                             'create_stream_context': method_3,
# *                             'set_stream_by_id': method_4,
# *                             'create_event': method_5}
#
# If the method to a specific semantics are not defined or implemented, please
# pass 'None'. When finish creating the methods dict, register it by using following API:
#
# * torch._dynamo.stream.register_stream_interface(device_name, device_stream_interface)
#
# Below is the cuda stream interface register:
if torch.cuda.is_available():
    cuda_stream_interface = {
        "current_stream": torch.cuda.current_stream,
        "create_stream_context": torch.cuda.stream,
        "set_stream": torch.cuda.set_stream,
        "set_stream_by_id": torch.cuda.set_stream_by_id,
        "create_event": torch.cuda.Event,
    }
    register_stream_interface("cuda", cuda_stream_interface)
