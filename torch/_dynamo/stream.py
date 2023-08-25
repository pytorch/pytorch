import torch
import torch.cuda
import warnings


def __singleton(cls):
    instance = {}

    def get_instance(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]
    return get_instance


# This class is used to contain the provided stream methods for dynamo to 
# capture the stream usage.
@__singleton
class StreamMethodContainer:
    def __init__(self) -> None:
        self.current_stream_method = {}
        self.stream_class_method = {}
        self.create_stream_context_method = {}
        self.set_stream_method = {}
        self.set_stream_by_id_method = {}


    def __init_subclass__(cls):
        raise TypeError("class StreamMethodContainer can not be inherited")


    def __get(self, container: str, device: str):
        assert device in getattr(self, container).keys(), "unknown device {device}"
        if getattr(self, container)[device] is None:
            warnings.warn(
                "no method is found for " + container + " and " + device
            )
        return getattr(self, container)[device]


    def __get_all(self, container: str) -> list:
        ret_method = []
        for device_key in getattr(self, container).keys():
            ret_method.append(getattr(self, container)[device_key])
        return ret_method


    def __register_stream_method(self, container: str, device: str, method):
        getattr(self, container)[device] = method


    def get_all_methods(self, container: str):
        if container in [
            'set_stream', 'set_stream_by_id', 'current_stream', 'stream_class', 'create_stream_context'
        ]:
            return self.__get_all(container + '_method')
        else:
            raise RuntimeError(f"Unknown stream method '{container}'")


    def get_method_by_device(self, container: str, device: str):
        return self.__get(container, device)


# the global instance to contain the stream methods
StreamMethodObject = StreamMethodContainer()


# Here are some APIs for developers to resgiter their stream methods, which are needed to 
# align with the specific semantics.
def register_stream_method(device: str, method_args_dict: dict):
    for key in method_args_dict.keys():
        StreamMethodObject.__register_stream_method(key + '_method', device, method_args_dict[key])


# A dict with specific semantics and associated method is required for register.
# The key in the dict represents the fixed semantics and cannot be changed, the value paired
# with the key is the associated method or class for this semantics
#
# * device_stream_method = {'current_stream': method_1,
# *                         'stream_class': method_2,
# *                         'create_stream_context': method_3,
# *                         'set_stream': method_4,
# *                         'set_stream_by_id': method_5}
#
# If the method to a specific semantics are not defined or implemented, please
# pass 'None'. When finish creating the methods dict, register it by using following API:
#
# * torch._dynamo.stream.register_stream_method(device_name, device_stream_method)
#
# Below is the cuda register:
if torch.cuda.is_available():
    cuda_stream_method = {'current_stream': torch.cuda.current_stream,
                          'stream_class': torch.cuda.streams.Stream,
                          'create_stream_context': torch.cuda.stream,
                          'set_stream': torch.cuda.set_stream,
                          'set_stream_by_id': torch.cuda.set_stream_by_id}
    register_stream_method('cuda', cuda_stream_method)
