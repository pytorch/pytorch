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
        self.create_stream_method = {}
        self.create_stream_context_method = {}
        self.set_stream_method = {}
        self.set_stream_by_id_method = {}


    def __init_subclass__(cls):
        raise TypeError("class StreamMethodContainer can not be inherited")


    def __register(self, container: str, device: str, method_args) -> None:
        if len(method_args) <= 0:
            return
        if device not in getattr(self, container).keys():
            getattr(self, container)[device] = []
        for method in method_args:
            getattr(self, container)[device].append(method)


    def __get(self, container: str, device: str):
        assert device in getattr(self, container).keys(), "unknown device {device}"
        assert len(getattr(self, container)[device]) == 1, "ambiguous methods found for {container} and {device}"
        if getattr(self, container)[device][0] is None:
            warnings.warn(
                "get a None method for " + container + " and " + device
            )
        return getattr(self, container)[device][0]


    def __get_all(self, container: str) -> list:
        ret_method = []
        for device_key in getattr(self, container).keys():
            ret_method.append(getattr(self, container)[device_key][0])
        return ret_method


    def register_stream_method(self, container: str, device: str, method_args):
        self.__register(container, device, method_args)


    def get_all_methods(self, container: str):
        if container in [
            'set_stream', 'set_stream_by_id', 'current_stream', 'create_stream', 'create_stream_context'
        ]:
            return self.__get_all(container + '_method')
        else:
            raise RuntimeError(f"Unknown stream method '{container}'")


    def get_method_by_device(self, container: str, device: str):
        return self.__get(container, device)


StreamMethodObject = StreamMethodContainer()


# Here are some APIs for developers to resgiter their stream methods, which are needed to 
# align with the specific semantics.
def register_current_stream_method(device: str, *method_args):
    StreamMethodObject.register_stream_method('current_stream_method', device, method_args)


def register_create_stream_method(device: str, *method_args):
    StreamMethodObject.register_stream_method('create_stream_method', device, method_args)


def register_create_stream_context_method(device: str, *method_args):
    StreamMethodObject.register_stream_method('create_stream_context_method', device, method_args)


def register_set_stream_method(device: str, *method_args):
    StreamMethodObject.register_stream_method('set_stream_method', device, method_args)


def register_set_stream_by_id_method(device: str, *method_args):
    StreamMethodObject.register_stream_method('set_stream_by_id_method', device, method_args)


# For backend developers, it is needed to register their stream usage by using:
# 
# * torch._dynamo.stream.register_current_stream_method(device, stream_method)
# 
# the stream_method is needed to align the semantics of torch.cuda.current_stream,
# which returns a current using stream.
# If there is no such method, please explicitly register None
#
# * torch._dynamo.stream.register_current_stream_method(device, None)
#
# Here register 5 CUDA stream methods for stream capture in dynamo
if torch.cuda.is_available():
    register_current_stream_method('cuda', torch.cuda.current_stream)
    register_create_stream_method('cuda', torch.cuda.streams.Stream)
    register_create_stream_context_method('cuda', torch.cuda.stream)
    register_set_stream_method('cuda', torch.cuda.set_stream)
    register_set_stream_by_id_method('cuda', torch.cuda.set_stream_by_id)
