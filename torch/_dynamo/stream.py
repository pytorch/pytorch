import torch
import torch.cuda


def __singleton(cls):
    instance = {}

    def get_instance(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]
    return get_instance


@__singleton
class StreamMethodContainer:
    def __init__(self) -> None:
        '''
        set_stream_method: {'cuda': [torch.cuda.set_stream]}
        set_stream_by_id_method: {'cuda': [torch.cuda.set_stream_by_id]}
        current_stream_method: {'cuda': [torch.cuda.current_stream]}
        create_stream_method: {'cuda': [torch.cuda.streams.Stream]}
        create_stream_context_method: {'cuda': [torch.cuda.stream]}
        '''
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
            if method is not None:
                getattr(self, container)[device].append(method)


    def __get(self, container: str, device: str):
        assert hasattr(self, container), "unknown container to get stream methods"
        assert device in getattr(self, container).keys(), "unknown device to get stream methods"
        assert len(getattr(self, container)[device]) == 1, "ambiguous functions for this method"
        return getattr(self, container)[device][0]


    def __get_all(self, container: str) -> list:
        assert hasattr(self, container), "unknown container to get stream methods"
        ret_method = []
        for device_key in getattr(self, container).keys():
            ret_method.append(getattr(self, container)[device_key][0])
        return ret_method


    def register_current_stream_method(self, device: str, *method_args):
        self.__register('current_stream_method', device, method_args)


    def register_create_stream_method(self, device: str, *method_args):
        self.__register('create_stream_method', device, method_args)


    def register_create_stream_context_method(self, device: str, *method_args):
        self.__register('create_stream_context_method', device, method_args)


    def register_set_stream_method(self, device: str, *method_args):
        self.__register('set_stream_method', device, method_args)


    def register_set_stream_by_id_method(self, device: str, *method_args):
        self.__register('set_stream_by_id_method', device, method_args)


    def get_all_methods(self, container: str):
        return self.__get_all(container)


    def get_one_method(self, container: str, device: str):
        return self.__get(container, device)


if torch.cuda.is_available():
    # register stream API for dynamo stream capture
    StreamAPIObject = StreamMethodContainer()
    StreamAPIObject.register_create_stream_method('cuda', torch.cuda.streams.Stream)
    StreamAPIObject.register_create_stream_context_method('cuda', torch.cuda.stream)
    StreamAPIObject.register_current_stream_method('cuda', torch.cuda.current_stream)
    StreamAPIObject.register_set_stream_method('cuda', torch.cuda.set_stream)
    StreamAPIObject.register_set_stream_by_id_method('cuda', torch.cuda.set_stream_by_id)
