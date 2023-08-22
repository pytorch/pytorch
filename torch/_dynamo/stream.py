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
class StreamAPIContainer:
    def __init__(self) -> None:
        '''
        set_stream_method: {'cuda': [torch.cuda.set_stream]}
        set_stream_by_id_method: {'cuda': [torch.cuda.set_stream_by_id]}
        current_stream_method: {'cuda': [torch.cuda.current_stream]}
        create_stream_method: {'cuda': [torch.cuda.streams.Stream]}
        create_stream_context_method: {'cuda': [torch.cuda.stream]}
        '''
        # {'cuda', torch.cuda.current_stream}
        self.current_stream_method = {}
        # {'cuda', torch.cuda.streams.Stream}
        self.create_stream_method = {}
        # {'cuda', torch.cuda.streams.Stream}
        self.create_stream_context_method = {}
        # {'cuda', torch.cuda.set_stream}
        self.set_stream_method = {}
        # {'cuda', torch.cuda.set_stream_by_id}
        self.set_stream_by_id_method = {}


    def __init_subclass__(cls):
        raise TypeError("class StreamAPIContainer can not be inherited")


    def __register(self, container: str, device: str, *method_args) -> None:
        if len(method_args) <= 0:
            return
        if device not in getattr(self, container).keys():
            getattr(self, container)[device] = []
        for method in method_args:
            if method is not None:
                getattr(self, container)[device].append(method)


    def __get(self, container: str, device: str):
        assert device in getattr(self, container).keys(), "unknown device to get stream methods"
        assert len(getattr(self, container)[device]) == 1, "ambiguous functions for this method"
        return getattr(self, container)[device][0]


    def __get_all(self, container: str) -> list:
        ret_method = []
        for device_key in getattr(self, container).keys():
            ret_method.append(getattr(self, container)[device_key])
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


    def get_all_current_stream_method(self):
        return self.__get_all('current_stream_method')


    def get_all_create_stream_method(self):
        return self.__get_all('create_stream_method')


    def get_all_create_stream_context_method(self):
        return self.__get_all('create_stream_context_method')


    def get_all_set_stream_method(self):
        return self.__get('set_stream_method')


    def get_current_stream_method(self, device: str):
        return self.__get('current_stream_method', device)


    def get_create_stream_method(self, device: str):
        return self.__get('create_stream_method', device)


    def get_create_stream_context_method(self, device: str):
        return self.__get('create_stream_context_method', device)


    def get_set_stream_method(self, device: str):
        return self.__get('set_stream_method', device)


    def get_set_stream_by_id_method(self, device: str):
        return self.__get('set_stream_by_id_method', device)
