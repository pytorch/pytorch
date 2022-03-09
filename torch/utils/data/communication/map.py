import time
import types

from torch.utils.data import communication, MapDataPipe

DEFAULT_NON_BLOCKING_SLEEP = 0.001


def default_not_available_hook():
    time.sleep(DEFAULT_NON_BLOCKING_SLEEP)


class NotAvailable(Exception):
    pass


class NonBlockingMap(MapDataPipe):
    not_available_hook = default_not_available_hook

    def __getitem__(self, index):
        while True:
            try:
                return self.nonblocking_getitem(index)
            except NotAvailable:
                if NonBlockingMap.not_available_hook is not None:
                    NonBlockingMap.not_available_hook()

    def __len__(self):
        try:
            return self.nonblocking_len()
        except NotAvailable:
            if NonBlockingMap.not_available_hook is not None:
                NonBlockingMap.not_available_hook()

    def nonblocking_len(self):
        raise NotImplementedError(
            "nonblocking_len is not implemented for %s" % self.__class__)

    def nonblocking_getitem(self, index):
        raise NotImplementedError(
            "nonblocking_getitem is not implemented for %s" % self.__class__)

    @staticmethod
    def register_not_available_hook(hook_function):
        NonBlockingMap.not_available_hook = hook_function


def EnsureNonBlockingMapDataPipe(validated_datapipe):
    if not isinstance(validated_datapipe, MapDataPipe):
        raise Exception(f'Not Map DataPipe - got {validated_datapipe.__class__}')
    if isinstance(validated_datapipe, NonBlockingMap):
        return validated_datapipe
    if not hasattr(validated_datapipe, 'nonblocking_len'):
        def nonblocking_len(self):
            return self.__len__()
        validated_datapipe.nonblocking_len = types.MethodType(  # type: ignore[attr-defined]
            nonblocking_len, validated_datapipe)
    if not hasattr(validated_datapipe, 'nonblocking_getitem'):
        def nonblocking_getitem(self, index):
            return self.__getitem__(index)
        validated_datapipe.nonblocking_getitem = types.MethodType(  # type: ignore[attr-defined]
            nonblocking_getitem, validated_datapipe)
    return validated_datapipe


def DataPipeBehindQueues(source_datapipe, protocol, full_stop=False, blocking_request_get=False):
    """
        Indefinitely iterates over req_queue and passing values from source_datapipe to res_queue
        If raise_stop is true, raises exception when StopIteration received from the source_datapipe
    """
    if not isinstance(protocol, communication.protocol.MapDataPipeQueueProtocolServer):
        raise Exception('Expecting MapDataPipeQueueProtocolServer, got', protocol)
    source_datapipe = EnsureNonBlockingMapDataPipe(source_datapipe)
    forever = True
    while forever:
        try:
            # Non-blocking call is Extremely slow here for python.mp, need to figure out a good workaround
            request = protocol.get_new_request(block=blocking_request_get)
        except communication.protocol.EmptyQueue:
            yield True
            continue

        if isinstance(request, communication.messages.TerminateRequest):
            forever = False
            protocol.response_terminate()

        elif isinstance(request, communication.messages.LenRequest):
            size = source_datapipe.nonblocking_len()
            protocol.response_len(size)

        elif isinstance(request, communication.messages.GetItemRequest):
            while forever:
                try:
                    value = source_datapipe.nonblocking_getitem(request.key)
                except NotAvailable:
                    yield True
                    continue
                except IndexError as e:
                    # Alternatively, we can just allow the underlying DataPipe to throw an exception?
                    protocol.response_index_out_of_bound()
                    if full_stop:
                        forever = False
                    else:
                        yield True
                    break
                protocol.response_item(request.key, value)
                yield True  # Returns control
                break
        else:
            raise Exception('Unrecognized type of request received', request)


class QueueWrapperForMap(NonBlockingMap):
    """
        Creates map.DataPipe which reads data from the DataLoader.Queue
    """
    def __init__(self, protocol, response_wait_time=0.00001):
        if not isinstance(protocol, communication.protocol.MapDataPipeQueueProtocolClient):
            raise Exception('Got', protocol)
        self.protocol = protocol
        self.counter = 0
        self._stop_iteration = False
        self._response_wait_time = response_wait_time

    def nonblocking_getitem(self, index):
        if self._stop_iteration:
            raise Exception(
                '`getitem` or `nonblocking_getitem` called after receiving StopIteration')
        if self.protocol.can_take_request():
            self.protocol.request_item(index)
        try:
            response = self.protocol.get_response_item(block=True, timeout=self._response_wait_time)
        except communication.protocol.EmptyQueue:
            raise NotAvailable
        if isinstance(response, communication.messages.StopIterationResponse):
            self._stop_iteration = True
            raise IndexError(f"Index {index} is out of bound.")
        return response.key, response.value

    def nonblocking_len(self):
        if self._stop_iteration:
            raise Exception(
                '`len` or `nonblocking_len` called after receiving StopIteration')
        if self.protocol.can_take_request():
            self.protocol.request_len()
        try:
            response = self.protocol.get_response_len(block=True, timeout=self._response_wait_time)
        except communication.protocol.EmptyQueue:
            raise NotAvailable
        return response.len
