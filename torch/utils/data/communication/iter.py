import time
import types

from torch.utils.data import IterDataPipe, communication

DEFAULT_NON_BLOCKING_SLEEP = 0.001


def default_not_available_hook():
    time.sleep(DEFAULT_NON_BLOCKING_SLEEP)


class NotAvailable(Exception):
    pass


class InvalidStateResetRequired(Exception):
    """
        Returned by DataPipe when it is expecting to get reset request,
        for example RouterDataPipe expecting all workers to request reset'
    """
    pass


class NonBlocking(IterDataPipe):
    not_available_hook = default_not_available_hook

    def __iter__(self):
        self.reset_iterator()
        return self

    def __next__(self):
        while True:
            try:
                return self.nonblocking_next()
            except StopIteration:
                raise StopIteration
            except NotAvailable:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def nonblocking_next(self):
        raise NotImplementedError(
            "nonblocking_next is not implemented for %s" % self.__class__)

    def reset_iterator(self):
        raise NotImplementedError(
            "reset_iterator is not implemented for %s" % self.__class__)

    @staticmethod
    def register_not_available_hook(hook_function):
        NonBlocking.not_available_hook = hook_function


def EnsureNonBlockingDataPipe(validated_datapipe):
    if not isinstance(validated_datapipe, IterDataPipe):
        raise Exception('Not Iterable DataPipe ' +
                        str(validated_datapipe.__class__))
    if isinstance(validated_datapipe, NonBlocking):
        return validated_datapipe
    if not hasattr(validated_datapipe, '_as_iterator'):
        validated_datapipe._as_iterator = None  # type: ignore[attr-defined]
    if not hasattr(validated_datapipe, 'nonblocking_next'):
        def nonblocking_next(self):
            if self._as_iterator is None:
                self._as_iterator = iter(self)
            return next(self._as_iterator)
        validated_datapipe.nonblocking_next = types.MethodType(  # type: ignore[attr-defined]
            nonblocking_next, validated_datapipe)
    if not hasattr(validated_datapipe, 'reset_iterator'):
        def reset_iterator(self):
            self._as_iterator = None
        validated_datapipe.reset_iterator = types.MethodType(  # type: ignore[attr-defined]
            reset_iterator, validated_datapipe)
    return validated_datapipe


def DataPipeBehindQueues(source_datapipe, protocol, full_stop=False, blocking_request_get=False):
    """
        Indefinitely iterates over req_queue and passing values from source_datapipe to res_queue
        If raise_stop is true, raises exception when StopIteration received from the source_datapipe
    """
    if not isinstance(protocol, communication.protocol.IterDataPipeQueueProtocolServer):
        raise Exception('Expecting IterDataPipeQueueProtocolServer, got', protocol)
    source_datapipe = EnsureNonBlockingDataPipe(source_datapipe)
    forever = True
    while forever:
        try:
            # Non-blocking call is Extremely slow here for python.mp, need to figure out a good workaround
            request = protocol.get_new_request(block=blocking_request_get)
        except communication.protocol.EmptyQueue:
            yield True
            continue

        if isinstance(request, communication.messages.ResetIteratorRequest):
            source_datapipe.reset_iterator()
            protocol.response_reset_iterator()

        elif isinstance(request, communication.messages.TerminateRequest):
            forever = False
            protocol.response_terminate()

        elif isinstance(request, communication.messages.GetNextRequest):
            while forever:
                try:
                    value = source_datapipe.nonblocking_next()
                except NotAvailable:
                    yield True
                    continue
                except StopIteration:
                    protocol.response_stop_iteration()
                    if full_stop:
                        forever = False
                    else:
                        yield True
                    break
                except InvalidStateResetRequired:
                    protocol.response_invalid_state()
                    if full_stop:
                        forever = False
                    else:
                        yield True
                    break
                protocol.response_next(value)
                yield True  # Returns control
                break
        else:
            raise Exception('Unrecognized type of request received', request)


class QueueWrapper(NonBlocking):
    """
        Creates iter.DataPipe which reads data from the DataLoader.Queue
    """

    def __init__(self, protocol, response_wait_time=0.00001):
        if not isinstance(protocol, communication.protocol.IterDataPipeQueueProtocolClient):
            raise Exception('Got', protocol)
        self.protocol = protocol
        self.counter = 0
        self._stop_iteration = False
        self._response_wait_time = response_wait_time

    def reset_iterator(self):
        self._stop_iteration = False
        self.counter = 0
        self.protocol.request_reset_iterator()
        while True:
            try:
                self.protocol.get_response_reset_iterator()
                break
            except communication.protocol.EmptyQueue:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def nonblocking_next(self):
        if self._stop_iteration:
            raise Exception(
                '`next` or `nonblocking_next` called after receiving StopIteration')
        if self.protocol.can_take_request():
            self.protocol.request_next()
        try:
            response = self.protocol.get_response_next(block=True, timeout=self._response_wait_time)
        except communication.protocol.EmptyQueue:
            raise NotAvailable
        if isinstance(response, communication.messages.StopIterationResponse):
            self._stop_iteration = True
            raise StopIteration
        if isinstance(response, communication.messages.InvalidStateResponse):
            raise NotAvailable
        return response.value
