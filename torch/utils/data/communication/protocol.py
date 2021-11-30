from torch.utils.data import communication


class Protocol(object):
    __slots__ = ('request_queue', 'response_queue')

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue


class ProtocolClient(Protocol):
    """
        ProtocolClient takes charge of putting requests into req_queue and returning results from res_queue.
    """
    _req_sent = None

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_sent = None

    def can_take_request(self):
        return self._req_sent is None

    def waiting_for_response(self):
        return self._req_sent is not None

    def request_sent(self, request=True):
        if not self.can_take_request():
            raise Exception('Protocol only supports one request in the Queue')
        self._req_sent = request

    def request_served(self, result=None):
        if not self.waiting_for_response():
            raise Exception(
                'Expected no peding requests, but something got served', result)
        self._req_sent = None


class ProtocolServer(Protocol):
    """
        ProtocolServer takes charge of getting requests from req_queue and fetching data from source datapipe.
    """
    _req_received = None

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_received = None

    def have_pending_request(self):
        return self._req_received is not None

    def get_new_request(self, block=False):
        if self.have_pending_request():
            raise Exception(
                'Trying to get next request, while having one unserved')
        try:
            response = self.request_queue.get(block=block)
        except Exception as e:  # TODO: Catch only timeout exceptions
            raise EmptyQueue('queue is empty')
        self._req_received = response
        return response

        # TODO: Validate supported requests

    def response_reset(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        if not isinstance(self._req_received, communication.messages.ResetIteratorRequest):
            raise Exception(
                "Replaying with reset status to other type of message")
        self.response_queue.put(communication.messages.ResetIteratorResponse())
        self._req_received = None

    def response_next(self, value):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.GetNextResponse(value))
        self._req_received = None

    def response_stop(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.StopIterationResponse())
        self._req_received = None

    def response_invalid(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.InvalidStateResponse())
        self._req_received = None

    def response_terminate(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        if not isinstance(self._req_received, communication.messages.TerminateRequest):
            raise Exception(
                "Replaying with terminate status to other type of message")
        self.response_queue.put(communication.messages.TerminateResponse())
        self._req_received = None


class MapDataPipeQueueProtocolClient(ProtocolClient):
    pass


class MapDataPipeQueueProtocolServer(ProtocolServer):
    pass


class EmptyQueue(Exception):
    pass


class IterDataPipeQueueProtocolServer(ProtocolServer):
    pass


class IterDataPipeQueueProtocolClient(ProtocolClient):
    def request_reset(self):
        if not self.can_take_request():
            raise Exception(
                'Can not reset while we are still waiting response for previous request')
        request = communication.messages.ResetIteratorRequest()
        self.request_queue.put(request)
        self.request_sent(request)

    def request_next(self):
        if not self.can_take_request():
            raise Exception(
                'Can not request next item while we are still waiting response for previous request')
        request = communication.messages.GetNextRequest()
        self.request_queue.put(request)
        self.request_sent(request)

    def get_response_reset(self, block=False):
        try:
            response = self.response_queue.get(block=block)
        except Exception as e:  # TODO: Catch only timeout exceptions
            raise EmptyQueue('queue is empty')
        self.request_served(response)

        if not isinstance(response, communication.messages.ResetIteratorResponse):
            raise Exception('Invalid response received')

    def get_response_next(self, block=False, timeout=None):
        if not self.waiting_for_response():
            raise Exception(
                'Can not expect any response without submitted request')
        try:
            response = self.response_queue.get(block=block, timeout=timeout)
        except Exception as e:  # TODO: Catch only timeout exceptions
            raise EmptyQueue('queue is empty')
        self.request_served(response)

        # TODO(VitalyFedyunin): Add possible response types validation here
        return response
