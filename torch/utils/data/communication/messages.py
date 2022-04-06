class DataLoaderQueueMessage(object):
    pass


class Request(DataLoaderQueueMessage):
    pass


class Response(DataLoaderQueueMessage):
    pass


class ResetIteratorRequest(Request):
    pass


class ResetIteratorResponse(Response):
    pass


class TerminateRequest(Request):
    pass


class TerminateResponse(Response):
    pass


class LenRequest(Request):
    pass


class LenResponse(Response):
    __slots__ = ('len')

    def __init__(self, len):
        self.len = len


class GetItemRequest(Request):
    __slots__ = ('key')

    def __init__(self, key):
        self.key = key


class GetItemResponse(Response):
    __slots__ = ('key', 'value')

    def __init__(self, key, value):
        self.key = key
        self.value = value


class GetNextRequest(Request):
    pass


class GetNextResponse(Response):
    __slots__ = ('value')

    def __init__(self, value):
        self.value = value


class StopIterationResponse(Response):
    pass


class InvalidStateResponse(Response):
    """
        Returned by DataPipe when it is expecting to get reset request,
        for example RouterDataPipe expecting all workers to request reset'
    """
    pass
