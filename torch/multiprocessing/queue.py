import multiprocessing
from itertools import chain

import torch
from .common import CustomizablePicklingQueue, reduce_torch_object


class Queue(CustomizablePicklingQueue):
    def __init__(self, context=None, reducers=None):
        if context is None:
            context = multiprocessing.get_context()
        if reducers is None:
            reducers = {}

        for t in chain(torch._tensor_classes, torch._storage_classes):
            reducers[t] = reduce_torch_object

        super(Queue, self).__init__(context, reducers)


