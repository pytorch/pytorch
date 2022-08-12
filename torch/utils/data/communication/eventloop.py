import torch
import threading
import pickle

from torch.utils.data import IterDataPipe, communication, MapDataPipe

try:
    import dill
    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

__all__ = [
    "DataPipeToQueuesLoop",
    "SpawnProcessForDataPipeline",
    "SpawnThreadForDataPipeline",
]

def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue):
    if isinstance(source_datapipe, IterDataPipe):
        pipe_type = communication.iter
        protocol_type = communication.protocol.IterDataPipeQueueProtocolServer
    elif isinstance(source_datapipe, MapDataPipe):
        pipe_type = communication.map  # type: ignore[misc]
        protocol_type = communication.protocol.MapDataPipeQueueProtocolServer  # type: ignore[assignment]
    else:
        raise Exception('Only supports IterDataPipe or MapDataPipe, got', source_datapipe)

    torch.set_num_threads(1)
    for _ in pipe_type.DataPipeBehindQueues(source_datapipe, protocol_type(req_queue, res_queue),
                                            blocking_request_get=True):
        pass


def SpawnProcessForDataPipeline(multiprocessing_ctx, datapipe):
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue))
    return process, req_queue, res_queue


def SpawnThreadForDataPipeline(datapipe):
    r"""
        Given a DataPipe, creates a copy of the DataPipe, starts a new Thread with DataPipeToQueuesLoop as target,
        and return the process, req_queue, res_queue, thread_local_datapipe.
    """
    req_queue = communication.queue.ThreadingQueue()
    res_queue = communication.queue.ThreadingQueue()

    try:
        new_datapipe = pickle.loads(pickle.dumps(datapipe))
    except Exception as pe:
        if HAS_DILL:
            try:
                new_datapipe = dill.loads(dill.dumps(datapipe))
            except Exception as de:
                raise Exception('Unable to dill DataPipe to make thread local copy', de)

        else:
            raise Exception('Unable to pickle DataPipe to make thread local copy (consider installing `dill`)', pe)

    process = threading.Thread(target=DataPipeToQueuesLoop, args=(
        new_datapipe, req_queue, res_queue), daemon=True)
    return process, req_queue, res_queue, new_datapipe
