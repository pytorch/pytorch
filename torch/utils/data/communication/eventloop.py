import torch
import threading
import pickle

from torch.utils.data import IterDataPipe, communication


def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue):
    if isinstance(source_datapipe, IterDataPipe):
        pipe_type = communication.iter
        protocol_type = communication.protocol.IterDataPipeQueueProtocolServer
    else:
        raise Exception('Only supports IterDataPipe, got', source_datapipe)
        # pipe_type = communication.map
        # protocol_type = communication.protocol.MapDataPipeQueueProtocolServer

    torch.set_num_threads(1)
    for _ in pipe_type.DataPipeBehindQueues(source_datapipe, protocol_type(req_queue, res_queue), blocking_request_get=True):
        pass


def SpawnProcessForDataPipeline(multiprocessing_ctx, datapipe):
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue))
    return process, req_queue, res_queue


def SpawnThreadForDataPipeline(datapipe):
    req_queue = communication.queue.ThreadingQueue()
    res_queue = communication.queue.ThreadingQueue()

    try:
        new_datapipe = pickle.loads(pickle.dumps(datapipe))
    except Exception as e:
        raise Exception('Unable to pickle DataPipe to make thread local copy', e)

    process = threading.Thread(target=DataPipeToQueuesLoop, args=(
        new_datapipe, req_queue, res_queue), daemon=True)
    return process, req_queue, res_queue, new_datapipe
