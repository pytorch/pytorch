import torch
import threading
from torch.utils.data import IterDataPipe
import torch.utils.data.communication.queue as queue
import torch.utils.data.communication.iter as iter_datapipe
import torch.utils.data.communication.protocol as datapipes_protocol


def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue):
    if isinstance(source_datapipe, IterDataPipe):
        pipe_type = iter_datapipe
        protocol_type = datapipes_protocol.IterDataPipeQueueProtocolServer
    else:
        raise Exception('Only supports IterDataPipe, got', source_datapipe)
        # pipe_type = datapipes.map
        # protocol_type = datapipes.protocol.MapDataPipeQueueProtocolServer

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
    req_queue = queue.ThreadingQueue()
    res_queue = queue.ThreadingQueue()

    process = threading.Thread(target=DataPipeToQueuesLoop, args=(
        datapipe, req_queue, res_queue))
    return process, req_queue, res_queue
