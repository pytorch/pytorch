import torch
import os
import time
# import pandas as pd
import torch.cuda
import torch.nn
import torch.autograd
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.algorithms.quantization.quantization as quant
from torch.distributed.algorithms.quantization.quantization import DQuantType


def run_quantize_collective(rank, size, output_tensor, input_tensor, tensorSize, qtype, collective):
    group = dist.new_group(range(size))
    if collective == 'all_gather':
        quantize_collective = quant.auto_quantize(dist.all_gather, qtype)
    else:
        quantize_collective = quant.auto_quantize(dist.all_to_all, qtype)
        quantize_collective(output_tensor, input_tensor, group=group, async_op=False)

def run_collective(rank, size, output_tensor, input_tensor, tensorSize, qtype, collective):
    group = dist.new_group(range(size))
    if collective == 'all_gather':
        dist.all_gather(output_tensor, input_tensor, group=group, async_op=False)
    else:
        dist.all_to_all(output_tensor, input_tensor, group=group, async_op=False)

def init_process(rank, size, output_tensor, input_tensor, tensorSize, qtype, collective, backend, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, output_tensor, input_tensor, tensorSize, qtype, collective)

def _build_tensor(size, value=None, dtype=torch.float32, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, dtype=dtype).fill_(value).cuda(device_id)

if __name__ == "__main__":
    world_size = 2
    tensorSizes = [[5, 5], [10, 10], [50, 50]]
    iterations = 10
    mp.set_start_method("spawn")
    collectives = ['all_gather', 'all_to_all']
    qtypes = [DQuantType.FP16, DQuantType.BFP16]
    row = []

    print('Collective op |Quantization Method|Tensor size     |Latency')
    for collective in collectives:
        backend = "gloo" if collective == 'all_gather' else 'nccl'
        for qtype in qtypes:
            for tensorSize in tensorSizes:
                avg = 0
                for _ in range(iterations):
                    processes = []
                    start = 0
                    end = 0
                    for dest in range(world_size):
                        input_torch = _build_tensor(tensorSize, 1 + 2 * dest, dtype=torch.float32)
                        output_tensor = [_build_tensor(tensorSize, -1, dtype=torch.float32) for _ in range(world_size)]
                        p = mp.Process(target=init_process,
                                       args=(dest,
                                             world_size,
                                             output_tensor,
                                             input_torch,
                                             tensorSize,
                                             qtype,
                                             collective,
                                             backend,
                                             run_quantize_collective))
                        if (dest == 0):
                            start = time.time()
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
                        if (p == processes[0]):
                            end = time.time()
                    avg = avg + (end - start)
                avg = avg / iterations
                # row.append([collective, qtype, tensorSize, avg])
                print(collective, "  ", tensorSize, "             ", qtype.value, "      ", avg)


        for tensorSize in tensorSizes:
            avg = 0
            for _ in range(iterations):
                processes = []
                start = 0
                end = 0
                for dest in range(world_size):
                    input_torch = _build_tensor(tensorSize, 1 + 2 * dest, dtype=torch.float32)
                    output_tensor = [_build_tensor(tensorSize, -1, dtype=torch.float32) for _ in range(world_size)]
                    p = mp.Process(target=init_process,
                                   args=(dest,
                                         world_size,
                                         output_tensor,
                                         input_torch,
                                         tensorSize,
                                         'None',
                                         collective,
                                         backend,
                                         run_collective))
                    if (dest == 0):
                        start = time.time()
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                    if (p == processes[0]):
                        end = time.time()
                avg = avg + (end - start)
            avg = avg / iterations
            print(collective, "  ", tensorSize, "             ", "None", "           ", avg)
    #         row.append([collective, 'None', tensorSize, avg])
    # df = pd.DataFrame(row, columns=['Collective op', 'Quantization Method', 'Tensor size', 'Latency'])
    # print(df.keys())
    # print(df.values)
