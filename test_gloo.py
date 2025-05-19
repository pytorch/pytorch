import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np


def exec_op(rank):
    dist.init_process_group(backend='gloo', rank=rank, world_size=2, init_method=f'tcp://127.0.0.1:40001')
    np.random.seed(1024 + rank)
    torch.manual_seed(1024 + rank)
    x = np.random.uniform(-65504, 65504, [m, k]).astype(np.float16)
    x = torch.from_numpy(x)
    # print(x.dtype)
    # x = torch.randn(m, k, dtype=torch.float16)
    # print(x.dtype)
    print(f"rank:{rank} before all_reduce x[7205]:{x[7205]}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"rank:{rank} after all_reduce x[7205]:{x[7205]}")


if __name__ == '__main__':
    m, k = [24063328, 1]
    p_list = []
    for g_rank in range(2):
        p = Process(target=exec_op, args=(g_rank,))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
