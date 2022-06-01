import torch.utils.data.graph_settings
import warnings
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def worker_init_fn(worker_id):
    warnings.warn("Usage of backward_compatibility.worker_init_fn is deprecated"
                  " as DataLoader automatically calls it in every worker")


def private_worker_init_fn(worker_id):
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    total_workers = info.num_workers
    datapipe = info.dataset
    if dist.is_available() and c10d.is_initialized():
        total_workers *= dist.get_world_size()
        global_worker_id = dist.get_rank() * info.num_workers + global_worker_id
    torch.utils.data.graph_settings.apply_sharding(datapipe, total_workers, global_worker_id)
