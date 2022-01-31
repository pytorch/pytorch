import torch.utils.data.graph_settings


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    num_workers = info.num_workers
    datapipe = info.dataset
    torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)
