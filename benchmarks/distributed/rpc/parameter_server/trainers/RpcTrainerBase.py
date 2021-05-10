import torch
import torch.distributed.rpc as rpc

from .TrainerBase import TrainerBase


class RpcTrainerBase(TrainerBase):

    RPC_FUT_METRIC = "rpc_fut_metric"
    RPC_FUT = "rpc_fut"
    RPC_METRIC = "rpc_metric"
    RPC = "rpc"

    def __init__(self, rank):
        super().__init__(rank)

    def send_async_request(self, key, rref, server_method, *args, cuda=True):
        self.record_rpc_fut_start(key, f"{self.RPC_FUT}_{server_method.__name__}", cuda)
        fut = rpc.rpc_async(
            rref.owner(),
            server_method,
            args=args
        )

        def request_time_callback(fut):
            self.record_rpc_fut_end(key)
            return fut.wait()

        return fut.then(request_time_callback)

    def send_sync_request(self, key, rref, server_method, *args, cuda=True):
        self.record_rpc_fut_start(key, f"{self.RPC}_{server_method.__name__}", cuda)
        result = rpc.rpc_sync(
            rref.owner(),
            server_method,
            args=args
        )
        self.record_rpc_fut_end(key)
        return result

    def record_rpc_fut_start(self, key, name, cuda=True):
        self.record_start(self.RPC_FUT_METRIC, key, name, cuda)

    def record_rpc_fut_end(self, key):
        self.record_end(self.RPC_FUT_METRIC, key)

    def sparse_tensor_to_rpc_format(self, sparse_tensor):
        sparse_tensor = sparse_tensor.coalesce()
        return [sparse_tensor.indices(), sparse_tensor.values(), torch.tensor(sparse_tensor.size())]

    def sparse_rpc_format_to_tensor(self, sparse_rpc_format):
        return torch.sparse_coo_tensor(
            sparse_rpc_format[0], sparse_rpc_format[1], torch.Size(sparse_rpc_format[2])
        ).coalesce()
