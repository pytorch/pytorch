from utils import process_bucket_with_remote_server

from .DdpTrainer import DdpTrainer


class DdpSparseRpcTrainer(DdpTrainer):

    def __init__(self, rank, trainer_count, process_group, use_cuda_rpc, server_rref, backend, epochs):
        r"""
        A trainer that implements a DDP training algorithm using a server and process group
        allreduce. The trainer sends sparse gradients using RPC, and the server averages and
        returns the gradients. The process group uses the backend allreduce implementation
        to average the dense gradients.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainer in the world
            process_group (ProcessGroup): distributed process group
            use_cuda_rpc (bool): indicator for CUDA RPC
            server_rref (RRef): remote reference to the server
            backend (str): distributed communication backend
            epochs (int): epoch count for training
        """
        super().__init__(rank, trainer_count, process_group, use_cuda_rpc, server_rref, backend, epochs)

    @staticmethod
    def hook(state, bucket):
        r"""
        A ddp communication hook that uses the current backend allreduce
        implementation for dense tensors and a server for sparse tensors.
        Args:
            state (object): maintains state during the training process
            bucket (GradBucket): gradient bucket
        """
        tensor = bucket.get_tensor()
        if tensor.is_sparse:
            return process_bucket_with_remote_server(state, bucket)
        else:
            cref = state.cref
            tensor = [tensor / state.process_group.size()]
            key = state.get_key(bucket.get_index())
            cref.record_hook_fut_start(key, f"{cref.backend}_dense_allreduce")
            fut = state.process_group.allreduce(tensor).get_future()

            def callback(fut):
                cref.record_hook_fut_end(key)
                return fut.wait()

            return fut.then(callback)

    def get_hook(self):
        r"""
        returns DdpSparseRpcTrainer.hook
        """
        return DdpSparseRpcTrainer.hook
