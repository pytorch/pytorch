from .DdpRpcTrainer import DdpRpcTrainer
from .DdpTrainer import DdpTrainer


class DdpSparseRpcTrainer(DdpTrainer, DdpRpcTrainer):

    def __init__(self, rank, trainer_count, process_group, use_cuda_rpc, ps_rref, backend, epochs):
        r"""
        a trainer that implements a  DDP training algorithm using a server and process group
        allreduce. the trainer sends sparse gradients using RPC, and the server averages and
        returns the gradients. the process group uses the backend allreduce implementation
        to average the dense gradients.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainer in the world
            process_group (object): distributed process group
            use_cuda_rpc (bool): indicator for CUDA RPC
            server_rref (object): remote reference to the server
            backend (string): distributed communication backend
            epochs (int): epoch count for training
        """
        super().__init__(rank, trainer_count, process_group, use_cuda_rpc, ps_rref, backend, epochs)

    @staticmethod
    def hook(state, bucket):
        r"""
        ddp communication hook that uses the current backend allreduce
        implementation for dense tensors and a server for sparse tensors.
        Args:
            state (object): maintains state during the training process
            bucket (object): gradient bucket
        """
        cref = state.cref
        tensor = bucket.get_tensor()
        if tensor.is_sparse:
            return cref.process_bucket(state, bucket)
        else:
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
