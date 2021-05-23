from .DdpRpcHelper import DdpRpcHelper
from .DdpTrainer import DdpTrainer


class DdpSparseDenseRpcTrainer(DdpTrainer, DdpRpcHelper):

    def __init__(self, rank, trainer_count, ps_rref, ps_name, backend, use_cuda_rpc, epochs):
        r"""
        a trainer that implements a  DDP training algorithm using a server.
        the trainer sends gradients using RPC, and the server averages and
        returns the gradients.
        Args:
            rank (int): worker rank
            trainer_count (int): count of trainer in the world
            process_group (object): distributed process group
            use_cuda_rpc (bool): indicator for CUDA RPC
            server_rref (object): remote reference to the server
            backend (string): distributed communication backend
            epochs (int): epoch count for training
        """
        super().__init__(rank, trainer_count, ps_rref, ps_name, backend, use_cuda_rpc, epochs)

    @staticmethod
    def hook(state, bucket):
        r"""
        ddp communication hook that averages sparse and dense tensors using
        process_bucket method.
        Args:
            state (object): maintains state during the training process
            bucket (object): gradient bucket
        """
        cref = state.cref
        return cref.process_bucket(state, bucket)

    def get_hook(self):
        r"""
        returns DdpSparseDenseRpcTrainer.hook
        """
        return DdpSparseDenseRpcTrainer.hook
