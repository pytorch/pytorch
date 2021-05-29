from utils import process_bucket_with_remote_server

from .DdpTrainer import DdpTrainer


class DdpSparseDenseRpcTrainer(DdpTrainer):

    def __init__(self, rank, trainer_count, process_group, use_cuda_rpc, server_rref, backend, epochs):
        r"""
        A trainer that implements a DDP training algorithm using a server.
        The trainer sends gradients using RPC, the server averages and
        returns the gradients.
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
        A ddp communication hook that averages sparse and dense tensors using
        process_bucket_with_remote_server method.
        Args:
            state (object): maintains state during the training process
            bucket (GradBucket): gradient bucket
        """
        return process_bucket_with_remote_server(state, bucket)

    def get_hook(self):
        r"""
        returns DdpSparseDenseRpcTrainer.hook
        """
        return DdpSparseDenseRpcTrainer.hook
