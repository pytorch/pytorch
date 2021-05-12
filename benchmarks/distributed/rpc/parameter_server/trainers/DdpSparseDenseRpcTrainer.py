from .DdpSparseRpcTrainer import DdpSparseRpcTrainer


class DdpSparseDenseRpcTrainer(DdpSparseRpcTrainer):

    def __init__(self, rank, trainer_count, ps_rref, ps_name, backend, use_cuda_rpc, epochs):
        super().__init__(rank, trainer_count, ps_rref, ps_name, backend, use_cuda_rpc, epochs)

    @staticmethod
    def hook(state, bucket):
        cref = state.cref
        return cref.process_bucket(state, bucket)

    def get_hook(self):
        return DdpSparseDenseRpcTrainer.hook
