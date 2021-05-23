from utils import sparse_tensor_to_rpc_format, sparse_rpc_format_to_tensor


class DdpRpcTrainer:

    RPC_SPARSE = "rpc_sparse"
    RPC_DENSE = "rpc_dense"

    @staticmethod
    def process_bucket(state, bucket):
        r"""
        processes a gradient bucket passed by a DDP communication hook
        during .backward(). the method supports processing sparse and dense
        tensors. it records RPC future completion time metric for the trainer.
        Args:
            state (object): maintains state during the training process
            bucket (object): gradient bucket
        """
        cref = state.cref
        tensor = bucket.get_tensor()
        if not cref.use_cuda_rpc:
            tensor = tensor.cpu()
        sparse = tensor.is_sparse
        if sparse:
            tensor = sparse_tensor_to_rpc_format(tensor)
        b_index = bucket.get_index()
        server_args = [
            cref.server_rref,
            state.batch_number,
            b_index,
            tensor
        ]
        key = state.get_key(b_index)
        cref.record_hook_fut_start(
            key,
            cref.RPC_SPARSE if sparse else cref.RPC_DENSE
        )
        fut = cref.server_rref.rpc_async().average_gradient(*server_args)

        def callback(fut):
            cref.record_hook_fut_end(key)
            tensor = fut.wait()
            if type(tensor) is list:
                tensor = sparse_rpc_format_to_tensor(tensor)
            tensor = tensor.cuda(cref.rank)
            return [tensor]

        return fut.then(callback)
