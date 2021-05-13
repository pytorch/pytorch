import copy

import torch
import torch.distributed as c10d
import torch.nn as nn
from servers.AverageParameterServer import AverageParameterServer
from torch.nn.parallel import DistributedDataParallel as DDP

from .DdpTrainerBase import DdpTrainerBase
from .RpcTrainerBase import RpcTrainerBase


class DdpSparseRpcTrainer(DdpTrainerBase, RpcTrainerBase):

    PS_MAP = {
        "AverageParameterServer": AverageParameterServer
    }

    class HookState:

        def __init__(self, cref, process_group):
            self.cref = cref
            self.process_group = process_group
            self.process_group_size = process_group.size()
            self.param_loc = 0
            self.batch_number = -1
            self.hook_gradient_futures = {}

        def add_hook_future(self, gradient, fut):
            self.hook_gradient_futures[gradient] = fut

        def get_key(self):
            return f"{self.batch_number},{self.param_loc}"

        def get_futures(self):
            return self.hook_gradient_futures

        def next_batch_state(self):
            self.param_loc = 0
            self.batch_number += 1
            self.hook_gradient_futures.clear()

    def __init__(self, rank, trainer_count, ps_rref, ps_name, backend, use_cuda_rpc, epochs):
        super().__init__(rank)
        self.rank = rank
        self.trainer_count = trainer_count
        self.ps_rref = ps_rref
        self.ps = self.PS_MAP[ps_name]
        self.backend = backend
        self.use_cuda_rpc = use_cuda_rpc
        self.epochs = epochs

    @staticmethod
    def get_tensor_fut(bucket):
        fut = torch.futures.Future()
        fut.set_result([bucket.get_tensor()])
        return fut

    @staticmethod
    def process_bucket(state, bucket):
        cref = state.cref
        parameters = cref.bucket_to_parameters(bucket)
        for parameter in parameters:
            # other solution?
            p_param = copy.deepcopy(parameter)
            sparse = p_param.is_sparse
            if sparse:
                p_param = cref.sparse_tensor_to_rpc_format(p_param)
            if not cref.use_cuda_rpc:
                if sparse:
                    p_param[0] = p_param[0].cpu()
                    p_param[1] = p_param[1].cpu()
                else:
                    p_param = p_param.cpu()
            if state.batch_number > 0:
                ps = cref.ps
                ps_args = [
                    cref.ps_rref,
                    state.batch_number,
                    state.param_loc,
                    p_param
                ]
                fut = cref.send_async_request(
                    state.get_key(),
                    cref.ps_rref,
                    ps.average_gradient,
                    *ps_args
                )
                state.add_hook_future(parameter, fut)
            state.param_loc += 1
        return cref.get_tensor_fut(bucket)

    @staticmethod
    def hook(state, bucket):
        cref = state.cref
        tensor = bucket.get_tensor()
        tensors_count = len(cref.bucket_to_parameters(bucket))
        if tensor.is_sparse:
            return cref.process_bucket(state, bucket)
        else:
            tensor = [tensor / state.process_group_size]
            key = state.get_key()
            cref.record_hook_fut_start(key, cref.NCCL_ALLREDUCE)
            fut = state.process_group.allreduce(tensor).get_future()
            state.param_loc += tensors_count

            def callback(fut):
                cref.record_hook_fut_end(key)
                return fut.wait()

            return fut.then(callback)

    def get_hook(self):
        return DdpSparseRpcTrainer.hook

    def process_gradient_futs(self, ddp_parameters, hook_gradient_futures):
        # TODO: investigate multi parameter layer hashing
        sgp_map = {}
        dgp_pair_list = []
        for param in ddp_parameters:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                sgp_map[param.grad] = param
            else:
                dgp_pair_list.append([param.grad, param])

        for gradient, fut in hook_gradient_futures.items():
            ps_gradient = fut.wait()
            if gradient.is_sparse:
                param = sgp_map[gradient]
                ps_gradient = self.sparse_rpc_format_to_tensor(ps_gradient)
            else:
                for d_pair in dgp_pair_list:
                    d_g = d_pair[0]
                    if d_g.equal(gradient):
                        param = d_pair[1]
            if not self.use_cuda_rpc:
                ps_gradient = ps_gradient.cuda(self.rank)
            param.grad = ps_gradient

    def train(self, model, data):
        torch.manual_seed(0)
        model = model.cuda(self.rank)
        for i in range(len(data)):
            data[i][0] = data[i][0].cuda(self.rank)
            data[i][1] = data[i][1].cuda(self.rank)
        torch.cuda.synchronize(self.rank)

        process_group_size = self.trainer_count

        store = c10d.FileStore("/tmp/tmpn_k_8so02", process_group_size)
        if self.backend == c10d.Backend.GLOO:
            process_group = c10d.ProcessGroupGloo(store, self.rank, process_group_size)
        elif self.backend == c10d.Backend.NCCL:
            process_group = c10d.ProcessGroupNCCL(store, self.rank, process_group_size)

        ddp_model = DDP(model, device_ids=[self.rank], process_group=process_group)
        hook_state = self.HookState(self, process_group)
        ddp_model.register_comm_hook(hook_state, self.get_hook())
        criterion = nn.CrossEntropyLoss().cuda(self.rank)
        optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)

        ddp_parameters = list(ddp_model.parameters())

        def epoch_key(epoch, index):
            return f"{epoch},{index}"

        for epoch in range(self.epochs):
            for index, batch in enumerate(data):
                hook_state.next_batch_state()
                input = batch[0]
                target = batch[1]
                self.record_batch_start(epoch_key(epoch, index))
                optimizer.zero_grad()
                self.record_forward_start(epoch_key(epoch, index))
                out = ddp_model(input)
                self.record_forward_end(epoch_key(epoch, index))
                loss = criterion(out, target)
                self.record_backward_start(epoch_key(epoch, index))
                loss.backward()
                self.process_gradient_futs(ddp_parameters, hook_state.get_futures())
                self.record_backward_end(epoch_key(epoch, index))
                optimizer.step()
                self.record_batch_end(epoch_key(epoch, index))

        torch.cuda.synchronize(self.rank)
