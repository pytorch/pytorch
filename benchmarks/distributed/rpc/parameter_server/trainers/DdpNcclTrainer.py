import torch
import torch.distributed as c10d
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .DdpTrainerBase import DdpTrainerBase


class DdpNcclTrainer(DdpTrainerBase):

    class HookState:

        def __init__(self, cref, process_group):
            self.cref = cref
            self.process_group = process_group
            self.process_group_size = process_group.size()
            self.param_location = 0
            self.batch_number = -1

        def get_key(self):
            return f"{self.batch_number},{self.param_location}"

        def next_batch_state(self):
            self.param_location = 0
            self.batch_number += 1

    def __init__(self, rank, trainer_count, ps_rref, epochs):
        super().__init__(rank)
        self.rank = rank
        self.trainer_count = trainer_count
        self.epochs = epochs

    @staticmethod
    def hook(state, bucket):
        cref = state.cref
        tensors_count = len(cref.bucket_to_parameters(bucket))
        tensors = [bucket.get_tensor() / state.process_group_size]
        key = state.get_key()
        cref.record_hook_fut_start(key, cref.NCCL_ALLREDUCE)
        fut = state.process_group.allreduce(tensors).get_future()
        state.param_location += tensors_count

        def callback(fut):
            cref.record_hook_fut_end(key)
            return fut.wait()

        return fut.then(callback)

    def train(self, model, data):
        torch.manual_seed(0)
        model = model.cuda(self.rank)
        for i in range(len(data)):
            data[i][0] = data[i][0].cuda(self.rank)
            data[i][1] = data[i][1].cuda(self.rank)
        torch.cuda.synchronize(self.rank)

        process_group_size = self.trainer_count

        store = c10d.FileStore("/tmp/tmpn_k_8so02", process_group_size)

        process_group = c10d.ProcessGroupNCCL(
            store, self.rank, process_group_size
        )

        ddp_model = DDP(
            model, device_ids=[self.rank], process_group=process_group
        )

        hook_state = self.HookState(self, process_group)

        ddp_model.register_comm_hook(hook_state, DdpNcclTrainer.hook)

        criterion = nn.CrossEntropyLoss().cuda(self.rank)

        optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)

        def epoch_key(epoch, index):
            return f"{epoch},{index}"

        for epoch in range(self.epochs):
            for index, batch in enumerate(data):
                hook_state.next_batch_state()
                input, target = batch[0], batch[1]

                self.record_batch_start(epoch_key(epoch, index))

                optimizer.zero_grad()

                self.record_forward_start(epoch_key(epoch, index))

                out = ddp_model(input)

                self.record_forward_end(epoch_key(epoch, index))

                loss = criterion(out, target)

                self.record_backward_start(epoch_key(epoch, index))

                loss.backward()

                self.record_backward_end(epoch_key(epoch, index))

                optimizer.step()

                self.record_batch_end(epoch_key(epoch, index))

        torch.cuda.synchronize(self.rank)
