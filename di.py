import os
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch._dynamo.config

torch._dynamo.config.enable_sync_dist_compilation = True

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

"""
class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
"""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = SimpleModel(10, 2).to(rank)
    model.forward = torch.compile(model.forward)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    """
    dataset = DummyDataset(1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """

    num_epochs = 10

    def B(s):
        return [torch.randn(s, 10), torch.randint(0, 2, (s,))]

    if rank == 0:
        dataloader = [B(5), B(8), B(6)]
    else:
        assert rank == 1
        dataloader = [B(6), B(6), B(3)]

    for epoch in range(num_epochs):
        # sampler.set_epoch(epoch)
        for data, labels in dataloader:
            data, labels = data.to(rank), labels.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = 2  # Total number of processes
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
