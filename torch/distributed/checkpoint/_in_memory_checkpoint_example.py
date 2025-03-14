import os
import socket
import time
from datetime import timedelta

from torch.distributed.checkpoint._in_memory_checkpoint import (PGTransport, InMemoryStorage)
import time
import torch
import torch.distributed as dist
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

if __name__ == "__main__":
    # Example usage:
    # torchrun --nproc-per-node=3 torch/distributed/checkpoint/_in_memory_checkpoint_example.py
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    pg = dist.new_group(backend="gloo")
    transport = PGTransport(
        pg, timeout=timedelta(seconds=30), device=torch.device("cpu")
    )
    if rank == 0:
        # keep this process alive, the storage in memory
        model = ToyModel()
        storage = InMemoryStorage()
        storage.save("sd", model.state_dict())
        print(f"Rank 0: Saved state_dict. {model.state_dict().keys()=}")

        # Set up a server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', 12345))
        server_socket.listen(1)
        print("Rank 0: Server listening for connections...")
        conn, addr = server_socket.accept()
        print(f"Rank 0: Connection from {addr}")
        # Send shared memory name and size
        shm_name, data_size = storage._storage["sd"]
        conn.sendall(f"{shm_name},{data_size}".encode('utf-8'))
        conn.close()
        storage.clear()

    if rank == 1:
        # TODO: waiting for server to start up
        time.sleep(3)
        # Connect to the server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 12345))
        # Receive shared memory name and size
        data = client_socket.recv(1024).decode('utf-8')
        shm_name, data_size = data.split(',')
        data_size = int(data_size)
        client_socket.close()

        # Access the shared memory
        storage = InMemoryStorage()
        storage._storage["sd"] = (shm_name, data_size)
        state_dict = storage.load("sd")

        print(f"Rank 1: Checkpoint loaded. {state_dict.keys()=}")
        transport.send_checkpoint(
            dst_ranks=[2], step=1, state_dict=state_dict, timeout=timedelta(seconds=30)
        )
        print("Rank 1: Checkpoint sent and saved.")
    elif rank == 2:
        # Rank 1 receives the checkpoint
        received_state_dict = transport.recv_checkpoint(
            src_rank=1, metadata="", step=1, timeout=timedelta(seconds=30)
        )
        print("Rank 2: Checkpoint received")
    dist.destroy_process_group()
