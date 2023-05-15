import torch
from memory_profiler import profile
import time

state_dict_name = "state_dict_mmap.pt"
tensor_size = 1_000_000_000

@profile
def f():
    # should be 1_000_000_000 * 8 bytes = 8 GB
    print("generating weight")
    state_dict = {'weight': torch.randn(tensor_size, dtype=torch.float64)}
    print("generation of weight done")
    saving_start = time.time()
    torch.save(state_dict, state_dict_name, _mmap=True)
    saving_end = time.time()
    print(f"saving done, time={saving_end - saving_start}")
    loading_start = time.time()
    state_dict_loaded = torch.load(state_dict_name, _mmap=True)
    loading_end = time.time()
    print(f"loading done, time={loading_end - loading_start}")
    x = state_dict_loaded['weight']
    y = x[:1000]
    assert torch.equal(state_dict['weight'], x)
    print("loading done")


if __name__ == '__main__':
    f()
