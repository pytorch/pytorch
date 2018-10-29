import torch
torch.manual_seed(123)
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist

class A(nn.Module):

    def __init__(self):
        super(A, self).__init__()
        self.c = nn.Conv2d(3, 2, 2, 1, 1, bias=True)
        self.c.bias.requires_grad = False

    def forward(self, x):
        return self.c(x)


class B(nn.Module):

    def __init__(self):
        super(B, self).__init__()
        self.a = A()
        self.c = nn.Conv2d(2, 2, 2, 1, 1, bias=True)
        
    def forward(self, x):
        x = self.a(x)
        return self.c(x)


def test_dist(rank):
    torch.cuda.set_device(rank)
    x = torch.ones((2, 3, 16, 16)).cuda()
    b = B().cuda()
    dist_b = torch.nn.parallel.DistributedDataParallel(b, device_ids=[rank], output_device=rank)
    optimizer = optim.SGD(dist_b.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    optimizer.zero_grad()
    l = dist_b(x).mean()
    l.backward()
    optimizer.step()
    return b.a.c.weight, b.a.c.bias, b.c.weight, b.c.bias

def test_without_dist():
    x = torch.ones((4, 3, 16, 16)).cuda()
    b = B().cuda()
    optimizer = optim.SGD(b.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    optimizer.zero_grad()
    l = b(x).mean()
    l.backward()
    optimizer.step()
    return b.a.c.weight, b.a.c.bias, b.c.weight, b.c.bias
    

if __name__ == "__main__":
    """
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 test_dist.py
    # original: IndexError: list assignment index out of range [bug](https://github.com/facebookresearch/maskrcnn-benchmark/issues/52)
    # now: result same as run $ python test_dist.py
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:23571", rank=args.local_rank, world_size=2)


    print([x.sum() for x in test_dist(args.local_rank)])
    """
    # python test_dist.py
    print([x.sum() for x in test_without_dist()])
