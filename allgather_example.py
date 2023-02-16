import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F



def allgather_with_dim(shard, dim, ranks, group_size, tag="") -> torch.Tensor:
    torch.ops.aten.all_gather(shard, tag, ranks, group_size, dim=dim)

# after sharded
# world_size = 4
# creating shard similar to the below semantics
# shard_on_each_rank = big_tensor.tensor_split(world_size, dim=0)

def generate_shard_on_my_rank(rank, world_size, big_tensor, shard_dim):
    shard_on_my_rank = big_tensor.tensor_split(world_size, dim=shard_dim)[rank]

    return shard_on_my_rank


def pad_shard_on_my_rank(rank, shard, shard_dim, idx_start_to_pad):
    if rank >= idx_start_to_pad:
        # pad tensor by 1 on the shard dim
        pad = [0, 0] * (shard.ndim - shard_dim)
        pad[-1] = 1
        return F.pad(tensor, pad)
    else:
        return shard

def unpad_shard_on_my_rank(rank, shard, shard_dim, idx_start_to_pad):
    if rank >= idx_start_to_pad:
        # unpad tensor by 1 on the shard dim
        return tensor.narrow(shard_dim, start=0, length=tensor.size(shard_dim) - 1)
    else:
        return shard

def all_gather_examples(rank, world_size):
    # set up world pg
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    

    # Test Case 1: shard on tensor dim 0 with padding
    big_tensor = torch.randn(10, 5)

    # mimic sharding a tensor on tensor dim 0
    # this would generate:
    # rank 0: torch.randn(3, 5)
    # rank 1: torch.randn(3, 5)
    # rank 2: torch.randn(2, 5)
    # rank 3: torch.randn(2, 5)
    # we need to pad the rank 2/3 before calling all_gather
    my_shard = generate_shard_on_my_rank(rank, world_size, big_tensor, shard_dim=0)

    idx_start_to_pad = big_tensor.size(0) % world_size
    padded_shard = pad_shard_on_my_rank(rank, my_shard, shard_dim=0, idx_start_to_pad=idx_start_to_pad)

    allgathered_tensor = all_gather_with_dim(my_shard, dim=0, ranks=[0, 1, 2, 3], 4)

    # it's an allgathered tensor with padding inside, need to split, unpad then recat
    gathered_list = allgathered_tensor.tensor_split(world_size, dim=0)
    for i in range(len(gathered_list)):
        gathered_list[i] = unpad_shard_on_my_rank(rank, gathered_list[i], shard_dim=0, idx_start_to_pad=idx_start_to_pad)

    recatted_big_tensor = torch.cat(gathered_list, dim=0)

    # would recatted_big_tensor == big_tensor?

    # Test case 2: shard on tensor dim 1 with padding

    big_tensor = torch.randn(5, 13)
    # mimic sharding a tensor on tensor dim 1
    # rank 0: torch.randn(5, 4)
    # rank 1: torch.randn(5, 3)
    # rank 2: torch.randn(5, 3)
    # rank 3: torch.randn(5, 3)
    my_shard = generate_shard_on_my_rank(rank, world_size, big_tensor, shard_dim=1)

    idx_start_to_pad = big_tensor.size(1) % world_size
    padded_shard = pad_shard_on_my_rank(rank, my_shard, shard_dim=1, idx_start_to_pad=idx_start_to_pad)

    allgathered_tensor = all_gather_with_dim(my_shard, dim=1, ranks=[0, 1, 2, 3], 4)

    # it's an allgathered tensor with padding inside, need to split, unpad then recat
    gathered_list = allgathered_tensor.tensor_split(world_size, dim=1)
    for i in range(len(gathered_list)):
        gathered_list[i] = unpad_shard_on_my_rank(rank, gathered_list[i], shard_dim=1, idx_start_to_pad=idx_start_to_pad)

    recatted_big_tensor = torch.cat(gathered_list, dim=1)
    
    '''
    Question/Experiment: for cases like un-even allgather, we have to pad the shards to even size,
    then call allgather, which returns a big tensor

    Then we need to split them, unpad the padded shard, then re-concat them together for correct shape

    Would inductor help eliminates those padding/unpadding in some way? 

    '''



if __name__ == '__main__':
    world_size = 4
    mp.spawn(all_gather_examples, args=(world_size,), nprocs=world_size, join=True)
