import torch

def uniform_(types, args=(), kwargs=None):
    r"""
    Fills the Tensor in sharded_tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        sharded_tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    from torch.distributed._sharded_tensor import ShardedTensor
    assert_not_none(kwargs, "kwargs")
    sharded_tensor = kwargs['tensor']
    assert_not_none(sharded_tensor, "sharded_tensor")
    a = kwargs['a']
    assert_not_none(a, "a")
    b = kwargs['b']
    assert_not_none(b, "b")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.uniform_(shard.tensor, a=a, b=b)
    return sharded_tensor

def assert_not_none(param, param_name):
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")
