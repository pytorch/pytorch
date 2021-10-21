import torch

def validate_param(param, param_name):
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")

def uniform_(types, args=(), kwargs=None):
    r"""
    Fills the Tensor in sharded_tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        sharded_tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, "kwargs")
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "sharded_tensor")
    a = kwargs['a']
    validate_param(a, "a")
    b = kwargs['b']
    validate_param(b, "b")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.uniform_(shard.tensor, a=a, b=b)
    return sharded_tensor
