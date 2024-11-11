import torch

from torch.nested._internal.nested_tensor import get_nested_cache

# Test 1:

offsets = torch.tensor(1.0)
lengths = torch.tensor(2.0)

cache1 = get_nested_cache(
    offsets=offsets,
    lengths=lengths,
    cpu_offsets=None,
    cpu_lengths=None,
)

# Test 2:

offsets2 = torch.tensor(1.0)

# Lengths was already registered to "cache1"

cache2 = get_nested_cache(
    offsets=offsets2,
    lengths=lengths,
    cpu_offsets=None,
    cpu_lengths=None,
)

assert cache2 is cache1

# Fresh tensors
offsets3 = torch.tensor(1.0)
lengths2 = torch.tensor(2.0)

cache3 = get_nested_cache(
    offsets=offsets3,
    lengths=lengths2,
    cpu_offsets=None,
    cpu_lengths=None,
)

assert cache3 is not cache1

# Test the priority of aliasing
cache4 = get_nested_cache(
    offsets=offsets3, lengths=lengths, cpu_offsets=None, cpu_lengths=None
)
assert cache4 is cache1

cache5 = get_nested_cache(
    offsets=offsets, lengths=lengths2, cpu_offsets=None, cpu_lengths=None
)
assert cache5 is cache3


# Test views are being created if we need to use one tensor across two caches.
offsets4 = torch.tensor(1.0)
cache6 = get_nested_cache(
    offsets=offsets4, lengths=None, cpu_offsets=None, cpu_lengths=None
)

lengths4 = torch.tensor(2.0)
cache7 = get_nested_cache(
    offsets=None, lengths=lengths4, cpu_offsets=None, cpu_lengths=None
)


print("get_nested_cache 8")
cache8 = get_nested_cache(
    offsets=offsets4, lengths=lengths4, cpu_offsets=None, cpu_lengths=None
)

# We prioritized aliasing the lengths' cache
assert cache8.data["cpu_offsets"]._is_view()
assert cache8.data["cpu_offsets"]._base is offsets4

assert not cache8.data["cpu_lengths"]._is_view()
assert cache8.data["cpu_lengths"] is lengths4


lengths5 = torch.tensor(1.0)
print("get_nested_cache 9")

# Even though cache9 is created using offsets4 which is used to create cache8
# Cache8 ignored offset4's cache.
# Therefore, it is actually different to grab cpu_offsets off of cache8
# which would be a view of offsets4
cache9 = get_nested_cache(
    offsets=cache8.data["cpu_offsets"],
    lengths=lengths5,
    cpu_offsets=None,
    cpu_lengths=None,
)

assert cache9 is cache8
assert cache9.data["cpu_lengths"] is lengths4
