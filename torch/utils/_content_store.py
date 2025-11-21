# mypy: allow-untyped-defs
# This module provides a FAST (on GPU) content addressable store for storages
# (and tensors on top of them) with VERY WEAK portability guarantees (e.g.,
# don't expect CPU/CUDA to address to the same hash, don't expect it to be
# portable across devices) that is NOT cryptographically secure.  In return,
# we are able to hash 40G of tensor data on GPU in less than a second,
# compared to running SHA-1 in CPU which would a minute or so.  The primary
# use case is for efficiently snapshotting intermediate tensor data for
# offline debugging, but it's been put in this module in case you think of
# another use case for it.  The hash function could be replaced with a
# straight reimplementation of SHA-1, which would give us much stronger
# portability guarantees.
#
# WARNING: THERE IS NO BC/FC GUARANTEE FOR THIS FORMAT!  If you need to format
# shift the result, consider packing it into a single torch.save object
# with traditional view sharing.
#
# Because of the weak portability guarantees, you can only write to the
# content store from a single process; we don't provide any capability
# of "reopening" a content store to add more things to it.  But we don't
# assume that you can keep all of the tensors you want to add to the store
# in memory at once, because you probably can't!  Nor do we assume that
# you know a priori whether or not two storages can be deduplicated or not.
#
# Note: only storages are content-addressed; tensors are name addressed
#
# Note: our padding strategy means that [1, 0] and [1] int16 tensors would
# map to the same (padded) storage.  We think this will be immaterial for most
# users.

import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict

import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef


def lazy_compile(**compile_kwargs):
    """Lazily wrap a function with torch.compile on the first call

    This avoids eagerly importing dynamo.
    """

    def decorate_fn(fn):
        @functools.wraps(fn)
        def compile_hook(*args, **kwargs):
            compiled_fn = torch.compile(fn, **compile_kwargs)
            globals()[fn.__name__] = functools.wraps(fn)(compiled_fn)
            return compiled_fn(*args, **kwargs)

        return compile_hook

    return decorate_fn


# Use of torch.compile is mandatory for (1) good memory usage
# and (2) xor_sum implementation.  This is our first instance of
# using PT2 to implement a kernel in PyTorch; if we get AOT capabilities
# it would be good to apply it here.
@lazy_compile(dynamic=True)
def hash_storage_kernel(x):
    # The randint calls are carefully written to hit things we
    # have lowerings for in inductor.  Lack of unsigned 32-bit integer
    # is a pain.
    a = torch.randint(
        -(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32
    ).abs()
    a = ((a % (2**31 - 1)) + 1).long()
    b = (
        torch.randint(-(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32)
        .abs()
        .long()
    )
    # This is a standard shift-multiply universal hash family
    # plus xor sum hash, using Philox to generate random numbers.
    # Our Philox RNG is not deterministic across devices so
    # don't use this for stable hashing.
    #
    # This assumes fixed length so you're also obligated to bucket
    # by the length of tensor as well
    return prims.xor_sum((a * x + b).int(), [0])


# Returns a hex digest of the data in the storage.  Guaranteed to be
# SHA-1 if stable_hash=True, otherwise it will consistent for a single
# process run but not necessarily across processes.
def hash_storage(storage: torch.UntypedStorage, *, stable_hash: bool = False) -> str:
    import torch._dynamo
    from torch._dynamo.utils import is_compile_supported

    device_type = storage.device.type
    if stable_hash or not is_compile_supported(device_type):
        cpu_storage = storage.cpu()
        # TODO: make storage support buffer protocol so this isn't
        # necessary
        buf = (ctypes.c_byte * cpu_storage.nbytes()).from_address(
            cpu_storage.data_ptr()
        )
        sha1 = hashlib.sha1(usedforsecurity=False)
        sha1.update(buf)
        return sha1.hexdigest()

    # TODO: factor this into a random utility
    if device_type == "cpu":
        generator = torch._C.default_generator
    elif device_type == "cuda":
        generator = torch.cuda.default_generators[storage.device.index]
    elif device_type == "mps":
        generator = torch.mps._get_default_mps_generator()
    elif device_type == "xpu":
        generator = torch.xpu.default_generators[storage.device.index]
    else:
        raise AssertionError(f"unhandled device type {device_type}")
    state = generator.get_state()
    try:
        generator.manual_seed(0)
        x = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)  # type: ignore[call-overload]
        # The dtype-casting view cannot be compiled, and so the
        # padding/reshaping also needs to be done externally even
        # though it could be profitably fused
        pad = -x.numel() % 4
        if pad > 0:
            x = F.pad(x, (0, pad), "constant", 0)
        x = x.view(torch.int32)
        # We run the 32-bit hash five times with differing parameters to
        # reduce chance of collision
        ITER = 5
        cs = [hash_storage_kernel(x).item() for _ in range(ITER)]
        return struct.pack(">" + "i" * ITER, *cs).hex()
    finally:
        generator.set_state(state)


class ContentStoreWriter:
    # Structure:
    #   storages/
    #     00/
    #       0000..00
    #   tensors/
    #     name
    def __init__(self, loc: str, stable_hash: bool = False) -> None:
        self.loc: str = loc
        self.seen_storage_hashes: set[str] = set()
        self.stable_hash = stable_hash

    # TODO: offer some sort of non-blocking API to speed things up
    def write_storage(self, storage: torch.UntypedStorage) -> str:
        h = hash_storage(storage, stable_hash=self.stable_hash)
        if h in self.seen_storage_hashes:
            return h
        # TODO: consider not using torch.save for this; we don't actually
        # need any metadata for the storage
        subfolder = os.path.join(self.loc, "storages")
        os.makedirs(subfolder, exist_ok=True)
        target = os.path.join(subfolder, h)
        if os.path.exists(target):
            return h
        torch.save(storage, target)
        self.seen_storage_hashes.add(h)
        return h

    def compute_tensor_metadata(self, t: torch.Tensor, h=None):
        if h is None:
            h = hash_storage(t.untyped_storage(), stable_hash=self.stable_hash)
        return (
            t.dtype,
            h,
            t.storage_offset(),
            tuple(t.shape),
            t.stride(),
            torch._utils.get_tensor_metadata(t),
        )

    def write_tensor(self, name: str, t: torch.Tensor) -> None:
        storage = t.untyped_storage()
        h = self.write_storage(storage)
        # TODO: Support more advanced snapshotting of requires_grad/grad/etc
        d, f = os.path.split(name)
        payload = self.compute_tensor_metadata(t, h=h)
        subfolder = os.path.join(self.loc, "tensors", d)
        os.makedirs(subfolder, exist_ok=True)
        torch.save(payload, os.path.join(subfolder, f))


class ContentStoreReader:
    def __init__(self, loc: str, *, cache=True) -> None:
        self.loc = loc
        self.storage_cache: (
            dict[torch.device | None, dict[str, StorageWeakRef]] | None
        ) = None
        if cache:
            self.storage_cache = defaultdict(dict)

    def read_storage(self, h: str, *, device=None) -> torch.UntypedStorage:
        if device is not None:
            device = torch.device(device)
        ws = (
            self.storage_cache[device].get(h)
            if self.storage_cache is not None
            else None
        )
        s: torch.UntypedStorage | None
        if ws is not None:
            s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
            if s is not None:
                return s
        s = torch.load(
            os.path.join(self.loc, "storages", h),
            weights_only=True,
            map_location=device,
        )._untyped_storage
        if s is None:
            raise AssertionError(
                f"expected storage for hash {h} in {os.path.join(self.loc, 'storages')}, got None"
            )
        if self.storage_cache is not None:
            self.storage_cache[device][h] = StorageWeakRef(s)
        return s

    def read_tensor_metadata(self, name: str):
        fn = os.path.join(self.loc, "tensors", name)
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        return torch.load(fn, weights_only=True)

    def read_tensor(self, name: str, *, device=None) -> torch.Tensor:
        dtype, h, storage_offset, size, stride, metadata = self.read_tensor_metadata(
            name
        )
        storage = self.read_storage(h, device=device)
        t = torch.tensor([], dtype=dtype, device=storage.device)
        t.set_(storage, storage_offset, size, stride)
        torch._utils.set_tensor_metadata(t, metadata)
        return t
