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

import os.path
import struct

import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F

from torch.multiprocessing.reductions import StorageWeakRef


def hash_storage(storage):
    # Use of torch.compile is mandatory for (1) good memory usage
    # and (2) xor_sum implementation
    @torch.compile(dynamic=True)
    def kernel(x):
        # The randint calls are carefully written to hit things we
        # have lowerings for in inductor.  Lack of unsigned 32-bit integer
        # is a pain.
        a = torch.randint(
            -(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32
        ).abs()
        a = ((a % (2**31 - 1)) + 1).long()
        b = (
            torch.randint(
                -(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32
            )
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

    with torch.random.fork_rng([storage.device]):
        torch.manual_seed(0)  # this can be anything, just needs to be fixed
        x = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
        # The dtype-casting view cannot be compiled, and so the
        # padding/reshaping also needs to be done externally even
        # though it could be profitably fused
        pad = x.numel() % 4
        if pad > 0:
            x = F.pad(x, (0, pad), "constant", 0)
        x = x.view(torch.int32)
        torch._dynamo.mark_dynamic(x, 0)
        # We run the 32-bit hash five times with differing parameters to
        # reduce chance of collision
        cs = [kernel(x).item() for _ in range(5)]
        return struct.pack(">iiiii", *cs).hex()


class ContentStoreWriter:
    # Structure:
    #   storages/
    #     00/
    #       0000..00
    #   tensors/
    #     name
    def __init__(self, loc):
        self.loc = loc
        self.seen_storage_hashes = set()

    def write_storage(self, storage):
        h = hash_storage(storage)
        if h in self.seen_storage_hashes:
            return h
        # TODO: consider not using torch.save for this; we don't actually
        # need any metadata for the storage
        subfolder = os.path.join(self.loc, "storages", h[:2])
        os.makedirs(subfolder, exist_ok=True)
        torch.save(storage, os.path.join(subfolder, h))
        self.seen_storage_hashes.add(h)
        return h

    def write_tensor(self, name, t):
        storage = t.untyped_storage()
        h = self.write_storage(storage)
        # TODO: Support more advanced snapshotting of requires_grad/grad/etc
        payload = (
            t.dtype,
            h,
            t.storage_offset(),
            tuple(t.shape),
            t.stride(),
            torch._C._get_tensor_metadata(t),
        )
        subfolder = os.path.join(self.loc, "tensors")
        os.makedirs(subfolder, exist_ok=True)
        torch.save(payload, os.path.join(subfolder, name))


class ContentStoreReader:
    def __init__(self, loc):
        self.loc = loc
        self.storage_cache = {}

    def read_storage(self, h):
        ws = self.storage_cache.get(h)
        if ws is not None:
            s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
            if s is not None:
                return s
        s = torch.load(
            os.path.join(self.loc, "storages", h[:2], h), weights_only=True
        )._untyped_storage
        self.storage_cache[h] = StorageWeakRef(s)
        return s

    def read_tensor(self, name):
        dtype, h, storage_offset, size, stride, metadata = torch.load(
            os.path.join(self.loc, "tensors", name), weights_only=True
        )
        storage = self.read_storage(h)
        t = torch.tensor([], dtype=dtype, device=storage.device)
        t.set_(storage, storage_offset, size, stride)
        torch._utils.set_tensor_metadata(t, metadata)
        return t
