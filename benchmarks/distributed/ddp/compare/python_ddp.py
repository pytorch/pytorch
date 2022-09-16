import functools
import torch
import torch.distributed as dist
import torch.nn as nn

class PythonDDP(nn.Module):
    """
    Python only implementation for DistributedDataParallel module.

    Unlike the production DistributedDataParallel which relies on many C++ core
    utils to manage gradient distribution and reduction. This class implement
    all functions in pure Python such as param bucketing, gradient
    synchronization and reduction. The only C++ dependency is the common utils:
    ``dist.all_reduce``

    The idea: parallelize gradient calculation and reduction, the same algo as
    https://pytorch.org/docs/stable/notes/ddp.html, main steps:
    1. Distribute params into list of buckets.
    2. Register per-param hook to be invoked when grad is ready during backward
    3. In the hook, copy grad to corresponding bucket. If bucket is full, kick
       off an async all_reduce operation to calculate average grad.
    4. After backward wait for all async ops to be done. Copy reduced grads back
       to original places.

    Two modes are supported, asynchronous reduction (async_reduction=True) and
    synchronous reduction (async_reduction=False) which shares the same algo as
    LegacyDistributedDataParallel.

    Same as DistributedDataParallel to use this class , a process group needs to
    be initiated.

    Example::

        >>> torch.distributed.init_process_group(
        >>>     backend='gloo', world_size=N, init_method='...'
        >>> )
        >>> pg = dist.distributed_c10d._get_default_group()
        >>> async_reduction = True
        >>> module = ToyModel()
        >>> ddp_model = PythonDDP(module, pg, async_reduction)
        >>> loss_fn = nn.MSELoss()
        >>> optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        >>> outputs = ddp_model(torch.randn(20, 10).to(rank))
        >>> labels = torch.randn(20, 10).to(rank)
        >>> loss_fn(outputs, labels).backward()
        >>>
        >>> # Reduce param grads
        >>> ddp_model.all_reduce_grads()
        >>> optimizer.step()
        >>>

    """

    class Bucket:
        """Bucket is a container for list of params. """

        def __init__(self, max_buffer_size):
            self.param_to_offset = {}
            self.buffer = None
            self.ready_param_grad_count = 0
            self.total_elements = 0
            self._MAX_BUFFER_SIZE = max_buffer_size

        def __str__(self):
            return "Bucket: num_params={}, total_elements={}, ready_param_grad_count={}".format(
                len(self.param_to_offset),
                self.total_elements,
                self.ready_param_grad_count)

        def is_full(self):
            """
            Returns whether grad for all the params in current bucket are ready
            and copied to self.buffer.
            """
            assert self.ready_param_grad_count >= 0
            assert self.ready_param_grad_count <= len(self.param_to_offset)
            return len(self.param_to_offset) == self.ready_param_grad_count

        def empty(self):
            self.ready_param_grad_count = 0

        def try_hold_param(self, param):
            """
            Checks whether current bucket has enough buffer to hold the incoming
            param. If there is enough space, distribute param into current
            bucket and Returns true. Otherwise, returns False.
            """
            if self.total_elements + param.numel() <= self._MAX_BUFFER_SIZE :
                self.param_to_offset[param] = self.total_elements
                self.total_elements += param.numel()
                return True
            else:
                return False

    def __init__(self, module, process_group, async_reduction=True, buffer_size=2 ** 22):
        super(PythonDDP, self).__init__()

        self.module = module
        self.process_group = process_group
        self.world_size = dist.get_world_size(group=self.process_group)
        self.async_reduction = async_reduction

        # Holds all_reduce handles, used when async_reduction is True
        self.async_handles = set()

        # Ensure buffer_size is large enough to hold largest param.
        max_numel = max(p.numel() for p in module.parameters())
        assert buffer_size > max_numel, "buffer_size: {} should be larger than largest param: {}".format(buffer_size, max_numel)

        # Build buckets for params
        self.param_to_bucket, self.buckets = self._build_buckets_for_params(buffer_size)

        # Register per-parameter hook to be invoked when grad is ready.
        for p in self.module.parameters():
            assert p.requires_grad
            p.register_hook(functools.partial(self._on_param_grad_ready, p))

    def _build_buckets_for_params(self, max_buffer_size):
        """
        Distributes params into list of buckets. Maintains param -> bucket
        mapping. Returns tuple of (param_to_buckets, buckets).
        """
        print("_build_buckets_for_params called")
        params_to_buckets = {}
        buckets = set()
        cur_bucket = self.Bucket(max_buffer_size)
        total_param = 0
        for param in self.module.parameters():
            total_param += 1
            assert param.requires_grad, "param.requires_grad must be True"
            if cur_bucket.try_hold_param(param):
                params_to_buckets[param] = cur_bucket
                buckets.add(cur_bucket)
            else:
                new_bucket = self.Bucket(max_buffer_size)
                assert new_bucket.try_hold_param(param), "param must be holded in a empty bucket"
                params_to_buckets[param] = new_bucket
                buckets.add(new_bucket)
                cur_bucket = new_bucket

        first_param = next(self.module.parameters())
        for bucket in buckets:
            bucket.buffer = first_param.new(bucket.total_elements)
            assert bucket.buffer is not None, 'bucket.buffer should not be None'
        print("len(param_to_bucket)={}, len(buckets)={}".format(
            len(params_to_buckets), len(buckets)))

        # Sanity check to ensure all params are distributed correctly into buckets
        total_params_in_buckets = 0
        for bucket in buckets:
            total_params_in_buckets += len(bucket.param_to_offset)
        assert total_param == total_params_in_buckets

        return params_to_buckets, buckets

    # Callback when param.grad is ready. Note during callback, param.grad won't
    # be ready yet, we MUST use the given ''grad'' which would be passed upon
    # callback.
    def _on_param_grad_ready(self, param, grad):
        """
        Callback when grad for param is ready. Copy grad to its corresponding
        bucket. When the bucket is full, kickoff an async all_reduce if
        async_reduction is set, and adds the resultant handle to
        self.async_handles.

        .. warning::
            Note param.grad isn't set yet. Use the passed grad instead.
        """
        # Validate bucket and offset are set.
        bucket = self.param_to_bucket.get(param)
        assert bucket is not None, "Failed to find bucket for param"
        offset = bucket.param_to_offset.get(param)
        assert offset is not None, "offset must be set for param"
        assert bucket.buffer is not None, "buffer must be allocated"

        # Copy grad to bucket, note param.grad isn't ready yet.
        sz = param.numel()
        assert grad is not None
        assert param.requires_grad
        assert param.numel() == grad.numel()
        bucket.buffer[offset : offset + sz].copy_(grad.detach().view(-1))
        bucket.ready_param_grad_count += 1

        # Kickoff grad reduction async when bucket is full. This ensures grad
        # reduction and other grad calculation runs in parallel.
        if self.async_reduction and bucket.is_full():
            bucket.buffer.div_(self.world_size)
            handle = dist.all_reduce(
                bucket.buffer, dist.ReduceOp.SUM, self.process_group, True)
            self.async_handles.add(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        """
        Reduces all gradients across worker and updates param gradients. The
        client should call this func post backward.

        If async_reduction is True, waits for all async handles (of all_reduce),
        otherwise, kicks off synchrous all_reduce for all buckets.

        Once all all buckets are reduced, copy the reduced grads back to their
        original parameters. After that, reset all buckets in prep for the next
        iteration.
        """
        if self.async_reduction:
            for handle in self.async_handles:
                handle.wait()
            self.async_handles.clear()
        else:
            for bucket in self.buckets:
                assert bucket.is_full()
                bucket.buffer.div_(self.world_size)
                dist.all_reduce(bucket.buffer, dist.ReduceOp.SUM, self.process_group)

        # Copy reduced-grad back into original place
        for bucket in self.buckets:
            assert bucket.is_full()
            for cur_p, cur_offset in bucket.param_to_offset.items():
                sz = cur_p.numel()
                if cur_p.grad is not None:
                    with torch.no_grad():
                        cur_p.grad.copy_(bucket.buffer[cur_offset : cur_offset + sz].view_as(cur_p))
                else:
                    cur_p.grad = bucket.buffer[cur_offset : cur_offset + sz].view_as(cur_p).clone()

        # Empty bucket for next epoch
        for bucket in self.buckets:
            bucket.empty()
