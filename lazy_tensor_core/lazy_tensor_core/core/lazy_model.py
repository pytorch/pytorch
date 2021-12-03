from __future__ import print_function

import io
import sys
import re
import threading
import time
import torch
import torch.nn.functional as F
import lazy_tensor_core
import lazy_tensor_core.core.ltc_env_vars as xenv
import lazy_tensor_core.debug.metrics_saver as ms
import lazy_tensor_core.utils.utils as xu
import lazy_tensor_core.utils.closures as xc

_DEVICES = xu.LazyProperty(lambda: lazy_tensor_core._LAZYC._ltc_get_devices())

REDUCE_SUM = 'sum'
REDUCE_MUL = 'mul'
REDUCE_AND = 'and'
REDUCE_OR = 'or'
REDUCE_MIN = 'min'
REDUCE_MAX = 'max'

_TORCH_DIST_GROUPS = dict()
_TORCH_DIST_LOCK = threading.Lock()

_DEVICE_CONTEXTS = dict()
_DEVICE_CONTEXTS_LOCK = threading.Lock()


class DeviceContext(object):
    __slot__ = ['device']

    def __init__(self, device):
        self.device = device


def _get_device_context(device=None):
    if device is None:
        device = lazy_tensor_core._LAZYC._ltc_get_default_device()
    else:
        device = str(device)
    with _DEVICE_CONTEXTS_LOCK:
        devctx = _DEVICE_CONTEXTS.get(device, None)
        if devctx is None:
            devctx = DeviceContext(device)
            _DEVICE_CONTEXTS[device] = devctx
        return devctx


class CollectiveContext(object):

    def __init__(self, groups=None):
        self.replica_devcount = lazy_tensor_core._LAZYC._ltc_get_replication_devices_count()
        self.world_size = xrt_world_size()
        self.ordinal = get_ordinal()
        if self.world_size > self.replica_devcount:
            # This is the sea-of-devices path.
            self.requires_interhost_reduce = self.world_size > 1
            # If groups are enabled we avoid using the two level reduce (first among the
            # fast interconnected cores, then using the torch.distributed support).
            # The intercore_group is always empty, which means all cores, but in the not
            # empty groups case, it won't be used as requires_intercore_reduce is False.
            self.intercore_group = []
            if groups:
                self.requires_intercore_reduce = False
                if self.requires_interhost_reduce:
                    self.interhost_group = _make_group_for_ordinal(self.ordinal, groups)
                    self.is_reduce_host = True
            else:
                self.requires_intercore_reduce = self.replica_devcount > 1
                if self.requires_interhost_reduce:
                    self.interhost_group, ranks = _make_interhost_group(
                        self.replica_devcount, self.world_size)
                    self.is_reduce_host = self.ordinal in ranks
        else:
            # Standard replication path.
            self.requires_intercore_reduce = self.replica_devcount > 1
            self.requires_interhost_reduce = False
            self.intercore_group = groups or []


def _get_torch_dist_group(ranks):
    import torch.distributed as dist

    with _TORCH_DIST_LOCK:
        pg = _TORCH_DIST_GROUPS.get(ranks, None)
        if not pg:
            pg = dist.new_group(ranks=ranks)
            _TORCH_DIST_GROUPS[ranks] = pg
        return pg


def _make_group_for_ordinal(ordinal, groups):
    for g in groups:
        if ordinal in g:
            return _get_torch_dist_group(sorted(g))
    raise RuntimeError('Ordinal {} not found in groups {}'.format(
        ordinal, groups))


def _make_interhost_group(replica_devcount, world_size):
    # Every host in a sea-of-devices case handles replica_devcount devices.
    # The replica device index 0 of each host does the inter-host replication
    # using torch.distributed.
    # The CPU is a special case where there is one process per CPU device,
    # which is also a virtual host within a physical host.
    ranks = tuple(range(0, world_size, replica_devcount))
    return _get_torch_dist_group(ranks), ranks


def is_lazy_tensor(tensor):
    return tensor.device.type == ''


def parse_lazy_device(device):
    m = re.match(r'(CPU|TPU|GPU):(\d+)$', device)
    if m:
        return (m.group(1), int(m.group(2)))


def get_lazy_supported_devices(devkind=None, max_devices=None):
    """Returns a list of supported devices of a given kind.

    Args:
      devkind (string..., optional): If specified, one of `TPU`, `GPU` or `CPU`.
      max_devices (int, optional): The maximum number of devices to be returned of
        that kind.

    Returns:
      The list of device strings.
    """
    devices = _DEVICES.value
    devkind = [devkind] if devkind else ['TPU', 'GPU', 'CPU']
    for kind in devkind:
        kind_devices = []
        for i, device in enumerate(devices):
            if re.match(kind + r':\d+$', device):
                kind_devices.append('lazy:{}'.format(i))
        if kind_devices:
            return kind_devices[:max_devices] if max_devices else kind_devices


def xrt_world_size(defval=1):
    """Retrieves the number of devices which is taking part of the replication.

    Args:
      defval (int, optional): The default value to be returned in case there is no
        replication information available.
        Default: 1

    Returns:
      The number of devices which is taking part of the replication.
    """
    return xu.getenv_as(xenv.WORLD_SIZE, int, defval=defval)


def get_ordinal(defval=0):
    """Retrieves the replication ordinal of the current process.

    The ordinals range from 0 to `xrt_world_size()` minus 1.

    Args:
      defval (int, optional): The default value to be returned in case there is no
        replication information available.
        Default: 0

    Returns:
      The replication ordinal of the current process.
    """
    return xu.getenv_as(xenv.ORDINAL, int, defval=defval)


def get_local_ordinal(defval=0):
    """Retrieves the replication local ordinal of the current process.

    The local ordinals range from 0 to the number of local devices minus 1.

    Args:
      defval (int, optional): The default value to be returned in case there is no
        replication information available.
        Default: 0

    Returns:
      The replication local ordinal of the current process.
    """
    ordinal = xu.getenv_as(xenv.LOCAL_ORDINAL, int, defval=-1)
    if ordinal >= 0:
        return ordinal
    return getattr(_get_device_context(), 'device_index', defval)


def is_master_ordinal(local=True):
    """Checks whether the current process is the master ordinal (0).

    Args:
      local (bool): Whether the local or global master ordinal should be checked.
        In case of multi-host replication, there is only one global master ordinal
        (host 0, device 0), while there are NUM_HOSTS local master ordinals.
        Default: True

    Returns:
      A boolean indicating whether the current process is the master ordinal.
    """
    ordinal = get_local_ordinal() if local else get_ordinal()
    return ordinal == 0


def master_print(*args, fd=sys.stdout, local=False, flush=False):
    if is_master_ordinal(local=local):
        print(*args, file=fd, flush=flush)


def lazy_device(n=None, devkind=None):
    """Returns a given instance of a lazy device.

    Args:
      n (int, optional): The specific instance (ordinal) to be returned. If
        specified, the specific lazy device instance will be returned. Otherwise
        the first device of `devkind` will be returned.
      devkind (string..., optional): If specified, one of `TPU`, `GPU` or `CPU`.

    Returns:
      A `torch.device` with the requested instance.
    """
    if n is None:
        devices = get_lazy_supported_devices(
            devkind=devkind if devkind is not None else None)
        assert devices, 'No devices of {} kind'.format(devkind or 'ANY')
        # This is a utility API mainly called from tests or simple code which wants
        # to just have a single device to run on. Set the default device so that
        # the tensor barrier can work correctly and avoid growing graphs surprises.
        device = devices[0]
    else:
        device = 'lazy:{}'.format(n)
    lazy_tensor_core._LAZYC._ltc_set_default_device(device)
    return torch.device(device)


def _lazy_real_device(device):
    device_str = str(device)
    m = re.match(r'lazy:(\d+)$', device_str)
    if not m:
        raise RuntimeError('Invalid device format: {}'.format(device_str))
    return _DEVICES.value[int(m.group(1))]


def lazy_real_devices(devices):
    return [_lazy_real_device(device) for device in devices]


def lazy_device_hw(device):
    """Returns the hardware type of the given device.

    Args:
      device (string or torch.device): The lazy device that will be mapped to the
        real device.

    Returns:
      A string representation of the hardware type (`CPU`, `TPU`, `GPU`) of the
      given device.
    """
    real_device = _lazy_real_device(device)
    return real_device.split(':')[0]


def lazy_replication_devices(local_devices):
    real_devices = lazy_real_devices(local_devices)
    device_types = set()
    for device in real_devices:
        xdev = parse_lazy_device(device)
        device_types.add(xdev[0])
    if len(device_types) != 1:
        # No replication if the device set spawns multiple device types.
        raise RuntimeError(
            'Cannot replicate across different device types: devices={}/{}'.format(
                local_devices, real_devices))
    device_type = device_types.pop()
    kind_devices = get_lazy_supported_devices(devkind=device_type)
    if len(kind_devices) != len(local_devices):
        # Replication can only happen among all devices of one kind.
        raise RuntimeError(
            'Cannot replicate if number of devices ({}) is different from {}'.
            format(len(local_devices), len(kind_devices)))
    replication_devices = []
    for device in lazy_tensor_core._LAZYC._ltc_get_all_devices():
        xdev = parse_lazy_device(device)
        if not xdev:
            raise RuntimeError('Invalid device format: {}'.format(device))
        if xdev[0] == device_type:
            replication_devices.append(device)
    sorted_by_ordinal = sorted(
        replication_devices, key=lambda device: parse_lazy_device(device)[1])
    return sorted_by_ordinal


def unlazy(tensors):
    """Blocks the program until `tensors` are materialized.

    This API is for benchmarking, don't use it in real models.

    Args:
      tensors (List[torch.Tensor]): List of `torch.Tensor`s to materialize.
        For each Tensor `t` in the list, `t.device` must be a lazy device.
    """
    lazy_tensor_core._LAZYC._ltc_sync_multi(tensors, devices=[], wait=True)


def set_replication(device, devices):
    device = str(device)
    devctx = _get_device_context(device=device)
    devices = [str(x) for x in devices]
    if devices:
        replication_devices = lazy_replication_devices(devices)
        lazy_tensor_core._LAZYC._ltc_set_replication_devices(replication_devices)
        devctx.device_index = devices.index(device)
    else:
        lazy_tensor_core._LAZYC._ltc_set_replication_devices([])
        devctx.device_index = 0
    devctx.all_reduce_token = None
    lazy_tensor_core._LAZYC._ltc_set_default_device(device)


class RateTracker(object):

    def __init__(self, smooth_factor=None):
        self._smooth_factor = xu.getenv_as(
            'RATE_TRACKER_SMOOTHING', float,
            0.4) if smooth_factor is None else smooth_factor
        self._start_time = time.time()
        self._partial_time = self._start_time
        self._partial_count = 0.0
        self._partial_rate = None
        self._count = 0.0

    def _update(self, now, rate):
        self._partial_count += self._count
        self._count = 0.0
        self._partial_time = now
        self._partial_rate = rate

    def add(self, count):
        self._count += count

    def _smooth(self, current_rate):
        if self._partial_rate is None:
            smoothed_rate = current_rate
        else:
            smoothed_rate = ((1 - self._smooth_factor) * current_rate +
                             self._smooth_factor * self._partial_rate)
        return smoothed_rate

    def rate(self):
        now = time.time()
        delta = now - self._partial_time
        report_rate = 0.0
        if delta > 0:
            report_rate = self._smooth(self._count / delta)
            self._update(now, report_rate)
        return report_rate

    def global_rate(self):
        delta = time.time() - self._start_time
        count = self._partial_count + self._count
        return count / delta if delta > 0 else 0.0


class ToLazyTensorArena(object):

    def __init__(self, convert_fn, select_fn):
        self._convert_fn = convert_fn
        self._select_fn = select_fn
        self._tensors = []

    def _add(self, tensor):
        self._tensors.append(tensor)

    def _convert(self):
        self._index = 0
        if self._tensors:
            self._converted_tensors = self._convert_fn(self._tensors)
        else:
            self._converted_tensors = []

    def _get_converted_tensor(self):
        assert self._index < len(self._converted_tensors)
        new_tensor = self._converted_tensors[self._index]
        self._index += 1
        return new_tensor

    def _collect_tensors(self, inputs):

        def collect_fn(value):
            self._add(value)

        xu.for_each_instance(inputs, lambda x: self._select_fn(x), collect_fn)

    def _replace_tensors(self, inputs):

        def convert_fn(value):
            return self._get_converted_tensor()

        return xu.for_each_instance_rewrite(inputs, lambda x: self._select_fn(x),
                                            convert_fn)

    def transform(self, inputs):
        self._tensors = []
        self._collect_tensors(inputs)
        self._convert()
        return self._replace_tensors(inputs)


def check_view_sharing(obj):
    tensors = set()
    aliases = dict()

    def tensor_info(t):
        return '{}{}'.format(t.dtype, list(t.size()))

    def tensor_id(t):
        if is_lazy_tensor(t):
            return lazy_tensor_core._LAZYC._ltc_get_tensor_id(t), 'lazy'
        return id(t), 'torch'

    def alias_id(t):
        if is_lazy_tensor(t):
            aid = lazy_tensor_core._LAZYC._ltc_get_tensor_view_alias_id(t)
            return None if aid == 0 else aid, 'lazy'
        return t.storage().data_ptr(), 'torch'

    def check_object(obj):
        tid = tensor_id(obj)
        if tid not in tensors:
            tensors.add(tid)
            aid = alias_id(obj)
            if aid[0] is not None:
                if aid in aliases:
                    oobj = aliases[aid]
                    raise RuntimeError(
                        'Tensor ID {} ({}) is sharing a view with tensor ID {} ({})'.
                        format(tid, tensor_info(obj), tensor_id(oobj), tensor_info(oobj)))
                aliases[aid] = obj

    xu.for_each_instance(obj, lambda x: type(x) == torch.Tensor, check_object)


def _fetch_gradients(optimizer):
    gradients = []
    for param_group in optimizer.__getstate__()['param_groups']:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        gradients.append(p.grad.data)
    return gradients


def _get_all_reduce_token():
    devctx = _get_device_context()
    token = getattr(devctx, 'all_reduce_token', None)
    if token is None:
        token = lazy_tensor_core._LAZYC._ltc_create_token(devctx.device)
        devctx.all_reduce_token = token
    return token, devctx


def _torch_all_reduce(reduce_type, inputs, group=None):
    import torch.distributed as dist

    if reduce_type == REDUCE_SUM:
        reduce_op = dist.ReduceOp.SUM
    elif reduce_type == REDUCE_MUL:
        reduce_op = dist.ReduceOp.PRODUCT
    elif reduce_type == REDUCE_MIN:
        reduce_op = dist.ReduceOp.MIN
    elif reduce_type == REDUCE_MAX:
        reduce_op = dist.ReduceOp.MAX
    elif reduce_type == REDUCE_OR:
        reduce_op = dist.ReduceOp.BOR
    elif reduce_type == REDUCE_AND:
        reduce_op = dist.ReduceOp.BAND
    else:
        raise RuntimeError('Invalid reduce type: {}'.format(reduce_type))

    results = []
    async_op = None
    for tensor in inputs:
        # Use async flag to overlap pytorch reduce ops with lazy tensor fetches.
        cpu_tensor = lazy_tensor_core._LAZYC._ltc_get_cpu_tensors([tensor])[0]
        results.append(cpu_tensor)
        if async_op is not None:
            async_op.wait()
        async_op = dist.all_reduce(
            cpu_tensor, reduce_op, async_op=True, group=group)
    if async_op is not None:
        async_op.wait()
    return results


def _host_all_reduce(reduce_type, inputs, cctx, scale=None):
    # Barrier must happen on all devices.
    lazy_tensor_core._LAZYC._ltc_sync_multi(
        inputs, devices=[], wait=True, sync_ltc_data=True)

    # Here we use the torch.distributed reductions only on one device in the
    # replication set, and then use in graph fast interconnect reduction to
    # transfer the result to all replication devices.
    # One core per fast interconnect replica group does the torch.distributed
    # reduction and post the result, while the others post zeros.
    if cctx.is_reduce_host:
        results = _torch_all_reduce(reduce_type, inputs, group=cctx.interhost_group)
        for i in range(0, len(inputs)):
            inputs[i].copy_(results[i])
            if scale is not None:
                inputs[i].mul_(scale)
    else:
        for tensor in inputs:
            tensor.zero_()
    if cctx.requires_intercore_reduce:
        token, devctx = _get_all_reduce_token()
        devctx.all_reduce_token = lazy_tensor_core._LAZYC._ltc_all_reduce_inplace(
            REDUCE_SUM, inputs, token, 1.0, [])


def all_reduce(reduce_type, inputs, scale=1.0, groups=None, cctx=None):
    """Performs an inplace reduce operation on the input tensor(s).

    Args:
      reduce_type (string): One of ``ltm.REDUCE_SUM``, ``ltm.REDUCE_MUL``,
      ``ltm.REDUCE_AND``, ``ltm.REDUCE_OR``, ``ltm.REDUCE_MIN`` and ``ltm.REDUCE_MAX``.
      inputs: Either a single `torch.Tensor` or a list of `torch.Tensor` to
        perform the all reduce op to.
      scale (float): A default scaling value to be applied after the reduce.
        Default: 1.0
      groups (list, optional): A list of list, representing the replica groups for
        the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.
    Returns:
      If a single `torch.Tensor` is passed, the return value is a `torch.Tensor`
      holding the reduced value (across the replicas). If a list/tuple is passed,
      this function performs an inplace all-reduce op on the input tensors, and
      returns the list/tuple itself.
    """
    # In a sea-of-devices case we use two level of reductions. One using the fast
    # device interconnect, and then using the torch.distributed reduction API to
    # reduce across the detached hosts.
    # One special case is CPU devices, which do not support in graph reductions,
    # so in that case we create differente processes having a single replication
    # device. That will skip the in graph reductions and use the torch.distributed
    # support across all CPU devices.
    if cctx is None:
        cctx = CollectiveContext(groups=groups)
    if cctx.requires_intercore_reduce:
        token, devctx = _get_all_reduce_token()
        if isinstance(inputs, torch.Tensor):
            result = lazy_tensor_core._LAZYC._ltc_all_reduce(reduce_type, inputs, token,
                                                             scale, cctx.intercore_group)
            devctx.all_reduce_token = result[1]
            results = [result[0]]
        else:
            devctx.all_reduce_token = lazy_tensor_core._LAZYC._ltc_all_reduce_inplace(
                reduce_type, inputs, token, scale, cctx.intercore_group)
            results = inputs
    else:
        if isinstance(inputs, torch.Tensor):
            results = [inputs.clone()]
        else:
            results = inputs

    if cctx.requires_interhost_reduce:
        assert groups is None, 'Groups are not supported in sea-of-devices mode'
        hscale = scale if cctx.replica_devcount <= 1 and scale != 1.0 else None
        _host_all_reduce(reduce_type, results, cctx, scale=hscale)
    return results[0] if isinstance(inputs, torch.Tensor) else results


def all_gather(value, dim=0, groups=None):
    """Performs an all-gather operation along a given dimension.

    Args:
      value (torch.Tensor): The input tensor.
      dim (int): The gather dimension.
        Default: 0
      groups (list, optional): A list of list, representing the replica groups for
        the `all_gather()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.
    Returns:
      A tensor which has, in the ``dim`` dimension, all the values from the
      participating replicas.
    """
    if dim < 0:
        dim = value.dim() + dim
    size = value.size(dim)
    padding = [0] * (2 * value.dim())
    ordinal = get_ordinal()
    if groups is None:
        left, right = ordinal, xrt_world_size() - 1 - ordinal
    else:
        ordinals = dict()
        for g in groups:
            for i, x in enumerate(g):
                ordinals[x] = (i, len(g) - 1 - i)
        left, right = ordinals[ordinal]
    idx = value.dim() - 1 - dim
    padding[2 * idx] = left * size
    padding[2 * idx + 1] = right * size
    return all_reduce(REDUCE_SUM, F.pad(value, padding), groups=groups)


def all_to_all(value,
               split_dimension,
               concat_dimension,
               split_count,
               groups=None):
    """Performs an `AllToAll()` operation on the input tensor.

    See: https://www.tensorflow.org/xla/operation_semantics#alltoall

    Args:
      value (torch.Tensor): The input tensor.
      split_dimension (int): The dimension upon which the split should happen.
      concat_dimension (int): The dimension upon which the concat should happen.
      split_count (int): The split count.
      groups (list, optional): A list of list, representing the replica groups for
        the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.

    Returns:
      The result `torch.Tensor` of the `all_to_all()` operation.
    """
    token, devctx = _get_all_reduce_token()
    result = lazy_tensor_core._LAZYC._ltc_all_to_all(value, token, split_dimension,
                                                     concat_dimension, split_count,
                                                     groups or [])
    devctx.all_reduce_token = result[1]
    return result[0]


def collective_permute(value, pairs):
    """Performs a `CollectivePermute()` operation on the input tensor.

    WARNING: This function is not very reliable, may produce wrong results under
             certain inputs. Use it at your own risk.

    See: https://www.tensorflow.org/xla/operation_semantics#collectivepermute

    Args:
      value (torch.Tensor): The input tensor.
      pairs (list): A list of (source_replica_id, target_replica_id) pairs,
        representing the sender and receiver for the `collective_permute()`
        operation. Example: `[[0, 1], [1, 2], [2, 0]]` defines three pairs. The
          tensor will be sent from replica 0 to replica 1, replica 1 to
          replica 2, and replica 2 to replica 0.

    Returns:
      The result `torch.Tensor` of the `collective_permute()` operation.
    """
    token, devctx = _get_all_reduce_token()
    result = lazy_tensor_core._LAZYC._ltc_collective_permute(value, token, pairs)
    devctx.all_reduce_token = result[1]
    return result[0]


def add_step_closure(closure, args=(), run_async=False):
    """Adds a closure to the list of the ones to be run at the end of the step.

    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.

    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
    devctx = _get_device_context()
    closures_type = 'async_step_closures' if run_async else 'step_closures'
    step_closures = getattr(devctx, closures_type, None)
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    step_closures.append(lambda a=args: closure(*a))


def _run_step_closures():
    devctx = _get_device_context()
    async_step_closures = getattr(devctx, 'async_step_closures', None)
    if async_step_closures is not None:
        devctx.async_step_closures = []
        async_closure_handler = getattr(devctx, 'async_closure_handler', None)
        if async_closure_handler is None:
            async_closure_handler = xc.AsyncClosureHandler()
            devctx.async_closure_handler = async_closure_handler
        async_closure_handler.run_all(async_step_closures)

    step_closures = getattr(devctx, 'step_closures', None)
    if step_closures is not None:
        devctx.step_closures = []
        for closure in step_closures:
            closure()
    return devctx


def mark_step():
    if xu.getenv_as('LTC_EMIT_STEPLOG', bool, False):
        print('lazy_tensor_core.core.lazy_model::mark_step', file=sys.stderr, flush=True)
    lazy_tensor_core._LAZYC._ltc_step_marker(
        lazy_tensor_core._LAZYC._ltc_get_default_device(), [],
        wait=xu.getenv_as('LTC_SYNC_WAIT', bool, False))
    # Only emit metrics from the first local device index, to avoid emitting the
    # same values from different threads.
    if is_master_ordinal():
        ms.save_metrics()
    devctx = _run_step_closures()
    devctx.all_reduce_token = None


def wait_device_ops(devices=None):
    """Waits for all the async operations on the given devices to complete.

    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    lazy_tensor_core._LAZYC._ltc_wait_device_ops(devices=devices)


def reduce_gradients(optimizer, groups=None):
    """Reduces all the gradients handled by an optimizer.

    Args:
      optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance
        containing the gradients to be reduced.
      groups (list, optional): A list of list, representing the replica groups for
        the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.
    """
    cctx = CollectiveContext()
    count = max(cctx.replica_devcount, cctx.world_size)
    if count > 1:
        gradients = _fetch_gradients(optimizer)
        all_reduce(
            REDUCE_SUM, gradients, scale=1.0 / count, groups=groups, cctx=cctx)


def optimizer_step(optimizer, barrier=False, optimizer_args=None, groups=None):
    """Run the provided optimizer step and issue the device step computation.

    Args:
      optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance whose
        `step()` function needs to be called. The `step()` function will be called
        with the `optimizer_args` named arguments.
      barrier (bool, optional): Whether the lazy tensor barrier should be issued in
        this API. If using the PyTorch lazy tensors `ParallelLoader` or `DataParallel`
        support, this is not necessary as the barrier will be issued by the
        data loader iterator `next()` call.
        Default: False
      optimizer_args (dict, optional): Named arguments dictionary for the
        `optimizer.step()` call.
      groups (list, optional): A list of list, representing the replica groups for
        the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.

    Returns:
      The same value returned by the `optimizer.step()` call.
    """
    if optimizer_args is None:
        optimizer_args = {}
    reduce_gradients(optimizer, groups=groups)
    loss = optimizer.step(**optimizer_args)
    if barrier:
        mark_step()
    return loss


def save(data, file_or_path, master_only=True, global_master=False):
    """Saves the input data into a file.

    The saved data is transferred to PyTorch CPU device before being saved, so a
    following `torch.load()` will load CPU data.
    Care must be taken when working with views. Instead of saving views it's
    recommended that you recreate them after the tensors have been loaded and
    moved to their destination device(s).

    Args:
      data: The input data to be saved. Any nested combination of Python objects
        (list, tuples, sets, dicts, ...).
      file_or_path: The destination for the data saving operation. Either a file
        path or a Python file object. If `master_only` is ``False`` the path or
        file objects must point to different destinations as otherwise all the
        writes from the same host will override each other.
      master_only (bool, optional): Whether only the master device should save the
        data. If False, the `file_or_path` argument should be a different file or
        path for each of the ordinals taking part to the replication, otherwise
        all the replicas on the same host will be writing to the same location.
        Default: True
      global_master (bool, optional): When ``master_only`` is ``True`` this flag
        controls whether every host's master (if ``global_master`` is ``False``)
        saves the content, or only the global master (ordinal 0).
        Default: False
    """
    should_write_data = not master_only or is_master_ordinal(
        local=not global_master)

    cpu_data = _maybe_convert_to_cpu(data, convert=should_write_data)
    if should_write_data:
        torch.save(cpu_data, file_or_path)
    rendezvous('lazy_tensor_core.core.lazy_model.save')


def _maybe_convert_to_cpu(data, convert=True):

    def convert_fn(tensors):
        lazy_tensor_core._LAZYC._ltc_sync_multi(
            tensors, devices=[], wait=True, sync_ltc_data=True)
        if not convert:
            return tensors
        return lazy_tensor_core._LAZYC._ltc_get_cpu_tensors(tensors)

    def select_fn(v):
        return type(v) == torch.Tensor and is_lazy_tensor(v)

    return ToLazyTensorArena(convert_fn, select_fn).transform(data)


def send_cpu_data_to_device(data, device):

    def convert_fn(tensors):
        devices = [str(device)] * len(tensors)
        return lazy_tensor_core._LAZYC._ltc_tensors_from_aten(tensors, devices)

    def select_fn(v):
        return type(v) == torch.Tensor and v.device.type == 'cpu'

    return ToLazyTensorArena(convert_fn, select_fn).transform(data)


def rendezvous(tag, payload=b'', replicas=None):
    """Waits for all the mesh clients to reach the named rendezvous.

    Args:
      tag (string): The name of the rendezvous to join.
      payload (bytes, optional): The payload to be sent to the rendezvous.
      replicas (list, int): The replica ordinals taking part of the rendezvous.
        Empty means all replicas in the mesh.
        Default: []

    Returns:
      The payloads exchanged by all the other cores, with the payload of core
      ordinal `i` at position `i` in the returned tuple.
    """
    if replicas is None:
        replicas = []
    return lazy_tensor_core._LAZYC._lazy_rendezvous(get_ordinal(), tag, payload, replicas)


def do_on_ordinals(target, data=(), ordinals=(0,)):
    """Runs a function only on a given set of ordinals.

    Args:
      target (callable): The function to be run on `ordinals`.
      data: Any input data for the `target` function which contains tensors. All
        the lazy tensors used by the `target` function must be passed in this
        argument. Every other data used by the function can be captured by the
        Python interpreter as usual.
        Default: ()
      ordinals (list, int): The list/set of ordinals where the `target` function
        should run.
        Default: (0,)

    Returns:
      In the ordinals that ran the `target` function, the function return value,
      otherwise `None`.
    """
    running = get_ordinal() in ordinals
    cpu_data = _maybe_convert_to_cpu(data, convert=running)
    if running:
        result = target(*cpu_data)
    else:
        result = None
    rendezvous('lazy_tensor_core.core.lazy_model.do_on_ordinals')
    return result


def mesh_reduce(tag, data, reduce_fn):
    """Performs an out-of-graph client mesh reduction.

    Args:
      tag (string): The name of the rendezvous to join.
      data: The data to be reduced. The `reduce_fn` callable will receive a list
        with the copies of the same data coming from all the mesh client processes
        (one per core).
      reduce_fn (callable): A function which receives a list of `data`-like
        objects and returns the reduced result.

    Returns:
      The reduced value.
    """
    cpu_data = _maybe_convert_to_cpu(data)
    bio = io.BytesIO()
    torch.save(cpu_data, bio)
    xdata = rendezvous(tag, bio.getvalue())
    xldata = []
    for xd in xdata:
        xbio = io.BytesIO(xd)
        xldata.append(torch.load(xbio))
    return reduce_fn(xldata) if xldata else cpu_data


def set_rng_state(seed, device=None):
    """Sets the random number generator state.

    Args:
      seed (integer): The state to be set.
      device (string, optional): The device where the RNG state needs to be set.
        If missing the default device seed will be set.
    """
    if device is None:
        device = lazy_tensor_core._LAZYC._ltc_get_default_device()
    lazy_tensor_core._LAZYC._ltc_set_rng_seed(seed, str(device) if device else '')


def get_rng_state(device=None):
    """Gets the current running random number generator state.

    Args:
      device (string, optional): The device whose RNG state needs to be retrieved.
        If missing the default device seed will be set.

    Returns:
      The RNG state, as integer.
    """
    if device is None:
        device = lazy_tensor_core._LAZYC._ltc_get_default_device()
    return lazy_tensor_core._LAZYC._ltc_get_rng_seed(str(device) if device else '')


def get_memory_info(device):
    """Retrieves the device memory information.

    Args:
      device (string): The device whose memory information are requested.

    Returns:
      A dictionary with `kb_free` (free memory in KB) and `kb_total` (total
      memory in KB) keys.
    """
    return lazy_tensor_core._LAZYC._ltc_memory_info(str(device))

def set_noop_execution_mode(enable):
    return lazy_tensor_core._LAZYC._ltc_set_noop_execution_mode(enable)
