## @package data_parallel_model
# Module caffe2.python.data_parallel_model




from collections import OrderedDict
from future.utils import viewitems, viewkeys, viewvalues
import logging
import copy

from multiprocessing import cpu_count

from caffe2.python import \
    model_helper, dyndep, scope, workspace, core, memonger, utils
from caffe2.proto import caffe2_pb2

import numpy as np
import warnings

dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/gloo:gloo_ops")

# We only import nccl operators when the machine has GPUs
# Otherwise the binary can be compiled with CPU-only mode, and
# will not be able to find those modules
if workspace.NumGpuDevices() > 0:
    dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/nccl:nccl_ops")
    dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/gloo:gloo_ops_gpu")

log = logging.getLogger("data_parallel_model")
log.setLevel(logging.INFO)

_DEFAULT_TIMEOUT_SEC = 30
_DEFAULT_BARRIER_NET_TIMEOUT_SEC = 300


def Parallelize_GPU(*args, **kwargs):
    kwargs['cpu_device'] = False
    Parallelize(*args, **kwargs)


def Parallelize_CPU(*args, **kwargs):
    kwargs['cpu_device'] = True
    Parallelize(*args, **kwargs)

def Parallelize_iDeep(*args, **kwargs):
    kwargs['ideep'] = True
    Parallelize(*args, **kwargs)

def Parallelize(
    model_helper_obj,
    input_builder_fun,
    forward_pass_builder_fun,
    param_update_builder_fun=None,
    optimizer_builder_fun=None,
    post_sync_builder_fun=None,
    pre_grad_net_transformer_fun=None,
    net_transformer_fun=None,
    devices=None,
    rendezvous=None,
    net_type='dag',
    broadcast_computed_params=True,
    optimize_gradient_memory=False,
    dynamic_memory_management=False,
    blobs_to_keep=None,
    use_nccl=False,
    max_concurrent_distributed_ops=16,
    cpu_device=False,
    ideep=False,
    num_threads_per_device=4,
    shared_model=False,
    combine_spatial_bn=False,
    barrier_net_timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC,
):
    '''
    Function to create a model that can run on many GPUs or CPUs.
      model_helper_obj: an object of ModelHelper
      input_builder_fun:
                         Function that adds the input operators
                         Note: Remember to instantiate reader outside of this
                         function so all devices share same reader object.
                         Signature:  input_builder_fun(model)
      forward_pass_builder_fun:
                        Function to add the operators to the model.
                        Must return list of loss-blob references that
                        are used to build the gradient. Loss scale parameter
                        is passed, as you should scale the loss of your model
                        by 1.0 / the total number of devices.
                        Signature: forward_pass_builder_fun(model, loss_scale)
      param_update_builder_fun:
                        Function that adds operators that are run after
                        gradient update, such as updating the weights and
                        weight decaying. This is called for each GPU separately.
                        Signature: param_update_builder_fun(model)
      optimizer_builder_fun:
                        Alternative to param_update_builder_fun, allows one
                        to add an optimizer for the whole model. Called only
                        once, without name or devicescope.
      net_transformer_fun:
                        Optional function to transform the network after the
                        network is built. It will be called once (NOT once per
                        GPU.)
                        Signature:
                        net_transformer_fun(
                            model, num_devices, device_prefix, device_type)
      pre_grad_net_transformer_fun:
                        Optional function to transform the network similar to
                        net_transformer_fun, but happens before gradient ops
                        been add.
                        Signature: pre_grad_net_transformer_fun(model)
      post_sync_builder_fun:
                        Function applied after initial parameter sync has been
                        completed, such as keeping multi-precision parameters
                        in sync.
                        Signature: post_sync_builder_fun(model)
      devices:          List of GPU ids, such as [0, 1, 2, 3],
      rendezvous:       used for rendezvous in distributed computation, if None
                        then only one node is used. To create rendezvous,
                        use <TBD>.
      net_type:         Network type
      optimize_gradient_memory: whether to apply 'memonger' to share blobs
      shared_model      (only for CPU) use same parameters on each device
                        in gradient computation to reduce memory footprint.
      dynamic_memory_management: Whether to apply dynamic memory optimization
                        by freeing unused blobs. The underlying (de)allocation
                        uses cached allocator. For GPU training PLEASE MAKE SURE
                        caffe2_cuda_memory_pool is set.
      blobs_to_keep :   A list of blob names to keep and don't free during
                        dynamic memory optimization (for example loss blob).
      cpu_device        Use CPU instead of GPU.
      ideep             Use ideep.
      combine_spatial_bn:
                        When set to True, applies batch normalization across
                        all devices within the node. If False, batch
                        normalization will be done separately for each device.
                        This option is currently only supported on the CPU.
      barrier_net_timeout_sec:
                        The timeout in seconds of the barrier net, which is run
                        to synchronize shards before a training epoch starts.
                        Defaults to 300 seconds.
    '''
    assert scope.CurrentDeviceScope() is None \
        or scope.CurrentDeviceScope().device_type == caffe2_pb2.CPU, \
        "Parallelize must be called without device-scope, \
        device scope was: {}".format(scope.CurrentDeviceScope())

    if devices is None:
        if not (cpu_device or ideep):
            devices = list(range(0, workspace.NumCudaDevices()))
        else:
            devices = list(range(0, cpu_count()))

    if not (cpu_device or ideep):
        for gpu in devices:
            if gpu >= workspace.NumGpuDevices():
                log.warning("** Only {} GPUs available, GPUs {} requested".format(
                    workspace.NumGpuDevices(), devices))
                break
        model_helper_obj._device_type = workspace.GpuDeviceType
        model_helper_obj._device_prefix = "gpu"
        model_helper_obj._shared_model = False
        device_name = "GPU"
        assert shared_model is False, "Shared model only supported on CPU"
    elif ideep:
        model_helper_obj._device_type = caffe2_pb2.IDEEP
        model_helper_obj._device_prefix = "ideep"
        device_name = "IDEEP"
        model_helper_obj._shared_model = shared_model
        if shared_model and rendezvous is not None:
            assert "Shared model only supported on single-node currently"
    else:
        model_helper_obj._device_type = caffe2_pb2.CPU
        model_helper_obj._device_prefix = "cpu"
        device_name = "CPU"
        model_helper_obj._shared_model = shared_model
        if shared_model and rendezvous is not None:
            assert "Shared model only supported on single-node currently"

    log.info("Parallelizing model for devices: {}".format(devices))
    extra_workers = 8 if rendezvous is not None else 0  # best-guess
    num_workers = len(devices) * num_threads_per_device + extra_workers
    max_concurrent_distributed_ops =\
        min(max_concurrent_distributed_ops, num_workers - 1)
    model_helper_obj.net.Proto().num_workers = num_workers
    model_helper_obj.net.Proto().type = net_type

    # Store some information in the model -- a bit ugly
    model_helper_obj._devices = devices
    model_helper_obj._rendezvous = rendezvous
    model_helper_obj._sync_barrier_net = None

    model_helper_obj._broadcast_context = None
    model_helper_obj._grad_names = []

    assert isinstance(model_helper_obj, model_helper.ModelHelper)

    # Keep track of params that were in the model before: they are not
    # data parallel, so we need to handle them separately
    non_datapar_params = copy.copy(model_helper_obj.params)

    # Add input and model
    log.info("Create input and model training operators")

    losses_by_gpu = {}
    num_shards = 1 if rendezvous is None else rendezvous['num_shards']
    loss_scale = 1.0 / (len(devices) * num_shards)

    has_parameter_updates = param_update_builder_fun is not None or \
        optimizer_builder_fun is not None
    assert not (
        param_update_builder_fun is not None and
        optimizer_builder_fun is not None
    ), 'Can only specify one of param_update_builder_fun, optimizer_builder_fun'

    # Check that a model that is used for validation/testing has
    # init_params False, otherwise running the param init net will overwrite
    # synchronized values by the training net
    if not has_parameter_updates and model_helper_obj.init_params:
        log.warning('')
        log.warning("############# WARNING #############")
        log.warning("Model {}/{} is used for testing/validation but".format(
            model_helper_obj.name, model_helper_obj))
        log.warning("has init_params=True!")
        log.warning("This can conflict with model training.")
        log.warning("Please ensure model = ModelHelper(init_params=False)")
        log.warning('####################################')
        log.warning('')
        # TODO: make into assert

    for device in devices:
        device_opt = core.DeviceOption(model_helper_obj._device_type, device)
        with core.DeviceScope(device_opt):
            with core.NameScope("{}_{}".format(model_helper_obj._device_prefix,
                                               device)):
                log.info("Model for {} : {}".format(device_name, device))
                input_builder_fun(model_helper_obj)
                losses = forward_pass_builder_fun(model_helper_obj, loss_scale)
                # Losses are not needed for test net
                if has_parameter_updates:
                    assert isinstance(losses, list), \
                        'Model builder function must return list of loss blobs'
                    for loss in losses:
                        assert isinstance(loss, core.BlobReference), \
                            'Model builder func must return list of loss blobs'

                losses_by_gpu[device] = losses
    _ValidateParams(model_helper_obj.params)

    # Create parameter map
    model_helper_obj._device_grouped_blobs =\
        _GroupByDevice(model_helper_obj, devices,
                       model_helper_obj.params, non_datapar_params)

    # computed params
    computed_params_grouped =\
        _GroupByDevice(model_helper_obj, devices,
                       model_helper_obj.GetComputedParams(''), [])
    model_helper_obj._device_grouped_blobs.update(computed_params_grouped)

    model_helper_obj._param_names =\
        list(viewkeys(model_helper_obj._device_grouped_blobs))
    model_helper_obj._computed_param_names =\
        list(viewkeys(computed_params_grouped))

    if pre_grad_net_transformer_fun:
        pre_grad_net_transformer_fun(model_helper_obj)

    if has_parameter_updates:
        log.info("Adding gradient operators")
        _AddGradientOperators(devices, model_helper_obj, losses_by_gpu)

    if net_transformer_fun:
        net_transformer_fun(
            model_helper_obj,
            len(devices),
            model_helper_obj._device_prefix,
            model_helper_obj._device_type)

    if not has_parameter_updates:
        log.info("Parameter update function not defined --> only forward")
        _InferBlobDevice(model_helper_obj)
        return

    if combine_spatial_bn:
        assert(has_parameter_updates), \
            'combine_spatial_bn should only be used for train model'
        _InterleaveOps(model_helper_obj)
        if cpu_device:
            _CPUInterDeviceBatchNormalization(model_helper_obj)
        else:
            _GPUInterDeviceBatchNormalization(model_helper_obj)

    _ValidateParams(model_helper_obj.params)

    # Group gradients by device and register to blob lookup
    param_to_grad = model_helper_obj.param_to_grad
    grads_ordered = [param_to_grad[p] for p in
                     model_helper_obj.params if p in param_to_grad]
    non_datapar_grads = [param_to_grad[p] for p in non_datapar_params]

    gradients_grouped = _GroupByDevice(
        model_helper_obj,
        devices,
        grads_ordered,
        non_datapar_grads
    )
    model_helper_obj._device_grouped_blobs.update(gradients_grouped)
    model_helper_obj._grad_names = list(viewkeys(gradients_grouped))
    model_helper_obj._losses_by_gpu = losses_by_gpu

    _InferBlobDevice(model_helper_obj)

    log.info("Add gradient all-reduces for SyncSGD")
    if broadcast_computed_params:
        _BroadcastComputedParams(devices, model_helper_obj, rendezvous, use_nccl)

    if len(model_helper_obj._grad_names) > 0:
        # Gradients in reverse order
        reverse_ordered_grads = _GetReverseOrderedGrads(model_helper_obj)
        assert(len(reverse_ordered_grads) > 0)
        _AllReduceBlobs(
            reverse_ordered_grads,
            devices,
            model_helper_obj,
            model_helper_obj.net,
            rendezvous,
            use_nccl,
            max_concurrent_distributed_ops,
        )
    else:
        log.info("NOTE: Param builder function did not create any parameters.")

    log.info("Post-iteration operators for updating params")
    num_shards = 1 if rendezvous is None else rendezvous['num_shards']

    all_params = set(model_helper_obj.GetParams(''))
    if shared_model:
        _PruneParametersForSharing(model_helper_obj)

    if param_update_builder_fun is not None:
        for device in devices:
            device_opt = core.DeviceOption(model_helper_obj._device_type, device)
            with core.DeviceScope(device_opt):
                with core.NameScope(
                    "{}_{}".format(model_helper_obj._device_prefix, device)
                ):
                    param_update_builder_fun(model_helper_obj)
    else:
        log.info("Calling optimizer builder function")
        optimizer = optimizer_builder_fun(model_helper_obj)
        model_helper_obj._optimizer = optimizer

    (sync_blobs, sync_names) = _ComputeBlobsToSync(model_helper_obj)
    sync_blobs_grouped = _GroupByDevice(
        model_helper_obj,
        devices,
        sync_blobs,
        [],
    )
    model_helper_obj._device_grouped_blobs.update(sync_blobs_grouped)

    _InferBlobDevice(model_helper_obj)
    _AnalyzeOperators(model_helper_obj)

    # Configure dagnet to run with only one worker on the first iteration,
    # to prevent concurrency problems with allocs and nccl.
    arg = model_helper_obj.Proto().arg.add()
    arg.name = "first_iter_only_one_worker"
    arg.i = 1

    # Add initial parameter syncs
    log.info("Add initial parameter sync")
    _SyncAllParams(
        devices,
        model_helper_obj,
        model_helper_obj.param_init_net,
        model_helper_obj.param_init_net,
        rendezvous,
        sync_names,
        max_concurrent_distributed_ops=1
    )

    # Handle any operations that need to be done after parameter sync
    # i.e. making sure multi-precision copies of parameters are up-to-date
    if post_sync_builder_fun is not None:
        for device in devices:
            device_opt = core.DeviceOption(model_helper_obj._device_type, device)
            with core.DeviceScope(device_opt):
                with core.NameScope(
                    "{}_{}".format(model_helper_obj._device_prefix, device)
                ):
                    post_sync_builder_fun(model_helper_obj)

    assert not (optimize_gradient_memory and dynamic_memory_management), \
        """It is not advised to use gradient optimization ('memonger')
        with dynamic memory management."""

    if optimize_gradient_memory:
        _OptimizeGradientMemorySimple(model_helper_obj, losses_by_gpu, devices)

    if dynamic_memory_management:
        _AddDynamicMemoryOptimization(model_helper_obj, blobs_to_keep, devices)


    model_helper_obj._data_parallel_model_init_nets = [
        model_helper_obj.param_init_net,
    ]

    model_helper_obj._data_parallel_model_nets = [
        model_helper_obj.net
    ]
    _AddBarrierToModelNets(model_helper_obj, barrier_net_timeout_sec)

    if shared_model:
        _RemapParameterBlobsForSharedModel(model_helper_obj, all_params)


def Parallelize_GPU_BMUF(*args, **kwargs):
    kwargs['cpu_device'] = False
    Parallelize_BMUF(*args, **kwargs)


def Parallelize_CPU_BMUF(*args, **kwargs):
    kwargs['cpu_device'] = True
    Parallelize_BMUF(*args, **kwargs)


def Parallelize_BMUF(
    model_helper_obj,
    input_builder_fun,
    forward_pass_builder_fun,
    param_update_builder_fun,
    block_learning_rate=1.0,
    block_momentum=None,
    devices=None,
    rendezvous=None,
    net_type='dag',
    master_device=None,
    use_nccl=False,
    nesterov=False,
    optimize_gradient_memory=False,
    reset_momentum_sgd=False,
    warmup_iterations=None,
    max_concurrent_distributed_ops=4,
    add_blobs_to_sync=None,
    num_threads_per_device=4,
    cpu_device=False,
    barrier_net_timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC,
):
    '''
    Function to create model that run on many GPUs and creates a net for
    parameter_updates that can be run independently for number of iterations
    then followed by another net that runs once to compute the final parameter
    updates according to block wise model update filtering rule described
    in : Scalable Training of Deep Learning Machines by Incremental Block
    Training with Intra-block Parallel Optimization and Blockwise Model-Update
    Filtering (ICASSP 2016).
    '''
    assert scope.CurrentDeviceScope() is None \
        or scope.CurrentDeviceScope().device_type == caffe2_pb2.CPU, \
        "Parallelize must be called without device-scope, \
        device scope was: {}".format(scope.CurrentDeviceScope())

    assert isinstance(model_helper_obj, model_helper.ModelHelper)

    if devices is None:
        devices = list(range(0, workspace.NumGpuDevices()))
    if master_device is None:
        master_device = devices[0]

    if not cpu_device:
        for gpu in devices:
            if gpu >= workspace.NumGpuDevices():
                log.warning("** Only {} GPUs available, GPUs {} requested".format(
                    workspace.NumGpuDevices(), devices))
                break
        model_helper_obj._device_type = workspace.GpuDeviceType
        model_helper_obj._device_prefix = "gpu"
    else:
        model_helper_obj._device_type = caffe2_pb2.CPU
        model_helper_obj._device_prefix = "cpu"

    model_helper_obj._devices = devices
    model_helper_obj._rendezvous = rendezvous
    model_helper_obj._sync_barrier_net = None
    model_helper_obj._broadcast_context = None
    model_helper_obj._shared_model = False
    master_dev_opt = core.DeviceOption(model_helper_obj._device_type, master_device)

    # question: rendezvous structure
    num_shards = rendezvous['num_shards'] if rendezvous else 1
    # num_devices is #devices across all machines
    num_devices = len(devices) * num_shards
    # num_workers is #threads to execute the DAG per shard
    num_workers = num_threads_per_device * len(devices)
    if rendezvous:
        num_workers += 8

    loss_scale = 1.0 / num_devices
    if block_momentum is None:
        block_momentum = 1.0 - 1.0 / num_devices

    max_concurrent_distributed_ops = min(
        max_concurrent_distributed_ops,
        num_workers - 1
    )

    model_helper_obj.net.Proto().num_workers = num_workers
    model_helper_obj.net.Proto().type = net_type

    # A net for initializing global model parameters. Its called once in the
    # same step as net parameters initialization.
    model_helper_obj._global_model_init_net = core.Net('global_model_init')
    model_helper_obj._global_model_init_net.Proto().type = net_type
    model_helper_obj._global_model_init_net.Proto().num_workers = \
        num_workers

    # A net for computing final parameter updates. Its will run once after
    # running net (local models updates) for `num_local_iterations` times.
    model_helper_obj._global_model_param_updates_net = core.Net('global_model')
    model_helper_obj._global_model_param_updates_net.Proto().type = net_type
    model_helper_obj._global_model_param_updates_net.Proto().num_workers = \
        num_workers

    def _v(param):
        return "{}_v".format(param)

    def _g(param):
        return "{}_g".format(param)

    def _v_prev(param):
        return "{}_prev".format(param)

    # Keep track of params that were in the model before: they are not
    # data parallel, so we need to handle them separately
    non_datapar_params = copy.copy(model_helper_obj.params)
    model_helper_obj._losses_by_gpu = {}

    def _InitializeModels(gpu_id):
        input_builder_fun(model_helper_obj)
        loss = forward_pass_builder_fun(model_helper_obj, loss_scale)
        model_helper_obj._losses_by_gpu[gpu_id] = loss
    _ForEachDevice(
        devices,
        _InitializeModels,
        device_type=model_helper_obj._device_type,
        device_prefix=model_helper_obj._device_prefix,
        scoped=True
    )
    _ValidateParams(model_helper_obj.params)

    model_helper_obj._device_grouped_blobs =\
        _GroupByDevice(model_helper_obj, devices,
                       model_helper_obj.params, non_datapar_params)

    model_helper_obj._param_names =\
        list(viewkeys(model_helper_obj._device_grouped_blobs))

    _AddGradientOperators(
        devices, model_helper_obj, model_helper_obj._losses_by_gpu
    )
    _ValidateParams(model_helper_obj.params)

    _InferBlobDevice(model_helper_obj)

    def _InitializeParamUpdate(gpu_id):
        param_update_builder_fun(model_helper_obj)
    _ForEachDevice(
        devices,
        _InitializeParamUpdate,
        device_type=model_helper_obj._device_type,
        device_prefix=model_helper_obj._device_prefix,
        scoped=True
    )

    model_parameter_names = list(
        viewkeys(model_helper_obj._device_grouped_blobs)
    )
    if warmup_iterations is not None:
        model_helper_obj._warmup_iterations = warmup_iterations
        # A net for broadcasting gpu-0 (master shard) parameters after
        # running net for `warmup_iterartions`.
        model_helper_obj._warmup_broadcast = core.Net('warmup-broadcast')
        model_helper_obj._warmup_broadcast.Proto().type = net_type
        model_helper_obj._warmup_broadcast.Proto().num_workers = \
           num_workers

        _SyncAllParams(
            devices,
            model_helper_obj,
            model_helper_obj.param_init_net,
            model_helper_obj._warmup_broadcast,
            rendezvous,
            model_parameter_names,
            max_concurrent_distributed_ops
        )
        for param_name in viewkeys(model_helper_obj._device_grouped_blobs):
            param = model_helper_obj._device_grouped_blobs[param_name][master_device]
            with core.DeviceScope(master_dev_opt):
                model_helper_obj._warmup_broadcast.Copy(param, _g(param))

    # (Step-0) Initialize momentum parameters on master device.
    for param_name in viewkeys(model_helper_obj._device_grouped_blobs):
        param = model_helper_obj._device_grouped_blobs[param_name][master_device]
        with core.DeviceScope(master_dev_opt):
            model_helper_obj._global_model_init_net.ConstantFill(
                param, _v(param), value=0.0
            )
            model_helper_obj._global_model_init_net.Copy(param, _g(param))
            if nesterov:
                model_helper_obj._global_model_init_net.ConstantFill(
                    param, _v_prev(param), value=0.0
                )

    # (Step-1) Update models for num_local_iterations.

    # (Step-2) Compute post-local-updates average of the params.
    # Sum model params across GPUs and store resutls in param_avg blob.
    _AllReduceBlobs(
        model_parameter_names,
        devices,
        model_helper_obj,
        model_helper_obj._global_model_param_updates_net,
        rendezvous,
        use_nccl,
        max_concurrent_distributed_ops
    )

    # (Step-3) Update momentum params :
    # param_v = block_momentum * param_v
    # + block_learning_Rate * (param_avg - param)
    # if nesterov momentum:
    # param = param + param_v
    # - block_momentum * (param_v - param_v_prev)
    # param_v_prev = param_v
    # else:
    # param = param + param_v
    for param_name in model_parameter_names:
        param = model_helper_obj._device_grouped_blobs[param_name][master_device]
        with core.DeviceScope(master_dev_opt):
            # TODO(ataei) : Stop building the graph here to get model average ?
            model_helper_obj._global_model_param_updates_net.Scale(
                param, param, scale=1.0 / num_devices
            )
            model_helper_obj._global_model_param_updates_net.Sub(
                [param, _g(param)], param
            )
            model_helper_obj._global_model_param_updates_net.Scale(
                param, param, scale=block_learning_rate
            )
            model_helper_obj._global_model_param_updates_net.Scale(
                _v(param), _v(param), scale=block_momentum
            )
            model_helper_obj._global_model_param_updates_net.Add(
                [_v(param), param], _v(param)
            )
            model_helper_obj._global_model_param_updates_net.Add(
                [_g(param), _v(param)], _g(param)
            )
            if nesterov:
                model_helper_obj._global_model_param_updates_net.Sub(
                    [_v(param), _v_prev(param)], _v_prev(param)
                )
                model_helper_obj._global_model_param_updates_net.Scale(
                    _v_prev(param), _v_prev(param), scale=block_momentum
                )
                model_helper_obj._global_model_param_updates_net.Sub(
                    [_g(param), _v_prev(param)], _g(param)
                )
                model_helper_obj._global_model_param_updates_net.Copy(
                    _v(param), _v_prev(param)
                )
            model_helper_obj._global_model_param_updates_net.Copy(
                _g(param), param
            )


    _SyncAllParams(
        devices,
        model_helper_obj,
        model_helper_obj.param_init_net,
        model_helper_obj._global_model_param_updates_net,
        rendezvous,
        model_parameter_names,
        max_concurrent_distributed_ops
    )

    # Add additional syncs
    if add_blobs_to_sync is not None:
        AddBlobSync(
            model_helper_obj,
            add_blobs_to_sync,
            net=model_helper_obj._global_model_param_updates_net)

    # Reset momentum-SGD parameters
    if reset_momentum_sgd:
        momentum_ops = [op for op in model_helper_obj.net.Proto().op
                        if op.type == 'MomentumSGDUpdate']
        for op in momentum_ops:
            momentum_blob = op.input[1]
            with core.DeviceScope(op.device_option):
                model_helper_obj._global_model_param_updates_net.ConstantFill(
                    [momentum_blob], momentum_blob, value=0.0
                )

    if optimize_gradient_memory:
        _OptimizeGradientMemorySimple(
            model_helper_obj, model_helper_obj._losses_by_gpu, devices
        )

    model_helper_obj._data_parallel_model_init_nets = [
        model_helper_obj.param_init_net,
        model_helper_obj._global_model_init_net
    ]

    model_helper_obj._data_parallel_model_nets = [
        model_helper_obj.net,
        (model_helper_obj._global_model_param_updates_net, 1)
    ]
    _AddBarrierToModelNets(model_helper_obj, barrier_net_timeout_sec)

def CreateNet(model, overwrite=False):
    for net_iters in model._data_parallel_model_nets:
        if isinstance(net_iters, tuple):
            workspace.CreateNet(net_iters[0], overwrite=overwrite)
        else:
            workspace.CreateNet(net_iters, overwrite=overwrite)


def RunInitNet(model):
    for init_net in model._data_parallel_model_init_nets:
        workspace.RunNetOnce(init_net)
    CreateNet(model)


def RunWarmup(model):
    workspace.RunNet(model.net, model._warmup_iterations)
    workspace.RunNetOnce(model._warmup_broadcast)


def RunNet(model, num_iterations):
    for net_iter in model._data_parallel_model_nets:
        if isinstance(net_iter, tuple):
            workspace.RunNet(net_iter[0].Proto().name, net_iter[1])
        else:
            workspace.RunNet(net_iter, num_iterations)


def _AddBarrierToModelNets(model, barrier_net_timeout_sec):
    if model._rendezvous is not None and model._rendezvous['engine'] == 'GLOO':
        # Synchronize DPM at the start of each epoch. This allows shards that
        # starts an epoch sooner to wait for slower shards.  Without this,
        # shards that are faster than others will begin training the next epoch
        # while stragglers are blocked on IO, and may timeout after 30 seconds
        # (_DEFAULT_TIMEOUT_SEC).
        # We pass in model.param_init_net so that the barrier net can be run as
        # part of the param_init_net.

        model._barrier_init_net = core.Net("barrier_init_net")

        model._barrier_net = _CreateBarrierNet(model, model._barrier_init_net,
        "pre_training", barrier_net_timeout_sec)

        model._data_parallel_model_init_nets.insert(0, model._barrier_init_net)

        model._data_parallel_model_nets.insert(0, model._barrier_net)


def _CreateBarrierNet(model, init_net, name_prefix, timeout_sec):
    log.info("Creating barrier net")
    assert model._rendezvous['engine'] == 'GLOO', "Engine does not support barrier"
    comm_world = _CreateOrCloneCommonWorld(
        init_net,
        name_prefix + "_barrier_cw",
        rendezvous=model._rendezvous,
        timeout_sec=timeout_sec,
    )
    barrier_net = core.Net(name_prefix + "_barrier_net")
    barrier_net.Barrier(
        inputs=[comm_world],
        outputs=[],
        engine=model._rendezvous['engine'],
    )
    return barrier_net


# DEPRECATED: See warnings below.
def Synchronize(model, timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC):
    warnings.warn("The Synchronize API has been deprecated.  We now have a "
            "barrier net which runs before training to ensure all hosts wait "
            "before training starts.  The default timeout for the barrier is "
            "300s and it can be overridden using the barrier_net_timeout_sec "
            "parameter when calling Parallelize.",
            category=DeprecationWarning, stacklevel=2)
    if model._rendezvous is None or model._rendezvous['num_shards'] <= 1:
        # Single host case
        return

    if model._sync_barrier_net is None:
        barrier_init_net = core.Net("sync_barrier_init_net")
        model._sync_barrier_net = _CreateBarrierNet(
            model, barrier_init_net, "sync", timeout_sec)
        workspace.RunNetOnce(barrier_init_net)
        workspace.CreateNet(model._sync_barrier_net)
        model._sync_barrier_net_timeout = timeout_sec
    assert model._sync_barrier_net_timeout == timeout_sec, \
        "Must use fixed timeout, {} != {}".format(
            model._sync_barrier_net_timeout, timeout_sec
        )
    log.info("Synchronize run barrier net.")
    workspace.RunNet(model._sync_barrier_net)


def ConvertNetForDevice(net, device=None):
    '''
    Converts all blobs in the net to have namescope gpu_X, and correct
    device scope. You can use this to enable AppendNet with a
    forward_pass_builder_fun:

       def builder_fun(model):
          ...
          model.net.AppendNet(
             data_parallel_model.ConvertNetForDevice(othermodel.net))
          model.param_init_net.AppendNet(
             data_parallel_model.ConvertNetForDevice(othermodel.param_init_net))
    '''
    mnet = copy.deepcopy(net)

    if device is None:
        device = scope.CurrentDeviceScope()
    if core.IsGPUDeviceType(device.device_type):
        device_prefix = "gpu"
    elif device.device_type == caffe2_pb2.IDEEP:
        device_prefix = "ideep"
    else:
        device_prefix = "cpu"

    namescope = "{}_{}/".format(device_prefix, device.device_id)
    for op in mnet.Proto().op:
        if "RecurrentNetwork" in op.type:
            raise NotImplementedError("RecurrentNetwork conversion not yet supported")
        for i, inputb in enumerate(op.input):
            op.input[i] = namescope + inputb
        for i, outputb in enumerate(op.output):
            op.output[i] = namescope + outputb
        for i, blob in enumerate(op.control_input):
            op.control_input[i] = namescope + blob
        op.device_option.CopyFrom(device)
    for i, einp in enumerate(mnet.Proto().external_input):
        mnet.Proto().external_input[i] = namescope + einp
    for i, eoutp in enumerate(mnet.Proto().external_output):
        mnet.Proto().external_output[i] = namescope + eoutp
    return mnet


def _ForEachDevice(devices, f, device_type, device_prefix, scoped=False,
                   *args, **kwargs):
    for device in devices:
        device_opt = core.DeviceOption(device_type, device)
        with core.DeviceScope(device_opt):
            if scoped:
                with core.NameScope("{}_{}".format(device_prefix, device)):
                    f(device, *args, **kwargs)
            else:
                f(device, *args, **kwargs)


def _AddGradientOperators(devices, model, losses_by_gpu):
    def create_grad(lossp):
        return model.ConstantFill(lossp, str(lossp) + "_grad", value=1.0)

    loss_grad = {}
    # Explicitly need to create gradients on each GPU
    for gpu_id in devices:
        device = core.DeviceOption(model._device_type, gpu_id)
        with core.DeviceScope(device):
            for l in losses_by_gpu[gpu_id]:
                lg = create_grad(l)
                loss_grad[str(l)] = str(lg)

    model.AddGradientOperators(loss_grad)


def ExtractPredictorNet(model, inputs, outputs, device):
    '''
    Returns (net, params) that can be exported to be used as a prediction
    net.
    '''
    master_device = model._devices[0]
    prefix = "{}_{}/".format(model._device_prefix, master_device)
    prefix_inputs = [prefix + str(b) for b in inputs]
    prefix_outputs = [prefix + str(b) for b in outputs]
    (predictor_net, export_blobs) = model_helper.ExtractPredictorNet(
        net_proto=model.net.Proto(),
        input_blobs=prefix_inputs,
        output_blobs=prefix_outputs,
        device=device,
        renames={
            a: b
            for (a, b) in zip(prefix_inputs + prefix_outputs, inputs + outputs)
        },
    )

    return (predictor_net, export_blobs)


def GetCheckpointParams(model):
    '''
    Returns a set of blobs that are needed for a complete check point.
    They are blobs for the first gpu and iteration blobs.
    '''
    (all_blobs, _) = _ComputeBlobsToSync(model)
    first_gpu_blobs = {
        b
        for b in all_blobs
        if str(b)
        .startswith("{}_{}/".format(model._device_prefix, model._devices[0]))
    }

    # Add iteration blobs that do not have namescope separately, since
    # it is important to checkpoint iteration counter
    iteration_blobs = set()
    for op in model.net.Proto().op:
        if op.type == 'Iter' or op.type == 'AtomicIter':
            if not op.output[0].startswith("{}_".format(model._device_prefix)):
                iteration_blobs.add(op.output[0])

    return first_gpu_blobs.union(iteration_blobs)


def FinalizeAfterCheckpoint(model, blobs=None, cpu_mode=False):
    '''
    This function should be called after loading parameters from a
    checkpoint / initial parameters file.
    '''

    if not hasattr(model, "_checkpoint_net"):
        if blobs is None:
            (_, uniq_blob_names) = _ComputeBlobsToSync(model)
        else:
            uniq_blob_names = [stripBlobName(p) for p in blobs]

        # Synchronize to the blob lookup map, as the provided
        # blobs might have non-parameters, such as momentum blobs.
        log.info("Creating checkpoint synchronization net")
        devices = model.GetDevices()
        for name in uniq_blob_names:
            if name not in model._device_grouped_blobs:
                grouped = {
                    d:
                    core.BlobReference("{}_{}{}{}".format(
                        model._device_prefix,
                        d,
                        scope._NAMESCOPE_SEPARATOR,
                        name)
                    ) for d in devices}
                model._device_grouped_blobs[name] = grouped

        model._checkpoint_net = core.Net("checkpoint_sync_net")
        if not cpu_mode:
            model._checkpoint_net.RunAllOnGPU()

        checkpoint_init_net = None
        if (model._rendezvous is not None and model._rendezvous['num_shards'] > 1):
            checkpoint_init_net = core.Net("checkpoint_init_net")
            if not cpu_mode:
                checkpoint_init_net.RunAllOnGPU()

        _SyncAllParams(
            devices,
            model,
            checkpoint_init_net,
            model._checkpoint_net,
            model._rendezvous,
            uniq_blob_names,
            max_concurrent_distributed_ops=1
        )
        if (checkpoint_init_net):
            workspace.RunNetOnce(checkpoint_init_net)

        workspace.CreateNet(model._checkpoint_net)

    # Run the sync
    log.info("Run checkpoint net")
    workspace.RunNet(model._checkpoint_net.Proto().name)


def GetLearningRateBlobNames(model):
    '''
    Returns a list of learning rates blob names used in the optimizer.
    '''
    if model._optimizer is not None:
        if model._device_type == caffe2_pb2.CPU or model._device_type == caffe2_pb2.IDEEP:
            return [model._optimizer.get_cpu_blob_name('lr')]
        elif core.IsGPUDeviceType(model._device_type):
            return [model._optimizer.get_gpu_blob_name('lr', gpu, '')
                    for gpu in model._devices]
        else:
            raise Exception(
                "Unsupported device type : {}".format(model._device_type)
            )
    else:
        lr_blob_names = []
        for op in model.net.Proto().op:
            if op.type == "LearningRate":
                lr_blob_names.append(op.output(0))
        return lr_blob_names


def _Broadcast(devices, model, net, param, use_nccl=False):
    # Copy params from gpu_0 to other
    master_dev = devices[0]

    if use_nccl:
        if _IsGPUBlob(model, param):
            master_device_opt = core.DeviceOption(model._device_type, master_dev)
            with core.DeviceScope(master_device_opt):
                # Note that the root is the root _rank_ and not the root
                # _device_. Thus we always use root=0, regardless of the
                # devices used.
                net.NCCLBroadcast(
                    list(viewvalues(model._device_grouped_blobs[param])),
                    list(viewvalues(model._device_grouped_blobs[param])),
                    root=0,
                )
                return

    for dev_idx in devices[1:]:
        if _IsGPUBlob(model, param):
            device_opt = core.DeviceOption(workspace.GpuDeviceType, dev_idx)
        else:
            device_opt = core.DeviceOption(caffe2_pb2.IDEEP, 0) if _IsIDEEPBlob(model, param) else \
                core.DeviceOption(caffe2_pb2.CPU, 0)
        with core.DeviceScope(device_opt):
            net.Copy(
                model._device_grouped_blobs[param][master_dev],
                model._device_grouped_blobs[param][dev_idx]
            )


def _AllReduce(devices, model, net, param, use_nccl=False, control_input=None):
    blobs_group = list(viewvalues(model._device_grouped_blobs[param]))
    if model._device_type == caffe2_pb2.CUDA and use_nccl:
        # TODO: for _shared_model, do only NCCLReduce
        model.NCCLAllreduce(
            blobs_group, blobs_group, control_input=control_input
        )
        return

    if model._device_type == workspace.GpuDeviceType:
        p2p_access_pattern = workspace.GetGpuPeerAccessPattern()
    else:
        p2p_access_pattern = None

    def sumN(*dev_indices):
        """Create a Sum op for 2 or more blobs on different devices.
        Saves the result on the first device.

        Args:
        dev_indices -- a list of device indices, which can be translated into
                       CUDA identifiers with model._devices
        """
        devices = [model._devices[idx] for idx in dev_indices]
        blobs = [blobs_group[idx] for idx in dev_indices]
        device_opt = core.DeviceOption(model._device_type, devices[0])
        with core.DeviceScope(device_opt):
            for i, peer in enumerate(devices):
                if i == 0:
                    continue  # Skip the first device
                if p2p_access_pattern is not None and p2p_access_pattern.size and not p2p_access_pattern[
                    devices[0], peer
                ]:
                    # Copy from peer to d0
                    blobs[i] = model.Copy(
                        blobs[i],
                        'gpu_{}/{}_gpu{}_copy'.format(devices[0], param, peer)
                    )
            net.Sum(blobs, [blobs[0]], name='dpm')

    if len(devices) == 16:
        # Special tree reduction for 16 gpus, TODO generalize like in muji.py
        for j in range(8):
            sumN(j * 2, j * 2 + 1)
        for j in range(4):
            sumN(j * 4, j * 4 + 2)
        for j in range(2):
            sumN(j * 8, j * 8 + 4)
        sumN(0, 8)
    elif len(devices) == 8:
        for j in range(4):
            sumN(j * 2, j * 2 + 1)
        for j in range(2):
            sumN(j * 4, j * 4 + 2)
        sumN(0, 4)
    elif len(devices) == 4:
        sumN(0, 1)
        sumN(2, 3)
        sumN(0, 2)
    else:
        sumN(*range(len(devices)))
    # TODO: for _shared_model, no need to broadcast
    _Broadcast(devices, model, net, param)


def _SyncAllParams(
    devices,
    model,
    init_net,
    net,
    rendezvous,
    unique_param_names,
    max_concurrent_distributed_ops=4
):
    if rendezvous is None or rendezvous['num_shards'] <= 1:
        _SyncAllParamsSingleHost(devices, model, net, unique_param_names)
    else:
        _SyncAllParamsDistributed(
            devices,
            model,
            init_net,
            net,
            rendezvous,
            unique_param_names,
            max_concurrent_distributed_ops
        )


def AddBlobSync(model, blobs, net=None):
    '''
    Sync a blob across devices and hosts
    '''
    if len(blobs) == 0:
        return
    net = model.net if net is None else net
    for b in blobs:
        assert not b.startswith(model._device_prefix), \
            "Provide unprefixed blob name: {}".format(b)
        model._device_grouped_blobs[b] = {
            d: core.BlobReference("{}_{}/{}".format(model._device_prefix, d, b))
            for d in model._devices
        }

    _SyncAllParams(
        model._devices,
        model,
        model.param_init_net,
        net,
        model._rendezvous,
        set(blobs))


def AddDistributedBlobSync(model, blobs):
    '''
    Sync blobs across machines (but not across devices)
    '''
    if model._rendezvous is None:
        return
    synth_name = "_".join([str(b) for b in blobs])
    comm_world = _CreateOrCloneCommonWorld(
        model.param_init_net,
        "blob_sync_cw_" + synth_name,
        rendezvous=model._rendezvous,
    )

    model.net.Allreduce(
        inputs=[comm_world] + blobs,
        outputs=blobs,
        engine=model._rendezvous['engine'],
    )


def _SyncAllParamsDistributed(
    devices,
    model,
    init_net,
    net,
    rendezvous,
    unique_param_names,
    max_concurrent_distributed_ops
):
    assert rendezvous['num_shards'] > 1

    gpu_device_opt = core.DeviceOption(model._device_type, devices[0])
    cpu_device_opt = core.DeviceOption(caffe2_pb2.CPU)
    ideep_device_opt = core.DeviceOption(caffe2_pb2.IDEEP)

    if model._broadcast_context is None:
        model._broadcast_context = CollectivesConcurrencyControl(
            "broadcast",
            max_concurrent_distributed_ops,
            init_net,
            rendezvous
        )
    context = model._broadcast_context

    for param_name in sorted(unique_param_names):
        master_param = model._device_grouped_blobs[param_name][devices[0]]
        params_group = list(viewvalues(model._device_grouped_blobs[param_name]))

        def broadcast(params):
            comm_world, control_input = context.get_control_and_context(params)
            net.Broadcast(
                inputs=[comm_world] + params,
                outputs=params,
                name=param_name,
                engine=rendezvous['engine'],
                control_input=control_input
            )

        device_opt = gpu_device_opt if _IsGPUBlob(
            model, param_name
        ) else ideep_device_opt if _IsIDEEPBlob(model, param_name) else cpu_device_opt

        if rendezvous['engine'] == 'GLOO':
            with core.DeviceScope(device_opt):
                broadcast(params_group)
        else:
            # Copy between GPU and CPU
            with core.DeviceScope(device_opt):
                param_cpu = net.CopyGPUToCPU(
                    master_param,
                    str(master_param) + "cpu"
                )
            with core.DeviceScope(cpu_device_opt):
                broadcast([param_cpu])
            with core.DeviceScope(device_opt):
                net.CopyCPUToGPU(param_cpu, master_param)

            # Broadcast locally
            _Broadcast(devices, model, net, param_name)


def _SyncAllParamsSingleHost(devices, model, net, unique_param_names):
    for param in unique_param_names:
        _Broadcast(devices, model, net, param)


def _AllReduceBlobs(blob_names, devices, model, net, rendezvous, use_nccl,
                    max_concurrent_distributed_ops):
    if rendezvous is None or rendezvous['num_shards'] <= 1:
        _AllReduceBlobsSingleHost(
            blob_names,
            devices,
            model,
            net,
            use_nccl
        )
    else:
        _AllReduceBlobsDistributed(
            blob_names,
            devices,
            model,
            net,
            rendezvous,
            max_concurrent_distributed_ops,
        )


def _PruneParametersForSharing(model):
    assert model._shared_model
    master_prefix = "{}_{}/".format(model._device_prefix, model._devices[0])

    # Remove non-master parameters so that they will not receive parameter
    # update operators.
    model.params = model.GetParams(master_prefix)
    paramset = set(model.params)

    model.param_to_grad = {
        p: model.param_to_grad[p]
        for p in model.param_to_grad if p in paramset
    }
    model.weights = [w for w in model.weights if w in paramset]
    model.biases = [w for w in model.biases if w in paramset]


def _RemapParameterBlobsForSharedModel(model, all_params):
    assert model._shared_model
    master_prefix = "{}_{}/".format(
        model._device_prefix, model._devices[0])
    log.info("Remapping param blobs to master -> {}".format(master_prefix))
    master_params = set(model.GetParams())

    # Remove all but master params
    def modify_ops(net):
        ops = []
        for op in net.Proto().op:
            delete_op = False
            # Delete ops that output non-master version of parameter
            for outp in op.output:
                if outp in all_params and outp not in master_params:
                    delete_op = True
                    log.debug("Delete b/c {}:  {}".format(outp, str(op)))
                    break
            if delete_op:
                continue
            # Remap inputs to point to the master param
            for j, inp in enumerate(op.input):
                if inp in all_params and inp not in master_params:
                    op.input[j] = master_prefix + stripBlobName(inp)
            ops.append(op)
        del net.Proto().op[:]
        net.Proto().op.extend(ops)

    modify_ops(model.param_init_net)
    modify_ops(model.net)


class CollectivesConcurrencyControl(object):
    """
    Creates common worlds (up to max_concurrent_context) and manage the
    sequential execution of collectives that shares the same context with
    cyclic control inputs.
    """
    def __init__(
        self,
        name,
        max_concurrent_context,
        param_init_net,
        rendezvous
    ):
        self.name = name
        self.param_init_net = param_init_net
        self.max_concurrent_context = max_concurrent_context
        self.counter = 0
        self.common_worlds = []
        self.control_inputs = []
        self.rendezvous = rendezvous

    def get_control_and_context(self, control_output_blob):
        common_world, control_input = [None, None]
        current_slot = self.counter % self.max_concurrent_context
        if len(self.common_worlds) < self.max_concurrent_context:
            common_world = _CreateOrCloneCommonWorld(
                self.param_init_net,
                "{}_{}_cw".format(self.name, current_slot),
                rendezvous=self.rendezvous,
            )
            self.common_worlds.append(common_world)
            self.control_inputs.append(control_output_blob)
        else:
            common_world = self.common_worlds[current_slot]
            control_input = self.control_inputs[current_slot]
            self.control_inputs[current_slot] = control_output_blob
        self.counter += 1
        return common_world, control_input


def _AllReduceBlobsDistributed(
    blob_names,
    devices,
    model,
    net,
    rendezvous,
    max_concurrent_distributed_ops,
):
    num_workers = model.net.Proto().num_workers
    assert num_workers > 1, "Please specify more than 1 worker"
    all_reduce_engine = rendezvous['engine']

    master_device_opt = core.DeviceOption(model._device_type, devices[0])

    reducing_device_opt = master_device_opt

    context = CollectivesConcurrencyControl(
        "allreduce",
        max_concurrent_distributed_ops,
        model.param_init_net,
        rendezvous
    )

    nccl_control_blob = None

    for blob_name in blob_names:
        master_blob = model._device_grouped_blobs[blob_name][devices[0]]
        blobs_group = list(viewvalues(model._device_grouped_blobs[blob_name]))

        assert master_blob in blobs_group

        # Remark: NCCLReduce does not support in-place modifications
        # so we need a temporary blob
        reduced_blob = str(master_blob) + "_red"

        def allreduce(blobs, **kwargs):
            with core.DeviceScope(reducing_device_opt):
                comm_world, control_input = \
                    context.get_control_and_context(blobs[0])
                net.Allreduce(
                    inputs=[comm_world] + blobs,
                    outputs=blobs,
                    name=blob_name,
                    engine=all_reduce_engine,
                    control_input=control_input,
                    **kwargs
                )

        if rendezvous['engine'] == 'GLOO':
            # With Gloo cross GPU and cross machine allreduce
            # can be executed in a single operation.
            # Try to use GPUDirect if transport == ibverbs.
            allreduce(
                blobs_group,
                gpu_direct=(rendezvous.get("transport", None) == "ibverbs"),
            )
        else:
            # Step 1: sum blobs from local GPUs to master GPU
            with core.DeviceScope(master_device_opt):
                model.ConstantFill(master_blob, reduced_blob, value=0.0)

                # Temp fix since NCCLReduce does not work
                net.NCCLAllreduce(
                    blobs_group,
                    blobs_group,
                    control_input=nccl_control_blob,
                )
                nccl_control_blob = blobs_group[0]
                net.Copy(master_blob, reduced_blob)

            # Step 2: allreduce between all hosts, between master GPUs
            allreduce([reduced_blob])

            with core.DeviceScope(master_device_opt):
                net.Copy(reduced_blob, master_blob)

            # Step 3: broadcast locally
            _Broadcast(devices, model, net, blob_name)


def _AllReduceBlobsSingleHost(blob_names, devices, model, net, use_nccl):
    """Performs NCCL AllReduce to distribute blobs to all the GPUs."""

    if len(devices) == 1:
        return

    # Now we need to Allreduce blobs on all the GPUs.
    # Pick GPU #0 as a master GPU.
    master_device_opt = core.DeviceOption(model._device_type, devices[0])
    last_out = None
    concatenated_idx = set()

    for blob_name in blob_names:
        # Group by blob_name for reduce.
        blobs_group = list(viewvalues(model._device_grouped_blobs[blob_name]))
        if len(blobs_group) == 1:
            # Non-reducible
            continue
        assert len(blobs_group) == len(devices), \
            "Each GPU from {}, should have a copy of {}.".format(
                devices, blob_name)

        if _IsGPUBlob(model, blob_name):
            with core.DeviceScope(master_device_opt):
                if not isinstance(blobs_group[0], core.GradientSlice):
                    _AllReduce(
                        devices, model, net, blob_name, use_nccl, last_out
                    )
                    # last_out is used to serialize the execution of nccls
                    last_out = blobs_group[0]

                else:
                    # Sparse gradients: all-gather for indices and values
                    master_ns = "{}_{}".format(model._device_prefix, devices[0])
                    '''
                    Skip if we have already copied concatenated indices
                    to the indices of GradientSlice. This happens when two
                    or more grad blobs are gathered with the same indices
                    blob
                    '''
                    skip_idx_concat = False
                    for g in blobs_group:
                        if g.indices in concatenated_idx:
                            skip_idx_concat = True

                    if not skip_idx_concat:
                        grad_idx_concat, _ = net.Concat(
                            [g.indices for g in blobs_group],
                            ["{}/{}_index_concat".format(master_ns, blob_name),
                             "{}/{}_index_splitinfo".format(master_ns, blob_name)],
                            axis=0,
                            name="note:data_parallel_model")

                        for gpu, g in viewitems(model._device_grouped_blobs[blob_name]):
                            device_opt = core.DeviceOption(model._device_type, gpu)
                            with core.DeviceScope(device_opt):
                                model.Copy(grad_idx_concat, g.indices)
                                concatenated_idx.add(g.indices)

                    grad_val_concat, _ = net.Concat(
                        [g.values for g in blobs_group],
                        ["{}/{}_val_concat".format(master_ns, blob_name),
                         "{}/{}_val_splitinfo".format(master_ns, blob_name)],
                        axis=0, name="note:data_parallel_model")

                    for gpu, g in viewitems(model._device_grouped_blobs[blob_name]):
                        device_opt = core.DeviceOption(model._device_type, gpu)
                        with core.DeviceScope(device_opt):
                            model.Copy(grad_val_concat, g.values)

        elif _IsIDEEPBlob(model, blob_name):
            assert not isinstance(blobs_group[0], core.GradientSlice), \
                "Synchronizing gradient slices not supported"
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.IDEEP)):
                net.Sum(blobs_group, [blobs_group[0]])
                if not model._shared_model:
                    _Broadcast(devices, model, net, blob_name)

        else:
            assert not isinstance(blobs_group[0], core.GradientSlice), \
                "Synchronizing gradient slices not supported"
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                # Poor man's allreduce
                net.Sum(blobs_group, [blobs_group[0]])
                if not model._shared_model:
                    _Broadcast(devices, model, net, blob_name)


def _BroadcastComputedParams(devices, model, rendezvous, use_nccl=False):
    if rendezvous is None:
        _BroadcastComputedParamsSingleHost(devices, model, use_nccl)
    else:
        _BroadcastComputedParamsDistributed(devices, model, rendezvous, use_nccl)


def _BroadcastComputedParamsDistributed(
    devices,
    model,
    rendezvous,
    use_nccl=False
):
    _BroadcastComputedParamsSingleHost(devices, model, use_nccl)
    log.warn("Distributed broadcast of computed params is not implemented yet")


def _BroadcastComputedParamsSingleHost(devices, model, use_nccl=False):
    '''
    Average computed params over all devices
    '''
    if len(devices) == 1:
        return

    for param_name in model._computed_param_names:
        # Copy from master to others -- averaging would be perhaps better,
        # but currently NCCLAllReduce is too prone to deadlock
        _Broadcast(devices, model, model.net, param_name, use_nccl)


def _GetReverseOrderedGrads(model):
    '''
    Returns the gradients in reverse order (namespace stripped),
    for the optimal synchronization order.
    '''
    return list(reversed(model._grad_names))


# A helper function to extract a parameter's name
def stripBlobName(param):
    # Format is "a/b/c/d" -> "b/c/d"
    if isinstance(param, core.GradientSlice):
        return stripBlobName(param.indices) + ":" + stripBlobName(param.values)
    else:
        name = str(param)
    return name[name.index(scope._NAMESCOPE_SEPARATOR) + 1:]


def _AnalyzeOperators(model):
    '''
    Look at all the operators and check that they do not cross device scopes
    '''
    for op in model.Proto().op:
        if "NCCL" in op.type or "Copy" in op.type or "Concat" in op.type:
            continue
        if "Sum" == op.type and op.name == "dpm":
            continue
        if "Allreduce" in op.type and "GLOO" in op.engine:
            continue

        op_dev = op.device_option
        op_gpu = op_dev.device_id

        # This avoids failing on operators that are only for CPU
        if not core.IsGPUDeviceType(op_dev.device_type):
            continue

        namescope = "{}_{}/".format(model._device_prefix, op_gpu)
        for inp in list(op.input) + list(op.output):
            if inp.startswith("{}_".format(model._device_prefix)
                             ) and not inp.startswith(namescope):
                raise Exception(
                    "Blob {} of op {}, should have namescope {}. Op: {}".format(
                        inp,
                        op.type,
                        "{}_{}/".format(model._device_prefix, op_gpu),
                        str(op),
                    )
                )


def _InferBlobDevice(model):
    '''
    Assign blob to device option based on the operator outputing it
    '''
    mapping = {}

    def map_ops(proto):
        for op in proto.op:
            device_option = op.device_option
            if op.type == "Iter":
                # Hack for Iters which have blob in CPU context
                device_option = caffe2_pb2.DeviceOption()
                device_option.device_type = caffe2_pb2.CPU
            for b in list(op.input) + list(op.output):
                if b not in mapping:
                    mapping[b] = device_option
            if op.type.startswith('RecurrentNetwork'):
                step_args = [a for a in op.arg if a.name.endswith("step_net")]
                for step_arg in step_args:
                    map_ops(step_arg.n)
    map_ops(model.param_init_net.Proto())
    map_ops(model.net.Proto())
    model._blob_to_device = mapping

def _IsIDEEPBlob(model, blob_name):
    if blob_name in model._blob_to_device:
        return model._blob_to_device[blob_name].device_type == caffe2_pb2.IDEEP
    else:
        blob_name = "{}_{}/{}".format(
            model._device_prefix, model._devices[0], blob_name
        )
        if blob_name not in model._blob_to_device:
            return model._device_type == caffe2_pb2.IDEEP
        return model._blob_to_device[blob_name].device_type == caffe2_pb2.IDEEP

def _IsGPUBlob(model, blob_name):
    if blob_name in model._blob_to_device:
        return core.IsGPUDeviceType(model._blob_to_device[blob_name].device_type)
    else:
        blob_name = "{}_{}/{}".format(
            model._device_prefix, model._devices[0], blob_name
        )
        if blob_name not in model._blob_to_device:
            return core.IsGPUDeviceType(model._device_type)
        return core.IsGPUDeviceType(model._blob_to_device[blob_name].device_type)


def _GroupByDevice(model, devices, params, non_data_params):
    '''
    Groups blobs by device, returning a map of [blobname] = {0: BlobRef, 1: ..}.
    Returns ordered dictionary, ensuring the original order.
    '''
    grouped = OrderedDict()
    # Only consider params that were created to be  "data parallel"
    params = params[len(non_data_params):]

    for _i, p in enumerate(params):
        assert isinstance(p, core.BlobReference) or \
            isinstance(p, core.GradientSlice), \
            "Param {} is not BlobReference or GradientSlice".format(p)

        name = stripBlobName(p)
        gpuid = None

        if isinstance(p, core.BlobReference):
            gpuid = int(p.GetNameScope().split("_")[1].split("/")[0])
            assert "{}_{}/".format(model._device_prefix, gpuid) in p.GetNameScope(),\
                "Param {} expected to have namescope '{}_{}'".format(str(p), model._device_prefix, gpuid)
        else:
            gpuid = int(p.indices.GetNameScope().split("_")[1].split("/")[0])
            assert "{}_{}/".format(model._device_prefix, gpuid) in p.indices.GetNameScope(),\
                "Indices {} expected to have namescope '{}_{}'".format(str(p), model._device_prefix, gpuid)
            assert "{}_{}/".format(model._device_prefix, gpuid) in p.values.GetNameScope(),\
                "Values {} expected to have namescope '{}_{}'".format(str(p), model._device_prefix, gpuid)

        if name not in grouped:
            grouped[name] = {}
        grouped[name][gpuid] = p

    return grouped


def _ValidateParams(params):
    set_params = set(params)
    if len(params) > len(set_params):
        dupes = []
        sp = sorted(params)
        for j, p in enumerate(sp):
            if j > 0 and sp[j - 1] == p:
                dupes.append(p)

        assert len(params) == len(set_params), \
            "Duplicate entries in params: {}".format(dupes)


def _ComputeBlobsToSync(model):
    '''
    We sync all blobs that are generated by param init net and
    are 'data parallel', i.e assigned to a device
    '''
    sync_names = set()

    # We don't sync params if the model is shared
    if model._shared_model:
        blobs_to_sync = [str(p) for p in model.GetComputedParams('')]
        sync_names = [stripBlobName(p) for p in blobs_to_sync]
    else:
        blobs_to_sync = []

        for op in model.param_init_net.Proto().op:
            dp_outputs = [
                o for o in op.output
                if o.startswith("{}_".format(model._device_prefix))
            ]
            sync_names.update([stripBlobName(o) for o in dp_outputs])
            blobs_to_sync.extend(dp_outputs)

        # Sanity check
        diff = set(model._param_names) - sync_names
        assert diff == set(), \
           "Some params not instantiated in param init net: {}".format(diff)

    # Remove duplicates and sort
    prefixlen = len(model._device_prefix) + 1

    def extract_sort_key(b):
        # Sort first based on device id, and then by whole string
        deviceid = int(b[prefixlen:b.index(scope._NAMESCOPE_SEPARATOR)])
        return (deviceid, b)

    blobs_to_sync = sorted(
        list(set(blobs_to_sync)),
        key=extract_sort_key)

    blobs_to_sync = [core.BlobReference(b) for b in blobs_to_sync]
    return (blobs_to_sync, sync_names)


def _OptimizeGradientMemorySimple(model, losses_by_gpu, devices):
    log.warning("------- DEPRECATED API, please use " +
                   "data_parallel_model.OptimizeGradientMemory() ----- ")
    for device in devices:
        namescope = "{}_{}/".format(model._device_prefix, device)
        model.net._net = memonger.share_grad_blobs(
            model.net,
            losses_by_gpu[device],
            set(viewvalues(model.param_to_grad)),
            namescope,
            share_activations=False,
        )


def _AddDynamicMemoryOptimization(model, blobs_to_keep, devices):
    blobs_to_keep_all_devices = set()
    if blobs_to_keep is not None:
        for device in devices:
            for blob_name in blobs_to_keep:
                blobs_to_keep_all_devices.add(
                    "{}_{}/{}".format(model._device_prefix, device, blob_name)
                )

    if model._rendezvous is not None:
        # GLOO operators expect the tensor addresses to remain same over
        # iterations so we need to remove param grads from the dynamic memory
        # management.
        blobs_to_keep_all_devices.update(
            [str(b) for b in viewvalues(model.param_to_grad)]
        )

    model.net._net = memonger.release_blobs_when_used(
        model.net.Proto(),
        blobs_to_keep_all_devices
    )


def OptimizeGradientMemory(model,
                           input_shapes,
                           excluded_blobs,
                           recycle_activations):
    """
    Optimize memory usage of the backward pass by recycling blobs for gradient
    inputs that have been 'used'.
    input_shapes:  dict of blob name to shape for the inputs of the model.
                   Pass empty dictionary if not known.
    excluded_blobs: list of blobs that cannot be recycled. These are blobs
                   that you will access externally.
    recycle_activations: whether to also recycle forward pass activations
    """
    if input_shapes is not None:
        input_shapes_all_devices = {}
        for b, shp in viewitems(input_shapes):
            for d in model._devices:
                input_shapes_all_devices["{}_{}/{}".
                                         format(model._device_prefix, d, b)] = shp

        (shapes, types) = workspace.InferShapesAndTypes(
            [model.param_init_net, model.net],
            input_shapes_all_devices,
        )
    else:
        shapes = None

    for device in model._devices:
        namescope = "{}_{}/".format(model._device_prefix, device)
        excluded_blobs_by_device = set(namescope + b for b in excluded_blobs)
        model.net._net = memonger.share_grad_blobs(
            model.net,
            model._losses_by_gpu[device],
            set(viewvalues(model.param_to_grad)),
            namescope,
            dont_share_blobs=excluded_blobs_by_device,
            share_activations=recycle_activations,
            blob_shapes=shapes,
        )


def _CreateOrCloneCommonWorld(
        net,
        common_world_blob,
        rendezvous,
        name=None,
        timeout_sec=None):

    if timeout_sec is None:
        timeout_sec = _DEFAULT_TIMEOUT_SEC

    timeout_ms = timeout_sec * 1000

    # Check if there is an existing CreateCommonWorld
    # with the same timeout we're looking for. If so,
    # we can clone it instead of creating a new one.
    existing = None
    for op in net.Proto().op:
        if op.type != "CreateCommonWorld":
            continue

        # Find common world timeout
        op_timeout_ms = -1
        for arg in op.arg:
            if arg.name == 'timeout_ms':
                op_timeout_ms = arg.i
                break
        if op_timeout_ms != timeout_ms:
            continue

        # This common world was created with the same timeout we're
        # looking for, so we can clone it
        existing = op.output[0]
        break

    if name is None:
        name = "{}_op".format(common_world_blob)

    if existing is not None:
        comm_world = net.CloneCommonWorld(
            [existing],
            common_world_blob,
            name=name,
            engine=rendezvous['engine'],
        )
    else:
        kwargs=dict()
        if 'transport' in rendezvous:
            kwargs['transport'] = rendezvous['transport']
        if 'interface' in rendezvous:
            kwargs['interface'] = rendezvous['interface']
        if 'mpi_rendezvous' in rendezvous:
            kwargs['mpi_rendezvous'] = rendezvous['mpi_rendezvous']
        comm_world = net.CreateCommonWorld(
            rendezvous['kv_handler'] or [],
            common_world_blob,
            name=name,
            size=rendezvous['num_shards'],
            rank=rendezvous['shard_id'],
            engine=rendezvous['engine'],
            timeout_ms=timeout_ms,
            **kwargs
        )

    return comm_world


def _RunComparison(model, blob_name, device=None):
    if device is None:
        device = model._blob_to_device[blob_name]
    with core.DeviceScope(device):
        rendezvous = model._rendezvous
        if rendezvous is None or rendezvous['num_shards'] == 1:
            return True

        test_data_arr = np.zeros(rendezvous['num_shards']).astype(np.float32)
        test_data_arr[rendezvous['shard_id']] = 1
        workspace.FeedBlob("compare_arr", test_data_arr)

        comparison_net = core.Net("allcompare_net")

        kwargs=dict()
        if 'mpi_rendezvous' in rendezvous:
            kwargs['mpi_rendezvous'] = rendezvous['mpi_rendezvous']
        comm_world = comparison_net.CreateCommonWorld(
            rendezvous['kv_handler'] or [],
            "initial_sync",
            name=model.net.Proto().name + ".cw_master_select",
            size=rendezvous['num_shards'],
            rank=rendezvous['shard_id'],
            engine=rendezvous['engine'],
            **kwargs
        )

        blob_name_checksum = blob_name + "_checksum"
        comparison_net.SumSqrElements(
            [blob_name], [blob_name_checksum], average=False
        )

        blob_name_gather = blob_name + "_gather"
        comparison_net.Mul(
            inputs=["compare_arr", blob_name_checksum],
            outputs=blob_name_gather,
            broadcast=1
        )

        comparison_net.Allreduce(
            inputs=[comm_world, blob_name_gather],
            outputs=[blob_name_gather],
            engine=rendezvous['engine'],
        )

        workspace.RunNetOnce(comparison_net)
        gather_arr = workspace.FetchBlob(blob_name_gather)

        baseline = gather_arr[0]
        for i in range(rendezvous['num_shards']):
            assert gather_arr[i] == baseline, \
                "allcompare failed on shard {}.".format(rendezvous['shard_id'])

        return True


def _InterleaveOps(model):
    '''
    Data Parallel Model creates a net with ops in one device grouped together.
    This will interleave the ops so that each op for each device is next
    to each other in the net. Kind of like combining decks of cards. This
    ensures that progress is made along the critical path roughly concurrently
    for each device, which is important due to the extra intra-node
    synchronization required for multi-device batch normalization.
    '''
    orig_ops = list(model.net.Proto().op)
    num_devices = len(model._devices)
    num_ops_per_dev = len(orig_ops) // num_devices
    assert num_devices * num_ops_per_dev == len(orig_ops), \
           'Number of ops per device in original net is not uniform'
    new_ops = []
    ops = {d: [] for d in range(num_devices)}
    for op in orig_ops:
        ops[op.device_option.device_id].append(op)

    for j in range(num_ops_per_dev):
        tp = None
        for d in model._devices:
            if tp is None:
                tp = ops[d][j].type
            new_ops.append(ops[d][j])
            # Sanity
            assert ops[d][j].type == tp, \
                "Type mismatch {} / {}".format(tp, ops[d][j].type)

    del model.net.Proto().op[:]
    model.net.Proto().op.extend(new_ops)


def _CPUInterDeviceBatchNormalization(model):
    orig_ops = list(model.net.Proto().op)
    new_ops = []
    num_devices = len(model._devices)
    batch_norm_ops = []
    injected_ops = []

    spatial_bn_phase = False
    sums_blobs = []
    sumsq_blobs = []
    name = []
    input_blob_name = None

    spatial_bn_gradient_phase = False
    scale_grad_blobs = []
    bias_grad_blobs = []

    def _cpuReduce(param, input_blobs, destination_blobs):
        """
        Reduce results from multiple cpus and distributes the results back
        to each device. This is done by copying values to cpu_0 and summing
        them. The cpu_0 result is then copied back to each of the devices.

        param: the name of the data (blobs) to reduce
        input_blobs: the list of blobs to reduce
        destination_blobs: list of blobs to copy the result to
        """
        added_ops = []
        result_blob = "cpu_0/" + param + "_combined"
        added_ops.append(core.CreateOperator("Sum", input_blobs, result_blob))
        for blob in destination_blobs:
            added_ops.append(core.CreateOperator("Copy", result_blob, blob))
        return added_ops

    for op in orig_ops:
        if op.type != 'SpatialBN' and op.type != 'SpatialBNGradient':
            if spatial_bn_phase:
                new_ops.extend(injected_ops)
                new_ops.append(
                    core.CreateOperator("Sum",
                                        sums_blobs,
                                        input_blob_name + "_sums_combined"))
                new_ops.append(
                    core.CreateOperator("Sum",
                                        sumsq_blobs,
                                        input_blob_name + "_sumsq_combined"))
                new_ops.extend(batch_norm_ops)
                injected_ops = []
                batch_norm_ops = []
                sums_blobs = []
                sumsq_blobs = []
                spatial_bn_phase = False
                input_blob_name = None
            elif spatial_bn_gradient_phase:
                new_ops.extend(injected_ops)
                new_ops.extend(_cpuReduce(
                    stripBlobName(scale_grad_blobs[0]),
                    scale_grad_blobs,
                    scale_grad_blobs))
                new_ops.extend(_cpuReduce(
                    stripBlobName(bias_grad_blobs[0]),
                    bias_grad_blobs,
                    bias_grad_blobs))
                new_ops.extend(batch_norm_ops)
                injected_ops = []
                batch_norm_ops = []
                scale_grad_blobs = []
                bias_grad_blobs = []
                spatial_bn_gradient_phase = False
            new_ops.append(op)
        elif op.type == 'SpatialBN':
            spatial_bn_phase = True
            if input_blob_name is None:
                input_blob_name = op.input[0]
            name = op.input[0]
            injected_ops.append(
                core.CreateOperator(
                    "ChannelStats",
                    name,
                    [name + "_sums", name + "_sumsq"]))
            sums_blobs.append(name + "_sums")
            sumsq_blobs.append(name + "_sumsq")
            op.input.append(input_blob_name + "_sums_combined")
            op.input.append(input_blob_name + "_sumsq_combined")
            op.arg.extend([utils.MakeArgument("num_batches", num_devices)])
            batch_norm_ops.append(op)
        elif op.type == 'SpatialBNGradient':
            spatial_bn_gradient_phase = True
            injected_ops.append(
                core.CreateOperator("ChannelBackpropStats",
                                    [op.input[0], op.input[3], op.input[4],
                                     op.input[2]],
                                    [op.output[1], op.output[2]]))
            scale_grad_blobs.append(op.output[1])
            bias_grad_blobs.append(op.output[2])
            op.arg.extend([utils.MakeArgument("num_batches", num_devices)])
            op.input.extend([op.output[1], op.output[2]])
            batch_norm_ops.append(op)

    assert not spatial_bn_phase, \
        "Net modification for cpu inter-device batch normalization failed"
    del model.net.Proto().op[:]
    model.net.Proto().op.extend(new_ops)


def _GPUInterDeviceBatchNormalization(model):
    orig_ops = list(model.net.Proto().op)
    new_ops = []
    num_devices = len(model._devices)
    batch_norm_ops = []
    injected_ops = []

    spatial_bn_phase = False
    sums_blobs = []
    sumsq_blobs = []
    name = []
    input_blob_name = None

    spatial_bn_gradient_phase = False
    scale_grad_blobs = []
    bias_grad_blobs = []
    master_device = "cpu_0"
    master_device_option = core.DeviceOption(caffe2_pb2.CPU)

    def _gpuReduce(param, num_devices, master_device, result_blobs=None):
        """
        Reduces results from multiple gpus and distributes the results back
        to each device. This is done by copying values to the master device
        and summing them. The master device result is then copied back to
        each of the devices.

        param: the name of the data (blobs) to reduce
        num_devices: the number of devices
        master_device: the device to copy/compute values on
        result_blobs: optional list of result blobs to copy to
        """
        added_ops = []
        source_blobs = []
        destination_blobs = []
        if result_blobs is None:
            result_blobs = [
                "gpu_{}/{}_combined".format(i, param) for i in range(num_devices)
            ]
        for i in range(num_devices):
            device_option = core.DeviceOption(model._device_type, i)
            source_blobs.append("gpu_{}/{}".format(i, param))
            destination_blobs.append(
                "{}/{}_gpu_{}_copy".format(master_device, param, i))
            added_ops.append(
                core.CreateOperator(
                    "CopyGPUToCPU",
                    source_blobs[i],
                    destination_blobs[i],
                    device_option=device_option))
        added_ops.append(
            core.CreateOperator(
                "Sum",
                destination_blobs,
                "{}/{}_combined".format(master_device, param),
                device_option=master_device_option))
        for i in range(num_devices):
            device_option = core.DeviceOption(model._device_type, i)
            added_ops.append(
                core.CreateOperator(
                    "CopyCPUToGPU",
                    "{}/{}_combined".format(master_device, param),
                    result_blobs[i],
                    device_option=device_option))
        return added_ops

    for op in orig_ops:
        if op.type != 'SpatialBN' and op.type != 'SpatialBNGradient':
            if spatial_bn_phase:
                new_ops.extend(injected_ops)
                new_ops.extend(_gpuReduce(
                    stripBlobName(input_blob_name) + "_sums",
                    num_devices,
                    master_device,
                ))
                new_ops.extend(_gpuReduce(
                    stripBlobName(input_blob_name) + "_sumsq",
                    num_devices,
                    master_device,
                ))
                new_ops.extend(batch_norm_ops)
                injected_ops = []
                batch_norm_ops = []
                sums_blobs = []
                sumsq_blobs = []
                spatial_bn_phase = False
                input_blob_name = None
            elif spatial_bn_gradient_phase:
                new_ops.extend(injected_ops)
                new_ops.extend(_gpuReduce(
                    stripBlobName(scale_grad_blobs[0]),
                    num_devices,
                    master_device,
                    scale_grad_blobs,
                ))
                new_ops.extend(_gpuReduce(
                    stripBlobName(bias_grad_blobs[0]),
                    num_devices,
                    master_device,
                    bias_grad_blobs,
                ))
                new_ops.extend(batch_norm_ops)
                injected_ops = []
                batch_norm_ops = []
                scale_grad_blobs = []
                bias_grad_blobs = []
                spatial_bn_gradient_phase = False
            new_ops.append(op)
        elif op.type == 'SpatialBN':
            spatial_bn_phase = True
            if input_blob_name is None:
                input_blob_name = op.input[0]
            name = op.input[0]
            device_option = core.DeviceOption(
                model._device_type,
                op.device_option.device_id,
            )
            injected_ops.append(
                core.CreateOperator(
                    "ChannelStats",
                    name,
                    [name + "_sums", name + "_sumsq"],
                    device_option=device_option))
            sums_blobs.append(name + "_sums")
            sumsq_blobs.append(name + "_sumsq")
            op.input.append(name + "_sums_combined")
            op.input.append(name + "_sumsq_combined")
            op.arg.extend([utils.MakeArgument("num_batches", num_devices)])
            batch_norm_ops.append(op)
        elif op.type == 'SpatialBNGradient':
            spatial_bn_gradient_phase = True
            device_option = core.DeviceOption(
                model._device_type,
                op.device_option.device_id,
            )
            injected_ops.append(
                core.CreateOperator("ChannelBackpropStats",
                                    [op.input[0], op.input[3], op.input[4],
                                     op.input[2]],
                                    [op.output[1], op.output[2]],
                                    device_option=device_option))
            scale_grad_blobs.append(op.output[1])
            bias_grad_blobs.append(op.output[2])
            op.arg.extend([utils.MakeArgument("num_batches", num_devices)])
            op.input.extend([op.output[1], op.output[2]])
            batch_norm_ops.append(op)

    assert not spatial_bn_phase, \
        "Net modification for gpu inter-device batch normalization failed"
    del model.net.Proto().op[:]
    model.net.Proto().op.extend(new_ops)
