## @package data_parallel_model
# Module caffe2.python.data_parallel_model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import copy

from caffe2.python import model_helper, dyndep, scope, workspace, core, memonger
from caffe2.proto import caffe2_pb2

dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/nccl:nccl_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/gloo:gloo_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/gloo:gloo_ops_gpu")

log = logging.getLogger("data_parallel_model")
log.setLevel(logging.INFO)


def Parallelize_GPU(*args, **kwargs):
    kwargs['cpu_device'] = False
    Parallelize(*args, **kwargs)


def Parallelize_CPU(*args, **kwargs):
    kwargs['cpu_device'] = True
    Parallelize(*args, **kwargs)


def Parallelize(
    model_helper_obj,
    input_builder_fun,
    forward_pass_builder_fun,
    param_update_builder_fun=None,
    optimizer_builder_fun=None,
    post_sync_builder_fun=None,
    devices=None,
    rendezvous=None,
    net_type='dag',
    broadcast_computed_params=True,
    optimize_gradient_memory=False,
    use_nccl=False,
    max_concurrent_distributed_ops=4,
    cpu_device=False,
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
                        in gradient computation to reduce memory footprint
      cpu_device        Use CPU instead of GPU
    '''
    if devices is None:
        devices = list(range(0, workspace.NumCudaDevices())),

    if not cpu_device:
        for gpu in devices:
            if gpu >= workspace.NumCudaDevices():
                log.warning("** Only {} GPUs available, GPUs {} requested".format(
                    workspace.NumCudaDevices(), devices))
                break
        model_helper_obj._device_type = caffe2_pb2.CUDA
        model_helper_obj._device_prefix = "gpu"
        device_name = "GPU"
    else:
        model_helper_obj._device_type = caffe2_pb2.CPU
        model_helper_obj._device_prefix = "cpu"
        device_name = "CPU"

    log.info("Parallelizing model for devices: {}".format(devices))
    extra_workers = 8 if rendezvous is not None else 0  # best-guess
    num_workers = len(devices) * 4 + extra_workers
    max_concurrent_distributed_ops =\
        min(max_concurrent_distributed_ops, num_workers - 1)
    model_helper_obj.net.Proto().num_workers = num_workers
    model_helper_obj.net.Proto().type = net_type

    # Store some information in the model -- a bit ugly
    model_helper_obj._devices = devices
    model_helper_obj._rendezvous = rendezvous
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
        model_helper_obj._device_grouped_blobs.keys()
    model_helper_obj._computed_param_names = computed_params_grouped.keys()

    if not has_parameter_updates:
        log.info("Parameter update function not defined --> only forward")
        _InferBlobDevice(model_helper_obj)
        return

    log.info("Adding gradient operators")
    _AddGradientOperators(devices, model_helper_obj, losses_by_gpu)

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
    model_helper_obj._grad_names = gradients_grouped.keys()
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
        optimizer_builder_fun(model_helper_obj)

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
    if (rendezvous is not None and num_shards > 1):
        _AddDistributedParameterSync(
            devices,
            model_helper_obj,
            model_helper_obj.param_init_net,
            model_helper_obj.param_init_net,
            rendezvous,
            sync_names,
        )

    _SyncParams(
        devices, model_helper_obj, model_helper_obj.param_init_net, sync_names
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

    if optimize_gradient_memory:
        _OptimizeGradientMemorySimple(model_helper_obj, losses_by_gpu, devices)

    model_helper_obj._data_parallel_model_init_nets = [
        model_helper_obj.param_init_net,
    ]
    model_helper_obj._data_parallel_model_nets = [model_helper_obj.net]


def Parallelize_GPU_BMUF(
    model_helper_obj,
    input_builder_fun,
    forward_pass_builder_fun,
    param_update_builder_fun,
    block_learning_rate=1.0,
    block_momentum=None,
    devices=None,
    rendezvous=None,
    net_type='dag',
    master_gpu=None,
    use_nccl=False,
    optimize_gradient_memory=False,
    reset_momentum_sgd=False,
    warmup_iterations=None,
    max_concurrent_distributed_ops=4,
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
    assert isinstance(model_helper_obj, model_helper.ModelHelper)

    if devices is None:
        devices = list(range(0, workspace.NumCudaDevices()))
    if master_gpu is None:
        master_gpu = devices[0]

    model_helper_obj._devices = devices
    model_helper_obj._rendezvous = rendezvous
    model_helper_obj._device_type = caffe2_pb2.CUDA
    model_helper_obj._device_prefix = 'gpu'
    master_gpu_opt = core.DeviceOption(caffe2_pb2.CUDA, master_gpu)

    num_shards = rendezvous['num_shards'] if rendezvous else 1
    num_workers = len(devices) * num_shards
    num_worker_threads = 4 * len(devices)
    if rendezvous:
        num_worker_threads += 8
    loss_scale = 1.0 / num_workers
    if block_momentum is None:
        block_momentum = 1.0 - 1.0 / num_workers

    max_concurrent_distributed_ops = min(
        max_concurrent_distributed_ops,
        num_worker_threads - 1
    )

    model_helper_obj.net.Proto().num_workers = num_worker_threads
    model_helper_obj.net.Proto().type = net_type

    # A net for initializing global model parameters. Its called once in the
    # same step as net parameters initialization.
    model_helper_obj._global_model_init_net = core.Net('global_model_init')
    model_helper_obj._global_model_init_net.Proto().type = net_type
    model_helper_obj._global_model_init_net.Proto().num_workers = \
        num_worker_threads

    # A net for computing final parameter updates. Its will run once after
    # running net (local models updates) for `num_local_iterations` times.
    model_helper_obj._global_model_param_updates_net = core.Net('global_model')
    model_helper_obj._global_model_param_updates_net.Proto().type = net_type
    model_helper_obj._global_model_param_updates_net.Proto().num_workers = \
        num_worker_threads

    def _v(param):
        return "{}_v".format(param)

    def _g(param):
        return "{}_g".format(param)

    # Keep track of params that were in the model before: they are not
    # data parallel, so we need to handle them separately
    non_datapar_params = copy.copy(model_helper_obj.params)
    model_helper_obj._losses_by_gpu = {}

    def _InitializeModels(gpu_id):
        input_builder_fun(model_helper_obj)
        loss = forward_pass_builder_fun(model_helper_obj, loss_scale)
        model_helper_obj._losses_by_gpu[gpu_id] = loss
    _ForEachGPU(devices, _InitializeModels, scoped=True)

    model_helper_obj._device_grouped_blobs =\
        _GroupByDevice(model_helper_obj, devices,
                       model_helper_obj.params, non_datapar_params)

    model_helper_obj._param_names =\
        model_helper_obj._device_grouped_blobs.keys()

    _AddGradientOperators(
        devices, model_helper_obj, model_helper_obj._losses_by_gpu
    )

    _InferBlobDevice(model_helper_obj)

    def _InitializeParamUpdate(gpu_id):
        param_update_builder_fun(model_helper_obj)
    _ForEachGPU(devices, _InitializeParamUpdate, scoped=True)

    model_parameter_names = model_helper_obj._device_grouped_blobs.keys()
    if warmup_iterations is not None:
        model_helper_obj._warmup_iterations = warmup_iterations
        # A net for broadcasting gpu-0 (master shard) parameters after
        # running net for `warmup_iterartions`.
        model_helper_obj._warmup_broadcast = core.Net('warmup-broadcast')
        model_helper_obj._warmup_broadcast.Proto().type = net_type
        model_helper_obj._warmup_broadcast.Proto().num_workers = \
            num_worker_threads
        if rendezvous is not None and rendezvous['num_shards'] > 1:
            _AddDistributedParameterSync(
                devices,
                model_helper_obj,
                model_helper_obj.param_init_net,
                model_helper_obj._warmup_broadcast,
                rendezvous,
                model_parameter_names
            )

        _SyncParams(
            devices,
            model_helper_obj,
            model_helper_obj._warmup_broadcast,
            model_parameter_names
        )

    # (Step-0) Initialize momentum parameters on master GPU.
    for param_name in model_helper_obj._device_grouped_blobs.keys():
        param = model_helper_obj._device_grouped_blobs[param_name][master_gpu]
        with core.DeviceScope(master_gpu_opt):
            model_helper_obj._global_model_init_net.ConstantFill(
                param, _v(param), value=0.0
            )
            model_helper_obj._global_model_init_net.Copy(param, _g(param))

    # (Step-1) Update models for num_local_iterations.

    # (Step-2) Comute post-local-updates average of the params.
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
    # param = param + param_v
    for param_name in model_parameter_names:
        param = model_helper_obj._device_grouped_blobs[param_name][master_gpu]
        with core.DeviceScope(master_gpu_opt):
            # TODO(ataei) : Stop building the graph here to get model average ?
            model_helper_obj._global_model_param_updates_net.Scale(
                param, param, scale=1.0 / num_workers
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
            model_helper_obj._global_model_param_updates_net.Copy(
                _g(param), param
            )

    if rendezvous is not None and rendezvous['num_shards'] > 1:
        _AddDistributedParameterSync(
            devices,
            model_helper_obj,
            model_helper_obj.param_init_net,
            model_helper_obj._global_model_param_updates_net,
            rendezvous,
            model_parameter_names
        )

    _SyncParams(
        devices,
        model_helper_obj,
        model_helper_obj._global_model_param_updates_net,
        model_parameter_names
    )


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


def RunInitNet(model):
    for init_net in model._data_parallel_model_init_nets:
        workspace.RunNetOnce(init_net)
    for net_iters in model._data_parallel_model_nets:
        if isinstance(net_iters, tuple):
            workspace.CreateNet(net_iters[0])
        else:
            workspace.CreateNet(net_iters)


def RunWarmup(model):
    workspace.RunNet(model.net, model._warmup_iterations)
    workspace.RunNetOnce(model._warmup_broadcast)


def RunNet(model, num_iterations):
    for net_iter in model._data_parallel_model_nets:
        if isinstance(net_iter, tuple):
            workspace.RunNet(net_iter[0].Proto().name, net_iter[1])
        else:
            workspace.RunNet(net_iter, num_iterations)


def _ForEachGPU(gpu_ids, f, scoped=False, *args, **kwargs):
    for gpu_id in gpu_ids:
        device_opt = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
        with core.DeviceScope(device_opt):
            if scoped:
                with core.NameScope("gpu_{}".format(gpu_id)):
                    f(gpu_id, *args, **kwargs)
            else:
                f(gpu_id, *args, **kwargs)


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


def FinalizeAfterCheckpoint(model, blobs=None):
    '''
    This function should be called after loading parameters from a
    checkpoint / initial parameters file.
    '''

    if not hasattr(model, "_checkpoint_net"):
        if blobs is None:
            (_, uniq_blob_names) = _ComputeBlobsToSync(model)
        else:
            uniq_blob_names = [stripParamName(p) for p in blobs]

        # Synchronize to the blob lookup map, as the provided
        # blobs might have non-parameters, such as momemtum blobs.
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
        model._checkpoint_net.RunAllOnGPU()

        if (model._rendezvous is not None and model._rendezvous['num_shards'] > 1):
            checkpoint_init_net = core.Net("checkpoint_init_net")
            checkpoint_init_net.RunAllOnGPU()
            _AddDistributedParameterSync(
                devices,
                model,
                checkpoint_init_net,
                model._checkpoint_net,
                model._rendezvous,
                uniq_blob_names,
            )
            workspace.RunNetOnce(checkpoint_init_net)

        # Setup sync of initial params
        _SyncParams(devices, model, model._checkpoint_net, uniq_blob_names)

        workspace.CreateNet(model._checkpoint_net)

    # Run the sync
    log.info("Run checkpoint net")
    workspace.RunNet(model._checkpoint_net.Proto().name)


def _Broadcast(devices, model, net, param, use_nccl=False):
    # Copy params from gpu_0 to other
    master_dev = devices[0]

    if use_nccl:
        if _IsGPUBlob(model, param):
            master_device_opt = core.DeviceOption(model._device_type, master_dev)
            with core.DeviceScope(master_device_opt):
                model.NCCLBroadcast(
                    model._device_grouped_blobs[param].values(),
                    model._device_grouped_blobs[param].values(),
                    root=master_dev
                )
                return

    for dev_idx in devices[1:]:
        if _IsGPUBlob(model, param):
            device_opt = core.DeviceOption(caffe2_pb2.CUDA, dev_idx)
        else:
            device_opt = core.DeviceOption(caffe2_pb2.CPU, 0)
        with core.DeviceScope(device_opt):
            net.Copy(
                model._device_grouped_blobs[param][master_dev],
                model._device_grouped_blobs[param][dev_idx]
            )


def _AllReduce(devices, model, net, param, use_nccl=False, control_input=None):
    blobs_group = model._device_grouped_blobs[param].values()
    if model._device_type == caffe2_pb2.CUDA and use_nccl:
        model.NCCLAllreduce(
            blobs_group, blobs_group, control_input=control_input
        )
        return

    if model._device_type == caffe2_pb2.CUDA:
        p2p_access_pattern = workspace.GetCudaPeerAccessPattern()
    else:
        p2p_access_pattern = None

    def sumN(*dev_indices):
        """Create a Sum op for 2 or more blobs on different devices.
        Saves the result on the first device.

        Arguments:
        dev_indices -- a list of device indices, which can be translated into
                       CUDA identifiers with model._devices
        """
        devices = [model._devices[idx] for idx in dev_indices]
        blobs = [blobs_group[idx] for idx in dev_indices]
        for i, peer in enumerate(devices):
            if i == 0:
                continue  # Skip the first device
            if p2p_access_pattern is not None and not p2p_access_pattern[
                devices[0], peer
            ]:
                # Copy from peer to d0
                blobs[i] = model.Copy(
                    blobs[i],
                    'gpu_{}/{}_gpu{}_copy'.format(devices[0], param, peer)
                )
        device_opt = core.DeviceOption(model._device_type, devices[0])
        with core.DeviceScope(device_opt):
            net.Sum(blobs, [blobs[0]], name='dpm')

    if len(devices) == 8:
        # Special tree reduction for 8 gpus, TODO generalize like in muji.py
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
    _Broadcast(devices, model, net, param)


def _SyncParams(devices, model, net, unique_param_names):
    for param in unique_param_names:
        _Broadcast(devices, model, net, param)


def _AddDistributedParameterSync(
    devices,
    model,
    init_net,
    net,
    rendezvous,
    uniq_param_names,
):
    assert rendezvous['num_shards'] > 1

    gpu_device_opt = core.DeviceOption(model._device_type, devices[0])
    cpu_device_opt = core.DeviceOption(caffe2_pb2.CPU)

    # Create a single common world for all broadcast operations.
    # This is not a problem since they are executed sequentially.
    comm_world = None
    for param_name in sorted(uniq_param_names):
        param = model._device_grouped_blobs[param_name][devices[0]]

        def broadcast(comm_world, param):
            if comm_world is None:
                comm_world = init_net.CreateCommonWorld(
                    rendezvous['kv_handler'],
                    "broadcast_cw",
                    name=net.Proto().name + ".broadcast_cw_op",
                    size=rendezvous['num_shards'],
                    rank=rendezvous['shard_id'],
                    engine=rendezvous['engine'],
                    status_blob="createcw_broadcast_status",
                )
            net.Broadcast(
                inputs=[comm_world, param],
                outputs=[param],
                engine=rendezvous['engine'],
                status_blob="broadcast_{}_status".format(str(param)),
            )
            return comm_world

        device_opt = gpu_device_opt if _IsGPUBlob(
            model, param_name
        ) else cpu_device_opt

        if rendezvous['engine'] == 'GLOO':
            with core.DeviceScope(device_opt):
                comm_world = broadcast(comm_world, param)
        else:
            # Copy between GPU and CPU
            with core.DeviceScope(device_opt):
                param_cpu = net.CopyGPUToCPU(param, str(param) + "cpu")
            with core.DeviceScope(cpu_device_opt):
                comm_world = broadcast(comm_world, param_cpu)
            with core.DeviceScope(device_opt):
                net.CopyCPUToGPU(param_cpu, param)


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

    master_device_opt = core.DeviceOption(caffe2_pb2.CUDA, devices[0])

    reducing_device_opt = master_device_opt

    # We need to specify a partial order using control_input to ensure
    # progress (all machines need to do same allreduce in parallel)
    num_controls = max_concurrent_distributed_ops
    cyclical_controls = []

    # Since num_controls determines the partial ordering of
    # allreduces, there is no need for more common world instances
    # than there are parallel allreduce operations.
    num_comm_worlds = num_controls
    cyclical_comm_worlds = []

    counter = 0
    nccl_control_blob = None

    # Note: sorted order to ensure each host puts the operators in
    # same order.
    for blob_name in blob_names:
        master_blob = model._device_grouped_blobs[blob_name][devices[0]]
        blobs_group = model._device_grouped_blobs[blob_name].values()

        assert master_blob in blobs_group

        # Remark: NCCLReduce does not support in-place modifications
        # so we need a temporary blob
        reduced_blob = str(master_blob) + "_red"

        control_input = None if len(cyclical_controls) < num_controls \
                        else cyclical_controls[counter % num_controls]
        comm_world = None if len(cyclical_comm_worlds) < num_comm_worlds \
                     else cyclical_comm_worlds[counter % num_comm_worlds]

        def allreduce(comm_world, blobs):
            with core.DeviceScope(reducing_device_opt):
                if comm_world is None:
                    comm_number = len(cyclical_comm_worlds)
                    comm_world = model.param_init_net.CreateCommonWorld(
                        rendezvous['kv_handler'],
                        "allreduce_{}_cw".format(comm_number),
                        name="allreduce_{}_cw_op".format(comm_number),
                        size=rendezvous['num_shards'],
                        rank=rendezvous['shard_id'],
                        engine=rendezvous['engine'],
                        status_blob="create_cw_{}_status".format(comm_number),
                    )
                net.Allreduce(
                    inputs=[comm_world] + blobs,
                    outputs=blobs,
                    name=blob_name,
                    engine=all_reduce_engine,
                    control_input=control_input,
                    status_blob="allreduce_{}_status".format(blob_name),
                )
                return comm_world

        if rendezvous['engine'] == 'GLOO':
            # With Gloo cross GPU and cross machine allreduce
            # can be executed in a single operation
            comm_world = allreduce(comm_world, blobs_group)
            control_output = blobs_group[0]
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
            comm_world = allreduce(comm_world, [reduced_blob])
            control_output = reduced_blob

            with core.DeviceScope(master_device_opt):
                net.Copy(reduced_blob, master_blob)

            # Step 3: broadcast locally
            _Broadcast(devices, model, net, blob_name)

        if len(cyclical_controls) < num_controls:
            cyclical_controls.append(control_output)
        else:
            cyclical_controls[counter % num_controls] = control_output

        if len(cyclical_comm_worlds) < num_comm_worlds:
            cyclical_comm_worlds.append(comm_world)
        else:
            assert cyclical_comm_worlds[counter % num_comm_worlds] == comm_world

        counter += 1


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
        blobs_group = model._device_grouped_blobs[blob_name].values()
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

                        for gpu, g in model._device_grouped_blobs[blob_name].items():
                            device_opt = core.DeviceOption(model._device_type, gpu)
                            with core.DeviceScope(device_opt):
                                model.Copy(grad_idx_concat, g.indices)
                                concatenated_idx.add(g.indices)

                    grad_val_concat, _ = net.Concat(
                        [g.values for g in blobs_group],
                        ["{}/{}_val_concat".format(master_ns, blob_name),
                         "{}/{}_val_splitinfo".format(master_ns, blob_name)],
                        axis=0, name="note:data_parallel_model")

                    for gpu, g in model._device_grouped_blobs[blob_name].items():
                        device_opt = core.DeviceOption(model._device_type, gpu)
                        with core.DeviceScope(device_opt):
                            model.Copy(grad_val_concat, g.values)

        else:
            assert not isinstance(blobs_group[0], core.GradientSlice), \
                "Synchronizing gradient slices not supported"
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                # Poor man's allreduce
                model.net.Sum(blobs_group, [blobs_group[0]])
                _Broadcast(devices, model, model.net, blob_name)


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
    log.warn("Distributed computed params all-reduce not implemented yet")


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
def stripParamName(param):
    # Format is "a/b/c/d" -> "b/c/d"
    if isinstance(param, core.GradientSlice):
        return stripParamName(param.indices) + ":" + stripParamName(param.values)
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
        op_gpu = op_dev.cuda_gpu_id

        # This avoids failing on operators that are only for CPU
        if op_dev.device_type != caffe2_pb2.CUDA:
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
                import google.protobuf.text_format as protobuftx
                step_args = [a for a in op.arg if a.name.endswith("step_net")]
                for step_arg in step_args:
                    step_proto = caffe2_pb2.NetDef()
                    protobuftx.Merge(step_arg.s.decode("ascii"), step_proto)
                    map_ops(step_proto)
    map_ops(model.net.Proto())
    model._blob_to_device = mapping


def _IsGPUBlob(model, blob_name):
    if blob_name in model._blob_to_device:
        return model._blob_to_device[blob_name].device_type == caffe2_pb2.CUDA
    else:
        blob_name = "{}_{}/{}".format(
            model._device_prefix, model._devices[0], blob_name
        )
        if blob_name not in model._blob_to_device:
            return model._device_type == caffe2_pb2.CUDA
        return model._blob_to_device[blob_name].device_type == caffe2_pb2.CUDA


def _GroupByDevice(model, devices, params, non_data_params):
    '''
    Groups blobs by device, returning a map of [blobname] = {0: BlobRef, 1: ..}.
    Returns ordered dictionary, ensuring the original order.
    '''
    grouped = OrderedDict()
    # Only consider params that were created to be  "data parallel"
    params = params[len(non_data_params):]
    assert len(params) % len(devices) == 0,\
           "There should be equal number of params per device"

    num_params_per_device = int(len(params) / len(devices))

    for i, p in enumerate(params):
        assert isinstance(p, core.BlobReference) or \
            isinstance(p, core.GradientSlice), \
            "Param {} is not BlobReference or GradientSlice".format(p)

        name = stripParamName(p)
        gpuid = devices[i // num_params_per_device]

        if isinstance(p, core.BlobReference):
            assert "{}_{}/".format(model._device_prefix, gpuid) in p.GetNameScope(),\
                "Param {} expected to have namescope '{}_{}'".format(str(p), model._device_prefix, gpuid)
        else:
            assert "{}_{}/".format(model._device_prefix, gpuid) in p.indices.GetNameScope(),\
                "Indices {} expected to have namescope '{}_{}'".format(str(p), model._device_prefix, gpuid)
            assert "{}_{}/".format(model._device_prefix, gpuid) in p.values.GetNameScope(),\
                "Values {} expected to have namescope '{}_{}'".format(str(p), model._device_prefix, gpuid)

        if name not in grouped:
            grouped[name] = {}
        grouped[name][gpuid] = p

    # Confirm consistency
    for j, (p, ps) in enumerate(grouped.items()):
        assert \
            len(ps) == len(devices), \
            "Param {} does not have value for each device (only {}: {})".format(
                p, len(ps), ps,
            )
        # Ensure ordering
        if (ps[devices[0]] != params[j]):
            log.error("Params: {}".format(params))
            log.error("Grouped: {}".format(grouped.keys()))
            assert ps[devices[0]] == params[j], \
                "Incorrect ordering: {}".format(ps)

    return grouped


def _ValidateParams(params):
    set_params = set(params)
    if len(params) > len(set_params):
        dupes = []
        sp = sorted(params)
        for j, p in enumerate(sp):
            if j > 0 and params[j - 1] == p:
                dupes.append(p)

        assert len(params) == len(set_params), \
            "Duplicate entries in params: {}".format(dupes)


def _ComputeBlobsToSync(model):
    '''
    We sync all blobs that are generated by param init net and
    are 'data parallel', i.e assigned to a gpu
    '''
    sync_names = set()
    blobs_to_sync = []
    for op in model.param_init_net.Proto().op:
        dp_outputs = [
            o for o in op.output
            if o.startswith("{}_".format(model._device_prefix))
        ]
        sync_names.update([stripParamName(o) for o in dp_outputs])
        blobs_to_sync.extend(dp_outputs)

    # Sanity check
    diff = set(model._param_names) - sync_names
    assert diff == set(), \
       "Some params not instantiated in param init net: {}".format(diff)

    # Remove duplicates and sort
    blobs_to_sync = sorted(list(set(blobs_to_sync)))

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
            set(model.param_to_grad.values()),
            namescope,
            share_activations=False,
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
    input_shapes_all_devices = {}
    for b, shp in input_shapes.items():
        for d in model._devices:
            input_shapes_all_devices["{}_{}/{}".
                                     format(model._device_prefix, d, b)] = shp

    (shapes, types) = workspace.InferShapesAndTypes(
        [model.param_init_net, model.net],
        input_shapes_all_devices,
    )

    for device in model._devices:
        namescope = "{}_{}/".format(model._device_prefix, device)
        excluded_blobs_by_device = set([namescope + b for b in excluded_blobs])
        model.net._net = memonger.share_grad_blobs(
            model.net,
            model._losses_by_gpu[device],
            set(model.param_to_grad.values()),
            namescope,
            dont_share_blobs=excluded_blobs_by_device,
            share_activations=recycle_activations,
            blob_shapes=shapes,
        )
