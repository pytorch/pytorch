from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from types import FunctionType
from functools import wraps
import six

from caffe2.python import cnn, dyndep, scope, workspace, core
from caffe2.proto import caffe2_pb2

dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/nccl:nccl_ops")


DATAPARALLEL_OPS = [
    "Conv",
    "ConvTranspose",
    "GroupConv",
    "FC",
    "FC_Decomp",
    "FC_Prune",
    "FC_Sparse",
    "LRN",
    "Dropout",
    "MaxPool",
    "AveragePool",
    "Concat",
    "DepthConcat",
    "Relu",
    "Transpose",
    "SpatialBN",
    "Accuracy",
    "Adam",
    "AveragedLoss",
    "Cast",
    "LabelCrossEntropy",
    "LearningRate",
    "Print",
    "Scale",
    "Snapshot",
    "Softmax",
    "StopGradient",
    "Summarize",
    "Sum",
    "Tanh",
    "WeightedSum",
    "SquaredL2Distance",
]


class _GPUDataParallelMetaClass(type):
    """A meta class to patch method in order to distribute them over multiple
    GPUs.
    """
    _devices = []

    @staticmethod
    def _data_parallel_wrapper(op):
        @wraps(op)
        def wrapped(cls, blob_in, blob_out, *args, **kwargs):
            # Helpers to extract a device specific blob or a global blob
            def self_or_item(d, key):
                if isinstance(d, dict):
                    assert key in d
                    return d[key]
                return d

            def get_input(gpu_id):
                if isinstance(blob_in, list):
                    return [self_or_item(blob, gpu_id) for blob in blob_in]
                return self_or_item(blob_in, gpu_id)

            def get_output(gpu_id):
                return self_or_item(blob_out, gpu_id)

            # If we have explicit device scope, we do not parallelize
            if cls.explicit_scope():
                return op(
                    cls,
                    blob_in,
                    blob_out,
                    *args,
                    **kwargs)

            devices = _GPUDataParallelMetaClass._devices
            results = {}
            for gpu_id in devices:
                with core.NameScope("gpu_{}".format(gpu_id)):
                    device = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                    with core.DeviceScope(device):
                        result = op(
                            cls,
                            get_input(gpu_id),
                            get_output(gpu_id),
                            *args,
                            **kwargs)
                        results[gpu_id] = result
            return results

        return wrapped

    def __new__(meta, classname, bases, class_dict):
        assert len(bases) == 1, "Expects only one base class"
        base = bases[0]
        assert base is cnn.CNNModelHelper, "Base class should be CNNModelHelper"
        new_class_dict = {}
        for name, attr in base.__dict__.items():
            if name not in DATAPARALLEL_OPS:
                continue
            attr = _GPUDataParallelMetaClass._data_parallel_wrapper(attr)
            new_class_dict[name] = attr
        for name, attr in class_dict.items():
            if name in new_class_dict:
                continue
            if isinstance(attr, FunctionType):
                if name in DATAPARALLEL_OPS:
                    new_class_dict[name] = \
                        _GPUDataParallelMetaClass._data_parallel_wrapper(attr)
                else:
                    new_class_dict[name] = attr
        return super(_GPUDataParallelMetaClass, meta).__new__(
            meta, classname, bases, new_class_dict)


@six.add_metaclass(_GPUDataParallelMetaClass)
class GPUDataParallelModel(cnn.CNNModelHelper):
    """A helper class that extends CNNModelHelper to support multi GPUs
    data parallel training.
    """
    def __init__(self, devices, *args, **kwargs):
        assert len(devices) >= 1, "Should have at least 1 GPU devices"
        assert len(devices) <= workspace.NumCudaDevices(), \
            "Requested # of devices {} is greater than the # of GPUs {}".\
            format(devices, workspace.NumCudaDevices())
        _GPUDataParallelMetaClass._devices = devices
        self._devices = devices
        self._explicit_scope = False
        self._gradient_reduce_all_added = False
        self._mpi_comm = None
        super(GPUDataParallelModel, self).__init__(*args, **kwargs)

    def explicit_scope(self):
        return self._explicit_scope

    def _call(self, name, *args, **kwargs):
        return super(GPUDataParallelModel, self).__getattr__(
            name)(*args, **kwargs)

    # TODO(denisy): try out decorators to avoid this code below
    def Accuracy(self, *args, **kwargs):
        return self._call("Accuracy", *args, **kwargs)

    def Adam(self, *args, **kwargs):
        return self._call("Adam", *args, **kwargs)

    def AveragedLoss(self, *args, **kwargs):
        return self._call("AveragedLoss", *args, **kwargs)

    def Cast(self, *args, **kwargs):
        return self._call("Cast", *args, **kwargs)

    def LabelCrossEntropy(self, *args, **kwargs):
        return self._call("LabelCrossEntropy", *args, **kwargs)

    def LearningRate(self, *args, **kwargs):
        return self._call("LearningRate", *args, **kwargs)

    def Print(self, *args, **kwargs):
        return self._call("Print", *args, **kwargs)

    def Scale(self, *args, **kwargs):
        return self._call("Scale", *args, **kwargs)

    def Snapshot(self, *args, **kwargs):
        return self._call("Snapshot", *args, **kwargs)

    def Softmax(self, *args, **kwargs):
        return self._call("Softmax", *args, **kwargs)

    def StopGradient(self, *args, **kwargs):
        return self._call("StopGradient", *args, **kwargs)

    def Sum(self, *args, **kwargs):
        return self._call("Sum", *args, **kwargs)

    def Summarize(self, *args, **kwargs):
        return self._call("Summarize", *args, **kwargs)

    def Tanh(self, *args, **kwargs):
        return self._call("Tanh", *args, **kwargs)

    def WeightedSum(self, *args, **kwargs):
        return self._call("WeightedSum", *args, **kwargs)

    def SquaredL2Distance(self, *args, **kwargs):
        return self._call("SquaredL2Distance", *args, **kwargs)

    def SetMPIComm(self, mpi_comm):
        self._mpi_comm = mpi_comm

    def FinalizeSetup(self):
        self.param_init_net.RunAllOnGPU()
        self.RunAllOnGPU()

        # If MPI enabled, broadcast params from master
        if (self._mpi_comm is not None):
            self._AddMPIParameterSync()

        # Setup sync of initial params
        self._SyncInitialParams()

    def AddGradientOperators(self, params, *args, **kwargs):
        def create_grad(param):
            return self.ConstantFill(param, str(param) + "_grad", value=1.0)

        param_grad = {}
        # Explicitly need to create gradients on each GPU
        for param in params:
            if not isinstance(param, dict):
                grad = create_grad(param)
                param_grad[str(param)] = str(grad)
            else:
                for gpu_id in self._devices:
                    device = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                    with core.DeviceScope(device):
                        assert gpu_id in param
                        p = param[gpu_id]
                        g = create_grad(p)
                        param_grad[str(p)] = str(g)

        return super(GPUDataParallelModel, self).AddGradientOperators(
            param_grad, *args, **kwargs)

    def AddWeightDecay(self, weight_decay):
        if weight_decay == 0.0:
            return

        assert(weight_decay > 0.0)

        self._explicit_scope = True
        assert \
            self._gradient_reduce_all_added, \
            "Weight decay must be done after gradient sync between gpus"

        for gpu_id in self._devices:
            with core.NameScope("gpu_{}".format(gpu_id)):
                device = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                with core.DeviceScope(device):
                    wd = self.param_init_net.ConstantFill([], 'wd', shape=[1],
                                                          value=weight_decay)
                    ONE = self.param_init_net.ConstantFill([], "ONE", shape=[1],
                                                           value=1.0)
                    # Only update parameters that belong to the current GPU
                    params = self._CurrentScopeParams()

                    # Take only params that are weights
                    print("Adding weigth-decay for gpu {}.".format(gpu_id))

                    gpu_weights = [p for p in params if p in self.weights]
                    for w in gpu_weights:
                        # Equivalent to grad -= w * param
                        grad = self.param_to_grad[w]
                        self.net.WeightedSum([grad, ONE, w, wd], grad)

        self._explicit_scope = False

    def _Broadcast(self, net, param):
        # TODO(akyrola): replace with NCCLBroadcast when it's working
        # Copy params from gpu_0 to other
        for gpu_idx in self._devices[1:]:
            device_opt = core.DeviceOption(caffe2_pb2.CUDA, gpu_idx)
            with core.DeviceScope(device_opt):
                net.Copy(
                    "gpu_{}/{}".format(self._devices[0], param),
                    "gpu_{}/{}".format(gpu_idx, param)
                )

    def _SyncInitialParams(self):
        unique_param_names = set(
            stripParamName(p)
            for p in self.params
        )

        self._explicit_scope = True
        for param in unique_param_names:
            self._Broadcast(self.param_init_net, param)

        self._explicit_scope = False

    def _AddMPIParameterSync(self):
        # Sync from master
        unique_param_names = set(
            stripParamName(p)
            for p in self.params
        )

        self._explicit_scope = True

        # Should this be done in GPU 0 scope?
        for param_name in unique_param_names:
            param = "gpu_{}/{}".format(self._devices[0], param_name)
            self.param_init_net.Broadcast(
                inputs=[self._mpi_comm, param],
                outputs=[param],
                engine='MPI'
            )
        self._explicit_scope = False

    def _AllReduceGradients(self):
        self._gradient_reduce_all_added = True

        if self._mpi_comm is None:
            self._AllReduceGradientsSingleHost()
        else:
            self._AllReduceGradientsWithMPI()

    def _AllReduceGradientsWithMPI(self):
        self._explicit_scope = True
        unique_grads_names = set(
            stripParamName(grad)
            for grad in self.param_to_grad.values()
        )

        # Step 1: sum gradients from local GPUs to master GPU
        last_out = None
        master_device_opt = core.DeviceOption(caffe2_pb2.CUDA, self._devices[0])

        # Note: sorted order to ensure each host puts the operators in
        # same order.
        for grad_name in sorted(unique_grads_names):
            grads_group = [
                grad
                for grad in self.param_to_grad.values()
                if stripParamName(grad) == grad_name
            ]
            master_grad = "gpu_{}/{}".format(self._devices[0], grad_name)
            assert master_grad in grads_group

            # Remark: NCCLReduce does not support in-place modifications
            # so we need a temporary gradient blob
            reduced_grad = "gpu_{}/{}_red".format(
                self._devices[0],
                grad_name
            )

            with core.DeviceScope(master_device_opt):
                self.ConstantFill(master_grad, reduced_grad, value=0.0)
                self.net.NCCLReduce(grads_group, reduced_grad)

                # Step 2: allreduce over MPI to all hosts, between master GPUs
                self.net.Allreduce(
                    inputs=[self._mpi_comm, reduced_grad],
                    outputs=[master_grad],
                    engine='MPI',
                    control_input=None if last_out is None else [last_out],
                )
                last_out = master_grad

            # Step 3: broadcast locally
            self._Broadcast(self.net, grad_name)

        self._explicit_scope = False

    def _AllReduceGradientsSingleHost(self):
        """Performs NCCL AllReduce to distribute gradients to all the GPUs."""

        if len(self._devices) == 1:
            return

        # Take only params that have gradient associated with them.
        unique_grads_names = set(
            stripParamName(grad)
            for grad in self.param_to_grad.values()
        )

        # Now we need to Allreduce gradients on all the GPUs.
        # Pick GPU #0 as a master GPU.
        self._explicit_scope = True
        master_device_opt = core.DeviceOption(caffe2_pb2.CUDA, self._devices[0])
        with core.DeviceScope(master_device_opt):
            # Group by grads for reduce.
            for grad_name in unique_grads_names:
                grads_group = [
                    grad
                    for grad in self.param_to_grad.values()
                    if stripParamName(grad) == grad_name
                ]
                assert len(grads_group) == len(self._devices), \
                    "Each GPU from {}, should have a copy of {}.".format(
                        self._devices, grad_name)
                self.NCCLAllreduce(grads_group, grads_group)
        self._explicit_scope = False

    def _BuildLR(self, base_lr, policy="fixed", **other_lr_params):
        """A helper to create learning rate."""
        ITER = self.Iter("ITER")
        # There is one interesting thing here: since we are minimizing, we are
        # doing "descent" so the learning rate is set to be negative.
        LR = self.net.LearningRate(
            [ITER],
            "LR",
            base_lr=base_lr,
            policy=policy,
            **other_lr_params
        )
        return LR

    def _BuildSGD(self, params, base_lr, policy="fixed", **other_lr_params):
        """A helper to construct gradient update for SGD."""
        base_lr = base_lr / len(self._devices)
        LR = self._BuildLR(base_lr, policy, **other_lr_params)
        ONE = self.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        for param in params:
            grad = self.param_to_grad[param]
            if isinstance(grad, core.GradientSlice):
                self.ScatterWeightedSum(
                    [param, ONE, grad.indices, grad.values, LR], param
                )
            else:
                self.WeightedSum([param, ONE, grad, LR], param)

    def _CurrentScopeParams(self):
        return [
            param
            for param in self.param_to_grad.keys()
            if str(param).startswith(scope.NAMESCOPE)
        ]

    def SGD(self, base_lr, policy="fixed", **other_lr_params):
        """Adds SGD optimizer to the model."""
        self._AllReduceGradients()

        # Create update params operators.
        self._explicit_scope = True
        for gpu_id in self._devices:
            with core.NameScope("gpu_{}".format(gpu_id)):
                device = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                with core.DeviceScope(device):
                    # Only update parameters that belong to the current GPU
                    params = self._CurrentScopeParams()

                    # Add optimizer update operators
                    self._BuildSGD(params, base_lr, policy, **other_lr_params)
        self._explicit_scope = False

    def CustomSGD(
        self,
        paramup_build_fn,
        base_lr,
        lr_policy,
        weight_decay,
        **other_lr_pars
    ):
        """Custom parameter update function"""
        self._AllReduceGradients()

        self.AddWeightDecay(weight_decay)

        # Run parameter update on each machine
        self._explicit_scope = True
        for gpu_id in self._devices:
            with core.NameScope("gpu_{}".format(gpu_id)):
                device = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                with core.DeviceScope(device):
                    LR = self._BuildLR(base_lr, lr_policy, **other_lr_pars)

                    params = self._CurrentScopeParams()
                    paramup_build_fn(self, params, LR)
        self._explicit_scope = False

    def ExecOnEachDevice(self, fn, *args, **kwargs):
        self._explicit_scope = True
        for gpu_id in self._devices:
            with core.NameScope("gpu_{}".format(gpu_id)):
                device = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                with core.DeviceScope(device):
                    fn(self, *args, **kwargs)

        self._explicit_scope = False


# A helper function to extract a parameter's name
def stripParamName(param):
    # Format is "a/b/c/d" -> d
    name = str(param)
    sep = scope._NAMESCOPE_SEPARATOR
    return name[name.rindex(sep) + 1:]


def SetupMPICluster(num_replicas, role, job_path):
    from caffe2.python import mpi
    print("Initing library")
    dyndep.InitOpsLibrary('@/caffe2/caffe2/mpi:mpi_ops')
    print("Setup peers")
    mpi.SetupPeers(
        replicas=int(num_replicas),
        role=role,
        job_path=job_path
    )
    print("Create mpi_init net")
    mpi_init_net = core.Net('mpi_init')
    print("Create commonworld")
    mpi_comm = mpi_init_net.CreateCommonWorld(
        inputs=[],
        outputs=['comm_world'],
        engine='MPI'
    )
    print("Run mpi_init net")
    workspace.RunNetOnce(mpi_init_net)
    print("Finished MPI setup")
    return mpi_comm
