from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, scope, workspace
import numpy as np

import logging


class ParameterType(object):
    DENSE = 'dense'
    SPARSE = 'sparse'


class ParameterInfo(object):
    def __init__(
            self, param_id, param, key=None, shape=None, length=None):
        assert isinstance(param, core.BlobReference)
        self.param_id = param_id
        self.name = str(param)
        self.blob = param
        self.key = key
        self.shape = shape
        self.size = None if shape is None else np.prod(shape)
        self.length = max(1, length if length is not None else 1)
        self.grad = None
        self._cloned_init_net = None

    def grad_type(self):
        # self.grad could be None for model parallelism with parameter server
        if self.grad is None:
            return
        return (
            ParameterType.SPARSE if isinstance(self.grad, core.GradientSlice)
            else ParameterType.DENSE)

    def cloned_init_net(self):
        if not self._cloned_init_net:
            init_net, outputs = self.blob.Net().ClonePartial(
                'param_%d_%s_init' % (self.param_id, self.name),
                inputs=[],
                outputs=[self.blob])
            self._cloned_init_net = (init_net, outputs[0])
        return self._cloned_init_net

    def __str__(self):
        return self.name


class ModelHelperBase(object):
    """A helper model so we can write models more easily, without having to
    manually define parameter initializations and operators separately.
    In order to add support for specific operators, inherit from this class
    and add corresponding methods. Operator representing methods should
    take care of adding their parameters to params
    """

    def __init__(self, name=None, init_params=True, allow_not_known_ops=True,
                 skip_sparse_optim=False, param_model=None):
        self.name = name or "model"
        self.net = core.Net(self.name)

        if param_model is not None:
            self.param_init_net = param_model.param_init_net
            self.param_to_grad = param_model.param_to_grad
            self.params = param_model.params
            self.computed_params = param_model.computed_params
        else:
            self.param_init_net = core.Net(name + '_init')
            self.param_to_grad = {}
            self.params = []
            self.computed_params = []

        self._param_info = []
        self._devices = []
        self.gradient_ops_added = False
        self.init_params = init_params
        self.allow_not_known_ops = allow_not_known_ops
        self.skip_sparse_optim = skip_sparse_optim

    def _infer_param_shape(self, param):
        for op in self.param_init_net.Proto().op:
            if str(param) in op.output:
                for arg in op.arg:
                    if arg.name == "shape":
                        return list(arg.ints)
        return None

    def _update_param_info(self):
        assert len(self._param_info) <= len(self.params)
        for param in self.params[len(self._param_info):]:
            if not isinstance(param, core.BlobReference):
                param = core.BlobReference(str(param), net=self._param_init_net)
            self._param_info.append(ParameterInfo(
                param_id=len(self._param_info),
                param=param,
                shape=self._infer_param_shape(param)))
        for info in self._param_info:
            info.grad = self.param_to_grad.get(info.name)

    def add_param(self, param, key=None, shape=None, length=None):
        self._update_param_info()
        if key is not None and self.net.input_record() is not None:
            idx = self.net.input_record().field_blobs().index(key)
            key = self.net.input_record().field_names()[idx]
        shape = shape if shape is not None else self._infer_param_shape(param)
        self.params.append(param)
        if not isinstance(param, core.BlobReference):
            param = core.BlobReference(str(param), net=self._param_init_net)
        self._param_info.append(ParameterInfo(
            param_id=len(self._param_info),
            param=param,
            shape=shape,
            key=key,
            length=length,
        ))
        return self._param_info[-1]

    def param_info(self, grad_type=None, id=None):
        self._update_param_info()
        if id is not None:
            assert grad_type is None
            info = self._param_info[id]
            assert info.param_id == id
            return info
        elif grad_type is not None:
            return [
                info for info in self._param_info
                if info.grad_type() == grad_type]
        else:
            return self._param_info

    def GetParams(self, namescope=None, top_scope=False):
        '''
        Returns the params in current namescope
        '''
        if namescope is None:
            namescope = scope.CurrentNameScope()
        else:
            if not namescope.endswith(scope._NAMESCOPE_SEPARATOR):
                namescope += scope._NAMESCOPE_SEPARATOR

        if namescope == '':
            return self.params[:]
        elif top_scope:
            return [
                p for p in self.params
                if p.GetNameScope().startswith(namescope)
            ]
        else:
            return [p for p in self.params if
                    p.GetNameScope().startswith(namescope)]

    def Proto(self):
        return self.net.Proto()

    def InitProto(self):
        return self.param_init_net.Proto()

    def RunAllOnGPU(self, *args, **kwargs):
        self.param_init_net.RunAllOnGPU(*args, **kwargs)
        self.net.RunAllOnGPU(*args, **kwargs)

    def CreateDB(self, blob_out, db, db_type, **kwargs):
        dbreader = self.param_init_net.CreateDB(
            [], blob_out, db=db, db_type=db_type, **kwargs)
        return dbreader

    def AddGradientOperators(self, *args, **kwargs):
        if self.gradient_ops_added:
            raise RuntimeError("You cannot run AddGradientOperators twice.")
        self.gradient_ops_added = True
        self.grad_map = self.net.AddGradientOperators(*args, **kwargs)
        self.param_to_grad = self.get_param_to_grad(self.params)
        return self.grad_map

    def get_param_to_grad(self, params):
        '''
        Given a list of parameters returns a dict from a parameter
        to a corresponding gradient
        '''

        param_to_grad = {}
        if not self.gradient_ops_added:
            raise RuntimeError("You need to run AddGradientOperators first.")
        # We need to use empty namescope when creating the gradients
        # to prevent duplicating the namescope prefix for gradient blobs.
        for p in params:
            if str(p) in self.grad_map:
                param_to_grad[p] = self.grad_map[str(p)]
        return param_to_grad

    def GetOptimizationPairs(self, params=None):
        '''
        Returns a map for param => grad.
        If params is not specified, all parameters will be considered.
        '''
        if not self.gradient_ops_added:
            raise RuntimeError("Need to call AddGradientOperators first")

        param_to_grad = self.param_to_grad
        if params:
            param_to_grad = self.get_param_to_grad(params)

        if not self.skip_sparse_optim:
            return param_to_grad
        else:
            return {param: grad for param, grad in param_to_grad.items()
                    if not isinstance(grad, core.GradientSlice)}

    def GetComputedParams(self, namescope=None):
        '''
        Returns the computed params in current namescope. 'Computed params'
        are such parameters that are not optimized via gradient descent but are
        directly computed from data, such as the running mean and variance
        of Spatial Batch Normalization.
        '''
        if namescope is None:
            namescope = scope.CurrentNameScope()
        else:
            if not namescope.endswith(scope._NAMESCOPE_SEPARATOR):
                namescope += scope._NAMESCOPE_SEPARATOR

        if namescope == '':
            return self.computed_params[:]
        else:
            return [p for p in self.computed_params
                    if p.GetNameScope() == namescope]

    def GetAllParams(self, namescope=None):
        return self.GetParams(namescope) + self.GetComputedParams(namescope)

    def TensorProtosDBInput(
        self, unused_blob_in, blob_out, batch_size, db, db_type, **kwargs
    ):
        """TensorProtosDBInput."""
        dbreader_name = "dbreader_" + db
        dbreader = self.param_init_net.CreateDB(
            [], dbreader_name,
            db=db, db_type=db_type)
        return self.net.TensorProtosDBInput(
            dbreader, blob_out, batch_size=batch_size)

    def AddOperator(self, op_type, inputs, parameters, *args, **kwargs):
        """
        Adds an operator to a model. Use parameters list
        to specify which operator inputs are model parameters to be
        optimized.

        Example of usage:

        model.SparseLengthsSum(
             [embedding, indices, lengths],
             parameters=[embedding],
        )

        Here embedding is a parameter to be optimized while indices
        and lengths are not.
        """

        extra_parameters = filter(lambda x: (x not in inputs), parameters)
        if len(extra_parameters) > 0:
            raise Exception("Some parameters are not inputs: {}".format(
                map(str, extra_parameters)
            ))

        self.params.extend(parameters)
        return self.net.__getattr__(op_type)(inputs, *args, **kwargs)

    def GetDevices(self):
        assert len(self._devices) > 0, \
            "Use data_parallel_model to run model on multiple GPUs."
        return self._devices

    def __getattr__(self, op_type):
        """Catch-all for all other operators, mostly those without params."""
        if op_type.startswith('__'):
            raise AttributeError(op_type)

        if not core.IsOperator(op_type):
            raise RuntimeError(
                'Method ' + op_type + ' is not a registered operator.' +
                ' Did you mean: [' +
                ','.join(workspace.C.nearby_opnames(op_type)) + ']'
            )
        # known_working_ops are operators that do not need special care.
        known_working_ops = [
            "Accuracy",
            "Adam",
            "Add",
            "Adagrad",
            "SparseAdagrad",
            "AveragedLoss",
            "Cast",
            "Checkpoint",
            "ConstantFill",
            "Copy",
            "CopyGPUToCPU",
            "CopyCPUToGPU",
            "DequeueBlobs",
            "EnsureCPUOutput",
            "Flatten",
            "FlattenToVec",
            "LabelCrossEntropy",
            "LearningRate",
            "MakeTwoClass",
            "MatMul",
            "NCCLAllreduce",
            "NHWC2NCHW",
            "PackSegments",
            "Print",
            "PRelu",
            "Scale",
            "ScatterWeightedSum",
            "Sigmoid",
            "SortedSegmentSum",
            "Snapshot", # Note: snapshot is deprecated, use Checkpoint
            "Softmax",
            "SoftmaxWithLoss",
            "SquaredL2Distance",
            "Squeeze",
            "StopGradient",
            "Summarize",
            "Tanh",
            "UnpackSegments",
            "WeightedSum",
            "ReduceFrontSum",
        ]
        if op_type not in known_working_ops:
            if not self.allow_not_known_ops:
                raise RuntimeError(
                    "Operator {} is not known to be safe".format(op_type))

            logging.warning("You are creating an op that the ModelHelperBase "
                            "does not recognize: {}.".format(op_type))
        return self.net.__getattr__(op_type)


def ExtractPredictorNet(
    net_proto,
    input_blobs,
    output_blobs,
    device=None,
    renames=None
):
    '''
    Takes a model net for training and returns a net which can be
    used for prediction. For example, all gradient operators and
    input operators are removed.
    @param net_proto protobuf of the net you want to process (net.Proto())
    @param input_blobs list/set of blob names that are the inputs of predictor
    @param output_blobs list/set of blob names that are outputs of predictor
    @param device optional device option that is assigned
    @param renames dictionary of blob name to a new name (optional)
    '''
    predict_net = core.Net(net_proto.name + "_predict")
    predict_proto = predict_net.Proto()

    orig_external_inputs = set(net_proto.external_input)
    orig_external_outputs = set(net_proto.external_output)
    input_blobs = {str(b) for b in input_blobs}
    known_blobs = set(orig_external_inputs).union(input_blobs)
    output_blobs = {str(b) for b in output_blobs}
    external_inputs = set(input_blobs)
    external_outputs = set(output_blobs)

    ops = list(net_proto.op)

    # Find the range of ops that we should include
    try:
        first_op_with_input = min(
            [
                j for j in range(len(ops))
                if input_blobs.intersection(ops[j].input) and ops[j].type !=
                'StopGradient'
            ]
        )
    except ValueError:
        raise Exception("No ops with input={}".format(input_blobs))
    try:
        last_op_with_output = max(
            [
                j for j in range(len(ops))
                if output_blobs.intersection(ops[j].output)
            ]
        )
    except ValueError:
        raise Exception("No ops with output={}".format(output_blobs))

    # Iterate through the ops and only include those whose inputs
    # we can satisfy.
    for op in ops[first_op_with_input:(last_op_with_output + 1)]:
        if known_blobs.issuperset(op.input):
            if device is not None:
                op.device_option.device_type = device.device_type
                op.device_option.cuda_gpu_id = device.cuda_gpu_id
            predict_proto.op.extend([op])
            known_blobs.update(op.output)
            external_inputs.update(
                set(op.input).intersection(orig_external_inputs)
            )
            external_outputs.update(
                set(op.output).intersection(orig_external_outputs)
            )
        else:
            logging.warning(
                "Op {} had unknown inputs: {}".format(
                    op.type, set(op.input).difference(known_blobs)
                )
            )

    def rename_list(proto_list):
        if renames is None:
            return

        # proto lists don't support assignments
        new_list = proto_list[:]
        for j, b in enumerate(new_list):
            if b in renames:
                new_list[j] = renames[b]

        del proto_list[:]
        proto_list.extend(new_list)

    # Predictor net's external inputs and outputs include only those
    # that are part of this net.
    predict_proto.external_input.extend(external_inputs)
    predict_proto.external_output.extend(external_outputs)

    rename_list(predict_proto.external_input)
    rename_list(predict_proto.external_output)

    for op in predict_proto.op:
        rename_list(op.input)
        rename_list(op.output)

    return predict_net
