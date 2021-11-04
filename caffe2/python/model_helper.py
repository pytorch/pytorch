## @package model_helper
# Module caffe2.python.model_helper





from caffe2.python import core, scope, workspace
from caffe2.python.helpers.db_input import db_input
from caffe2.python.modeling import parameter_info
from caffe2.python.modeling.parameter_sharing import (
    parameter_sharing_context,
)
from caffe2.python.optimizer_context import (
    OptimizerContext,
    DEFAULT_OPTIM,
)
from caffe2.python.regularizer_context import RegularizerContext

from future.utils import viewitems, viewkeys
from itertools import chain

import logging


# _known_working_ops are operators that do not need special care.
_known_working_ops = [
    "Accuracy",
    "Adam",
    "Add",
    "Adagrad",
    "SparseAdagrad",
    "Adadelta",
    "SparseAdadelta",
    "AveragedLoss",
    "Cast",
    "Checkpoint",
    "ConstantFill",
    "Copy",
    "CopyGPUToCPU",
    "CopyCPUToGPU",
    "DequeueBlobs",
    "EnsureCPUOutput",
    "ExpandDims",
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
    "ReduceFrontSum",
    "Scale",
    "ScatterWeightedSum",
    "Sigmoid",
    "SortedSegmentSum",
    "Snapshot",  # Note: snapshot is deprecated, use Checkpoint
    "Softmax",
    "SoftmaxWithLoss",
    "SquaredL2Distance",
    "Squeeze",
    "StopGradient",
    "Summarize",
    "Tanh",
    "Transpose",
    "UnpackSegments",
    "WeightedSum",
    "YellowFin"
]


class ModelHelper(object):
    """A helper model so we can manange models more easily. It contains net def
    and parameter storages. You can add an Operator yourself, e.g.

        model = model_helper.ModelHelper(name="train_net")
        # init your weight and bias as w and b
        w = model.param_init_net.XavierFill(...)
        b = model.param_init_net.ConstantFill(...)
        fc1 = model.FC([input, w, b], output, **kwargs)

    or you can use helper functions in brew module without manually
    defining parameter initializations and operators.

        model = model_helper.ModelHelper(name="train_net")
        fc1 = brew.fc(model, input, output, dim_in, dim_out, **kwargs)

    """

    def __init__(self, name=None, init_params=True, allow_not_known_ops=True,
                 skip_sparse_optim=False, param_model=None, arg_scope=None):
        self.name = name or "model"
        self.net = core.Net(self.name)

        if param_model is not None:
            self.param_init_net = param_model.param_init_net
            self.param_to_grad = param_model.param_to_grad
            self.params = param_model.params
            self._parameters_info = param_model._parameters_info
            self._computed_params = param_model._computed_params
        else:
            self.param_init_net = core.Net(self.name + '_init')
            self.param_to_grad = {}
            self.params = []
            self._parameters_info = {}
            self._computed_params = []

        self._param_info_deprecated = []
        self._devices = []
        self.gradient_ops_added = False
        self.init_params = init_params
        self.allow_not_known_ops = allow_not_known_ops
        self.skip_sparse_optim = skip_sparse_optim
        self.weights = []
        self.biases = []
        self._arg_scope = {
            'order': "NCHW",
            'use_cudnn': True,
            'cudnn_exhaustive_search': False,
        }
        if arg_scope is not None:
            # Please notice value as None is not acceptable. We are not checking it
            # here because we already have check in MakeArgument.
            self._arg_scope.update(arg_scope)

    @property
    def arg_scope(self):
        return self._arg_scope

    def get_name(self):
        return self.name

    def _infer_param_shape(self, param):
        for op in self.param_init_net.Proto().op:
            if str(param) in op.output:
                for arg in op.arg:
                    if arg.name == "shape":
                        return list(arg.ints)
        return None

    def _update_param_info_deprecated(self):
        assert len(self._param_info_deprecated) <= len(self.params)
        for param in self.params[len(self._param_info_deprecated):]:
            if not isinstance(param, core.BlobReference):
                raise ValueError(
                    "Param %s must be a BlobReference!" % str(param))
            self._param_info_deprecated.append(parameter_info.ParameterInfo(
                param_id=len(self._param_info_deprecated),
                param=param,
                shape=self._infer_param_shape(param)))
        for info in self._param_info_deprecated:
            info.grad = self.param_to_grad.get(info.name)

    def _normalize_tags(self, tags):
        tags = tags or []
        return set(tags) if isinstance(tags, list) else set([tags])

    def create_param(self, param_name, shape, initializer, tags=None):
        """
        Creates parameter with a given name and initializer.

        If param_name is instance of BlobRefernce - then this blob will be used
        to store parameter (no any logic will affect it's location).

        If param_name is instance of a string type, then the final blob will
        be created in the CurrentNameScope with the respect of all parameter
        sharing logic, i.e. 'resolved_name_scope/param_name'.

        Parameter sharing logic is going to override CurrentNameScope according
        to the rules that are specified through ParameterSharing contexts,
        all ParameterSharing contexts are applied recursively until there are no
        extra overrides present, where on each step the best match will be
        applied first.

        The following examples should clarify the way ParameterSharing logic
        works:

        As an example if this function is called with parameter 'w':
        a. Call from some scope 'global_scope' with no Parameter sharing:
          'global_scope/w'
        b. Call from scope 'scope_b', with override {'scope_b': 'scope_a'}:
          'scope_a/w'
        c. Call from scope 'scope_a', with override {'scope_a': ''}:
          'scope_a/w'
        d. Call from scope 'scope_b/shared', with overrides
          {'scope_b/shared': 'scope_b', 'scope_b': 'scope_a'}:
          'scope_a/w'
        d. Call from scope 'scope_b/unshared', with overrides
          {'scope_b/shared': 'scope_b', 'scope_b': 'scope_a'}:
          'scope_a/unshared/w'
        """
        # ParameterSharing works only for case when param_name is instance of
        # a string type. If param_name is a BlobReference - no attempt for
        # ParameterSharing will be applied.
        if isinstance(param_name, core.BlobReference):
            param_name = str(param_name)
        elif isinstance(param_name, str):
            # Parameter name will be equal to current Namescope that got
            # resolved with the respect of parameter sharing of the scopes.
            param_name = parameter_sharing_context.get_parameter_name(
                param_name)
        else:
            raise TypeError("Unsupported type for param_name")

        if param_name in self._parameters_info:
            assert self._parameters_info[param_name].shape == shape
            return self._parameters_info[param_name].blob

        param_info = initializer.create_param(
            param_name=core.BlobReference(param_name),
            init_net=self.param_init_net,
            shape=shape,
        )
        optim_context = OptimizerContext.current()
        for tag in self._normalize_tags(tags):
            if optim_context.has_optimizer(tag):
                # param_info will check optimizer has not been set
                param_info.optimizer = optim_context.get_optimizer(tag)
        if not param_info.optimizer and optim_context.has_optimizer(DEFAULT_OPTIM):
            param_info.optimizer = optim_context.get_optimizer(DEFAULT_OPTIM)

        reg_context = RegularizerContext.current()
        param_info.regularizer = reg_context

        self._parameters_info[param_name] = param_info
        # Add param to legacy structs as well, so all other functions for
        # parameters are still working.
        self.AddParameter(param_info.blob, tags)
        return param_info.blob

    def get_param_info(self, param):
        assert isinstance(param, core.BlobReference), \
            "Param {} is not a BlobReference".format(param)
        return self._parameters_info.get(param, None)

    # This method is deprecated, use create_param method which
    # also does parameter initialization when needed
    def add_param_DEPRECATED(self, param, key=None, shape=None, length=None):
        logging.warning("add_param method is DEPRECATED")
        self._update_param_info_deprecated()
        self.AddParameter(param)
        if key is not None and self.net.input_record() is not None:
            idx = self.net.input_record().field_blobs().index(key)
            key = self.net.input_record().field_names()[idx]
        shape = shape if shape is not None else self._infer_param_shape(param)
        if not isinstance(param, core.BlobReference):
            raise ValueError("Param %s must be a BlobReference!" % str(param))
        self._param_info_deprecated.append(parameter_info.ParameterInfo(
            param_id=len(self._param_info_deprecated),
            param=param,
            shape=shape,
            key=key,
            length=length,
        ))
        return self._param_info_deprecated[-1]

    def AddParameter(self, param, tags=None):
        assert isinstance(param, core.BlobReference)
        tags = self._normalize_tags(tags)
        if parameter_info.ParameterTags.COMPUTED_PARAM in tags:
            self._computed_params.append(param)
        else:
            self.params.append(param)

        if parameter_info.ParameterTags.WEIGHT in tags:
            self.weights.append(param)
        if parameter_info.ParameterTags.BIAS in tags:
            self.biases.append(param)

    @staticmethod
    def _NormalizeNamescope(namescope):
        if namescope is None:
            return scope.CurrentNameScope()
        elif namescope == '' or namescope.endswith(scope._NAMESCOPE_SEPARATOR):
            return namescope
        else:
            return namescope + scope._NAMESCOPE_SEPARATOR

    def GetParams(self, namescope=None, top_scope=False):
        '''
        Returns the params in current namescope
        '''
        namescope = ModelHelper._NormalizeNamescope(namescope)

        if namescope == '':
            return self.params[:]
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
        self.Validate()

        self.gradient_ops_added = True
        self.grad_map = self.net.AddGradientOperators(*args, **kwargs)
        self.param_to_grad = self.get_param_to_grad(self.params)

        # Populate ParameterInfo for all parameters if missing
        # and add gradient blob information. So optimizers can use it
        for param, grad in self.param_to_grad.items():
            param_info = self.get_param_info(param)
            if param_info:
                param_info.grad = grad
            else:
                self._parameters_info[param] = parameter_info.ParameterInfo(
                    param_id=None,
                    param=param,
                    grad=grad,
                )

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

    def GetOptimizationParamInfo(self, params=None):
        '''
        Returns a map for param => grad.
        If params is not specified, all parameters will be considered.
        '''
        if not self.gradient_ops_added:
            raise RuntimeError("Need to call AddGradientOperators first")

        param_to_grad = self.param_to_grad
        if params:
            param_to_grad = self.get_param_to_grad(params)

        return [
            self.get_param_info(param) for param, grad in viewitems(param_to_grad)
            if (
                not self.skip_sparse_optim or
                not isinstance(grad, core.GradientSlice)
            )
        ]

    def _Validate(self):
        '''
        Check for duplicate params
        '''
        params_list = [str(p) for p in self.params]
        params_set = set(params_list)

        dupes = []
        if len(params_set) != len(params_list):
            params_list = sorted(params_list)
            for j, p in enumerate(params_list):
                if j > 0 and params_list[j - 1] == p:
                    if p not in dupes:
                        dupes.append(p)

        return dupes

    def Validate(self):
        dupes = self._Validate()
        assert dupes == [], "Duplicate params: {}".format(dupes)

    def GetComputedParams(self, namescope=None):
        '''
        Returns the computed params in current namescope. 'Computed params'
        are such parameters that are not optimized via gradient descent but are
        directly computed from data, such as the running mean and variance
        of Spatial Batch Normalization.
        '''
        namescope = ModelHelper._NormalizeNamescope(namescope)

        if namescope == '':
            return self._computed_params[:]
        else:
            return [p for p in self._computed_params
                    if p.GetNameScope().startswith(namescope)]

    def GetAllParams(self, namescope=None):
        return self.GetParams(namescope) + self.GetComputedParams(namescope)

    def TensorProtosDBInput(
        self, unused_blob_in, blob_out, batch_size, db, db_type, **kwargs
    ):
        """TensorProtosDBInput."""
        assert len(unused_blob_in) == 0, \
            """You cannot pass reader to model_helper.TensorProtosDBInput.
               Use model.net.TensorProtosDBInput instead to create the op."""

        return db_input(
            self, blob_out, batch_size, db, db_type, **kwargs)

    def GetDevices(self):
        assert len(self._devices) > 0, \
            "Use data_parallel_model to run model on multiple GPUs."
        return self._devices

    def __getattr__(self, op_type):
        """Catch-all for all other operators, mostly those without params."""
        if op_type.startswith('__'):
            raise AttributeError(op_type)

        if not core.IsOperator(op_type):
            raise AttributeError(
                'Method ' + op_type + ' is not a registered operator.' +
                ' Did you mean: [' +
                ','.join(workspace.C.nearby_opnames(op_type)) + ']'
            )
        if op_type not in _known_working_ops:
            if not self.allow_not_known_ops:
                raise AttributeError(
                    "Operator {} is not known to be safe".format(op_type))

            logging.warning("You are creating an op that the ModelHelper "
                            "does not recognize: {}.".format(op_type))
        return self.net.__getattr__(op_type)

    def __dir__(self):
        return sorted(set(chain(
            dir(type(self)),
            viewkeys(self.__dict__),
            _known_working_ops
        )))

    def GetCompleteNet(self):
        r""" Return param_init_net + net Net.
        Returns:
          'core.Net' containing param_init_net and net
        """
        new_net = self.param_init_net.Clone(
            self.name + "_complete_net", keep_schema=True)
        # add init net info to debug info
        for op in new_net.Proto().op:
            op.debug_info = op.debug_info + "/param_init_net"
        new_net.AppendNet(self.net)
        # keep the execution optimization
        if self.net.Proto().HasField("type"):
            new_net.Proto().type = self.net.Proto().type
        return new_net

    def ConstructInitTrainNetfromNet(self, net):
        r""" construct init net and train net from complete_net
        Inputs:
          net: 'core.Net' containing param_init_net and train net
        """
        param_op_mask = []
        train_op_mask = []
        for idx, op in enumerate(net.Proto().op):
            if op.debug_info.endswith("/param_init_net"):
                param_op_mask.append(idx)
            else:
                train_op_mask.append(idx)

        self.param_init_net = net.Clone(
            net.Name() + "/generated_param_init_net",
            keep_schema=True,
            op_id_mask=param_op_mask,
            update_external_list=True,
        )
        self.net = net.Clone(
            net.Name() + "/generated_net",
            keep_schema=True,
            op_id_mask=train_op_mask,
            update_external_list=True,
        )


def ExtractPredictorNet(
    net_proto,
    input_blobs,
    output_blobs,
    device=None,
    renames=None,
    disabled_inputs=None,
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
    @param disabled_inputs optional set of blobs that are 'switched off'. This
                will cause branches with those blobs as inputs to be removed
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

    if renames is None:
        renames = {}

    if disabled_inputs is not None:
        known_blobs = known_blobs - set(disabled_inputs)

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

    def validate_op(op):
        # Check that the op does not have is_test = 0 set. This is a common
        # pitfall with SpatialBN op, at lest.
        for arg in op.arg:
            if arg.name == "is_test" and arg.i == 0:
                raise Exception(
                    "An operator had is_test=0, did you try to extract a " +
                    "predictor from a train model (instead of test model)?" +
                    " Op was: {}".format(str(op))
                )

    def rename_list(proto_list):
        # proto lists don't support assignments
        new_list = proto_list[:]
        for j, b in enumerate(new_list):
            if b in renames:
                new_list[j] = renames[b]

        del proto_list[:]
        proto_list.extend(new_list)

    # Iterate through the ops and only include those whose inputs
    # we can satisfy.
    for op in ops[first_op_with_input:(last_op_with_output + 1)]:
        if known_blobs.issuperset(op.input):

            # Special handling for recurrent nets
            # TODO: when standard argument type for "nets" is introduced,
            # this can be more general
            if op.type == 'RecurrentNetwork':
                for arg in op.arg:
                    if arg.name == 'backward_step_net':
                        arg.ClearField(str('n'))
                    elif arg.name == 'step_net':
                        for step_op in arg.n.op:
                            rename_list(step_op.input)
                            rename_list(step_op.output)
                            if device is not None:
                                step_op.device_option.device_type = device.device_type
                                step_op.device_option.device_id = device.device_id

                        rename_list(arg.n.external_input)
                        rename_list(arg.n.external_output)

                        # Add additional external inputs
                        external_inputs.update(
                            set(arg.n.external_input).intersection(
                                orig_external_inputs
                            )
                        )

            if device is not None:
                op.device_option.device_type = device.device_type
                op.device_option.device_id = device.device_id
            validate_op(op)
            predict_proto.op.extend([op])
            known_blobs.update(op.output)
            external_inputs.update(
                set(op.input).intersection(orig_external_inputs)
            )
            external_outputs.update(
                set(op.output).intersection(orig_external_outputs)
            )

        else:
            logging.debug(
                "Op {} had unknown inputs: {}".format(
                    op.type, set(op.input).difference(known_blobs)
                )
            )

    # Predictor net's external inputs and outputs include only those
    # that are part of this net.
    predict_proto.external_input.extend(external_inputs)
    predict_proto.external_output.extend(external_outputs)

    rename_list(predict_proto.external_input)
    rename_list(predict_proto.external_output)

    renamed_input_blobs = []
    for b in input_blobs:
        if b in renames:
            renamed_input_blobs.append(renames[b])
        else:
            renamed_input_blobs.append(b)

    for op in predict_proto.op:
        rename_list(op.input)
        rename_list(op.output)

    return predict_net, list(
        set(predict_proto.external_input) - set(renamed_input_blobs)
    )
