from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import atexit
import sys
from collections import namedtuple

from ._import_c_extension import *

from caffe2.proto import caffe2_pb2
from collections import Counter, defaultdict
from caffe2.python import utils, workspace

import logging

_REGISTERED_OPERATORS = set(s.decode() for s in workspace.RegisteredOperators())

def IsOperator(op_type):
    return (op_type in _REGISTERED_OPERATORS)

# The name scope and device scope when creating a new operator.
_NAMESCOPE = ''
_DEVICESCOPE = None


class NameScope(object):
    """Helper class to create embedded name scopes."""
    # SEPARATOR is defined to be "/" so it is consistent with TensorFlow's
    # visualization tools.
    SEPARATOR = '/'
    def __init__(self, prefix):
        assert isinstance(prefix, basestring), \
            "NameScope takes in a string as its argument."
        self._prefix = prefix + NameScope.SEPARATOR

    def __enter__(self):
        global _NAMESCOPE
        _NAMESCOPE += self._prefix

    def __exit__(self, type, value, traceback):
        global _NAMESCOPE
        assert _NAMESCOPE.endswith(self._prefix), \
            "The namescope variable is changed from outside NameScope() calls."
        _NAMESCOPE = _NAMESCOPE[:-len(self._prefix)]


class DeviceScope(object):
    """Helper class to switch device scopes."""
    def __init__(self, scope):
        assert isinstance(scope, caffe2_pb2.DeviceOption), \
            "DeviceScope takes in a caffe2_pb2.DeviceOption as its argument."
        self._scope = scope

    def __enter__(self):
        global _DEVICESCOPE
        self._old_scope = _DEVICESCOPE
        _DEVICESCOPE = self._scope

    def __exit__(self, type, value, traceback):
        global _DEVICESCOPE
        assert _DEVICESCOPE == self._scope, \
            "The device scope is changed from outside DeviceScope() calls."
        _DEVICESCOPE = self._old_scope


def DeviceOption(device_type, cuda_gpu_id, random_seed=None):
    option = caffe2_pb2.DeviceOption()
    option.device_type = device_type
    option.cuda_gpu_id = cuda_gpu_id
    if random_seed is not None:
        option.random_seed = random_seed
    return option


GradientSlice = namedtuple('GradientSlice', ['indices', 'values'])


class BlobReference(object):
    """A wrapper around a blob in a net.

    BlobReference gives us a way to refer to the network that the blob is
    generated from. Note that blobs are, essentially, just strings in the
    current workspace.
    """

    def __init__(self, name, net=None):
        self._name = name
        self._from_net = net
        # meta allows helper functions to put whatever metainformation needed
        # there.
        self.meta = {}

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if isinstance(other, basestring):
            return self._name == other
        elif isinstance(other, BlobReference):
            return self._name == other._name
        else:
            return False

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return self._name

    def __add__(self, other):
        if not isinstance(other, basestring):
            raise RuntimeError('Cannot add BlobReference to a non-string.')
        return BlobReference(self._name + other, self._from_net)

    def Net(self):
        return self._from_net

    def _CreateAndAddToNet(self, op_type, inputs=None, *args, **kwargs):
        """Internal function that routes the operator generation to the
        network's __getattr__ function.
        """
        inputs = [] if inputs is None else inputs
        if isinstance(inputs, BlobReference) or isinstance(inputs, str):
            inputs = [inputs]
        # add self to the input list.
        inputs.insert(0, self)
        return self._from_net.__getattr__(op_type)(inputs, *args, **kwargs)

    def __getattr__(self, op_type):
        """A wrapper allowing one to initiate operators from a blob reference.

        Example: for a blob reference b that comes from network n, doing
            b.Relu(...)
        is equivalent to doing
            net.Relu([b], ...)
        """
        if self._from_net is None:
            raise RuntimeError(
                'You cannot use a blob reference that does not have a net '
                'source to create operators. Create the operator from an '
                'explicit net object.')
        if not IsOperator(op_type):
            raise RuntimeError(
                'Method ' + op_type + ' is not a registered operator.'
            )
        return lambda *args, **kwargs: self._CreateAndAddToNet(
            op_type, *args, **kwargs)

def _RectifyInputOutput(blobs):
    """A helper function to rectify the input or output of the CreateOperator
    interface.
    """
    if isinstance(blobs, basestring):
        # If blobs is a single string, prepend _NAMESCOPE and put it as a list.
        # TODO(jiayq): enforce using BlobReference instead of raw strings.
        return [BlobReference(_NAMESCOPE + blobs)]
    elif type(blobs) is BlobReference:
        # If blob is a BlobReference, simply put it as a list.
        return [BlobReference(str(blobs))]
    elif type(blobs) is list:
        # If blob is a list, we go through it and type check.
        rectified = []
        for blob in blobs:
            if isinstance(blob, basestring):
                rectified.append(BlobReference(_NAMESCOPE + blob))
            elif type(blob) is BlobReference:
                rectified.append(BlobReference(str(blob)))
            else:
                raise TypeError(
                    "I/O blob #{} of unsupported type: {} of type {}"
                    .format(len(rectified), str(blob), type(blob)))
        return rectified
    else:
        raise TypeError(
            "Unknown input/output type: %s of type %s." %
            (str(inputs), type(inputs))
        )


def CreateOperator(
    operator_type,
    inputs,
    outputs,
    name='',
    device_option=None,
    arg=None,
    engine=None,
    **kwargs
):
    """A function wrapper that allows one to create operators based on the
    operator type. The type should be a string corresponding to an operator
    registered with Caffe2.
    """
    operator = caffe2_pb2.OperatorDef()
    operator.type = operator_type
    operator.name = name
    # Add rectified inputs and outputs
    inputs = _RectifyInputOutput(inputs)
    outputs = _RectifyInputOutput(outputs)
    operator.input.extend([str(i) for i in inputs])
    operator.output.extend([str(o) for o in outputs])
    # Set device option:
    # (1) If device_option is explicitly set, use device_option.
    # (2) If not, but _DEVICESCOPE is set, then we use the _DEVICESCOPE.
    # (3) Otherwise, do not set device option.
    if device_option is not None:
        operator.device_option.CopyFrom(device_option)
    elif _DEVICESCOPE is not None:
        operator.device_option.CopyFrom(_DEVICESCOPE)
    if engine is not None:
        operator.engine = engine
    # random seed is defined in the device option, so we need to do special
    # care.
    if 'random_seed' in kwargs:
        operator.device_option.random_seed = kwargs['random_seed']
        del kwargs['random_seed']
    # Add given arguments that do not need parsing
    if arg is not None:
        operator.arg.extend(arg)
    # Add all other arguments
    for key, value in kwargs.items():
        operator.arg.add().CopyFrom(utils.MakeArgument(key, value))
    return operator


class GradientRegistry(object):
    """GradientRegistry holds the mapping from operators to their gradients."""
    gradient_registry_ = {}

    @classmethod
    def RegisterGradient(cls, op_type):
        """A decorator for registering gradient mappings."""

        def Wrapper(func):
            cls.gradient_registry_[op_type] = func
            return func

        return Wrapper

    @classmethod
    def _GetGradientForOpCC(cls, op_def, g_output):
        grad_defs_str, g_input = cc_GetGradientDefs(
            op_def.SerializeToString(), g_output)
        # C++ return tuple for sparse gradients, and we will convert it to
        # namedtuple here.
        g_input = [
            (GradientSlice(*g) if type(g) is tuple else g)
            for g in g_input
        ]
        grad_defs = []
        for grad_def_str in grad_defs_str:
            grad_def = caffe2_pb2.OperatorDef()
            grad_def.ParseFromString(grad_def_str)
            grad_defs.append(grad_def)
        return grad_defs, g_input

    @classmethod
    def GetGradientForOp(cls, op, g_output):
        try:
            gradient_ops, g_input = cls._GetGradientForOpCC(op, g_output)
        except Exception as err:
            # Not supported in C++; will try python registration next.
            try:
                gradient_ops, g_input = cls.gradient_registry_[op.type](
                    op, g_output)
            except KeyError as err:
                raise KeyError('No gradient registered for op: %s' % op.type)
        if gradient_ops is None:
            return [], g_input
        if type(gradient_ops) is not list:
            gradient_ops = [gradient_ops]
        if op.HasField("device_option"):
            for gradient_op in gradient_ops:
                gradient_op.device_option.CopyFrom(op.device_option)
        if op.HasField("engine"):
            for gradient_op in gradient_ops:
                gradient_op.engine = op.engine
        return gradient_ops, g_input

    @classmethod
    def GetBackwardPass(cls, operators, external_gradients=None):
        # external_gradients should be a map of {blob -> gradient} where gradient
        # can be either a single blob or GradientSlice.

        all_input_to_grad = {}
        if external_gradients is None:
            external_gradients = {}
        elif not isinstance(external_gradients, dict):
            raise TypeError("external_gradients should be a dictionary.")
        else:
            for inp, g in external_gradients.iteritems():
                # Since the C++ gradient registry does not have notion of
                # NameScopes, we will convert all references to strings.
                all_input_to_grad[str(inp)] = (
                    GradientSlice(str(g[0]), str(g[1]))
                    if isinstance(g, GradientSlice) else str(g))
        # (1) "Play" the forward pass of the network, so we know the version of
        #    any tensors that are being written multiple times.
        # After running this, we will have:
        # a) fwd_metadata: a list of [op, input_versions, output_versions]
        #    recording the input and the output version of the operator.
        # b) versioned_input_count: a dictionary specifying for each blob and
        #    each of its version, how many times it is used as input for another
        #    op.
        # c) current_versions: maintaining the current versions of the tensors
        #    we are having in the workspace. This is useful because if a
        #    gradient is trying to access an earlier version of a blob, we know
        #    that it is no longer there, and thus it should not be referred to
        #    at all.
        current_versions = defaultdict(int)
        versioned_input_count = defaultdict(lambda: defaultdict(int))
        fwd_metadata = []
        OpFwdMetadata = namedtuple(
            'OpFwdMetadata', ['op', 'input_versions', 'output_versions']
        )
        for op in operators:
            # For input, they are the current version in the dict.
            input_versions = {}
            for s in op.input:
                input_versions[s] = current_versions[s]
                versioned_input_count[s][current_versions[s]] += 1
            # For output, they are the current version plus one. If this is a
            # newly created blob, its version starts with zero.
            output_versions = {}
            for s in op.output:
                if s in current_versions:
                    current_versions[s] += 1
                output_versions[s] = current_versions[s]
            fwd_metadata.append(
                OpFwdMetadata(op, input_versions, output_versions)
            )

        # (2) Now, after having the virtual play above, we now play the ops
        # backwards, creating the gradients along the path. Note that although
        # we are playing it backwards, any value being overwritten can not be
        # recovered, and any reference to a blob already being overwritten would
        # trigger an error.

        all_gradient_ops = []
        # current_gradient_versions maps the name of the original blob to its
        # version that the gradient corresponds to.
        current_gradient_versions = {}
        for s, g in external_gradients.iteritems():
            current_gradient_versions[s] = current_versions[s]
        versioned_gradient_count = defaultdict(lambda: defaultdict(int))
        for forward_op, cur_fwd in zip(operators[::-1], fwd_metadata[::-1]):
            g_output = list(
                all_input_to_grad.get(name, None) for name in forward_op.output)
            gradient_ops, g_input = cls.GetGradientForOp(forward_op, g_output)
            # Now, the constraints for the inputs of the gradient operators are:
            #
            # (1) for inputs:
            # (1a) If it is a dense or sparse gradient name, it should match the
            #      version of the corresponding output.
            # (1b) If it is an output name, the current version should match the
            #      version when the operator was run.
            # (1c) If it is an input name, the current version should match the
            #      version when the operator was run.
            # (1d) If it is none of the above, it should be a blob that is
            #      generated locally by one of the previous gradient operators.
            #
            # (2) for outputs:
            # (2a) If it is a gradient name, it must be the gradient name of an
            #      input blob, and we will mark the gradient as being
            #      corresponding to the version of the input.
            # (2b) If it is anything else it is OK - we will simply "play" the
            #      op to update the current versions of blobs.
            locally_generated_blobs = []
            multiuse_input_ready_to_sum = []
            for grad_op in gradient_ops:
                # (1)
                for s in grad_op.input:
                    # TODO(jiayq): yuck. clean this statement.
                    original_indices = [
                        i for i, g in enumerate(g_output)
                        if ((type(g) is GradientSlice and
                             (g.indices == s or g.values == s))
                            or g == s)]
                    # (1a)
                    if len(original_indices):
                        original_name = forward_op.output[original_indices[0]]
                        if (
                            cur_fwd.output_versions[original_name] !=
                            current_gradient_versions[original_name]
                        ):
                            raise RuntimeError(
                                'Gradient name "%s" is expected to correspond '
                                'to version %d of "%s", but currently we have '
                                'version %d.' % (
                                    s, cur_fwd.output_versions[original_name],
                                    original_name,
                                    current_gradient_versions[original_name])
                            )
                    # (1b)
                    elif s in cur_fwd.output_versions:
                        if current_versions[s] != cur_fwd.output_versions[s]:
                            raise RuntimeError(
                                'Gradient operator needs output "%s" at version'
                                ' %d, but currently we have version %d.' % (
                                    s, cur_fwd.output_versions[s],
                                    current_versions[s]
                                )
                            )
                    # (1c)
                    elif s in cur_fwd.input_versions:
                        if (current_versions[s] != cur_fwd.input_versions[s]):
                            raise RuntimeError(
                                'Gradient operator needs input "%s" at version '
                                '%d, but currently we have version %d.' % (
                                    s, cur_fwd.input_versions[s],
                                    current_versions[s]
                                )
                            )
                    # (1d)
                    else:
                        if s not in locally_generated_blobs:
                            raise RuntimeError(
                                'Blob name "%s" not in the scope of operator: '
                                '%s\nand is not generated by any of the local '
                                'gradient operators.' % (s, str(cur_fwd.op))
                            )
                # (2)
                for idx, s in enumerate(grad_op.output):
                    original_indices = [
                        i for i, g in enumerate(g_input)
                        if ((type(g) is GradientSlice and
                             (g.indices == s or g.values == s))
                            or g == s)]
                    # (2a)
                    if len(original_indices):
                        original_idx = original_indices[0]
                        original_name = forward_op.input[original_idx]
                        # Set the current gradient version.
                        version = cur_fwd.input_versions[original_name]
                        current_gradient_versions[original_name] = version
                        # Now we should also check if the gradient we product is
                        # a multi-use input, in which case we will automatically
                        # add split nodes.
                        # TODO: Instead of adding split nodes, we can also
                        # choose to sequentially compute and accumulate
                        # gradients. Maybe implement that in the future.
                        if versioned_input_count[original_name][version] > 1:
                            assert type(g_input[original_idx]) \
                                is not GradientSlice, \
                                'Automatic splitting does not work with ' \
                                'sparse gradients yet.'
                            # rename the gradient.
                            grad_op.output[idx] = '_%s_autosplit_%d' % (
                                s, versioned_gradient_count[s][version]
                            )
                            versioned_gradient_count[s][version] += 1
                            assert (
                                versioned_gradient_count[s][version] <=
                                versioned_input_count[original_name][version]
                            )
                            if (
                                versioned_gradient_count[s][version] ==
                                versioned_input_count[original_name][version]
                            ):
                                # We have calculated all the autosplit gradients
                                # and will now need to add a sum after this
                                # gradient computation.
                                multiuse_input_ready_to_sum.append(
                                    (
                                        s, versioned_gradient_count[s][
                                            version
                                        ], grad_op
                                    )
                                )
                    else:
                        # (2b)
                        locally_generated_blobs.append(s)
            # If some of the multi use inputs are ready to be summed, we will do
            # so.
            for s, count, source_op in multiuse_input_ready_to_sum:
                additional_sum_op = CreateOperator(
                    'Sum',
                    ['_%s_autosplit_%d' % (s, i) for i in range(count)], [s]
                )
                if source_op.HasField('device_option'):
                    additional_sum_op.device_option.CopyFrom(
                        source_op.device_option
                    )
                gradient_ops.append(additional_sum_op)
            for name, grad in zip(forward_op.input, g_input):
                all_input_to_grad[name] = grad

            # Now, for bookkeeping purposes, we will need to "play" the gradient
            # operators. The reason is that the gradient operators may (although
            # in most cases they shouldn't) change some of the existing blobs,
            # in which case this explicit bookkeeping is going to catch them.
            for op in gradient_ops:
                for s in op.output:
                    if s in current_versions:
                        current_versions[s] += 1
                    output_versions[s] = current_versions[s]
            all_gradient_ops += gradient_ops
        # After we have done computation for each op, we now have the gradient
        # operators ready.
        # For the output map, we will convert everything to BlobReferences.
        all_input_to_grad_out = {}
        for key, val in all_input_to_grad.items():
            if val is not None:
                all_input_to_grad_out[BlobReference(key)] = (
                    BlobReference(val) if isinstance(val, basestring) else
                    GradientSlice(BlobReference(val[0]), BlobReference(val[1])))
        return all_gradient_ops, all_input_to_grad_out


class Net(object):
    operator_registry_ = {}

    def __init__(self, name):
        if type(name) is caffe2_pb2.NetDef:
            # We rae initializing a network by a NetDef. In this case, we will
            # initialize our network with the given netdef.
            self._net = caffe2_pb2.NetDef()
            self._net.CopyFrom(name)
            # Set the next name index properly.
            existing_names = set(
                sum(
                    [list(op.input) for op in self._net.op], []
                ) + sum(
                    [list(op.output) for op in self._net.op], []
                )
            )
            prefix_len = len(self._net.name + '_blob_')
            autogen_indices = [
                int(name[prefix_len:])
                for name in existing_names
                if name.startswith(self._net.name + '_blob_')
            ]
            if len(autogen_indices):
                self._next_name_index = max(autogen_indices) + 1
            else:
                self._next_name_index = 0
        else:
            self._net = caffe2_pb2.NetDef()
            self._net.name = name
            self._next_name_index = 0

    def __str__(self):
        return self._net.name

    def Proto(self):
        return self._net

    def NextName(self):
        """Returns the next name to be used, if you do not want to explicitly
        name your blob."""
        output_name = self._net.name + '_blob_' + str(self._next_name_index)
        self._next_name_index += 1
        return str(output_name)

    def AddGradientOperators(self, skip=0, external_gradients=None):
        """Add the gradient for operators in the net.

        Inputs:
          skip: skips the first n operators. This is provided mainly because a
              lot of nets may use the first few operators for data generation
              like stuff which really do not need to have gradients.

        Outputs:
          returns a map from the blob name in the input network to a blob
          containing gradient or a GradientSlice in case of sparse gradient

        Currently, this is hard-coded for float operators if there are branches
        (i.e. a blob is used as input to multiple operators). This is because
        the inserted SplitOp is hard-coded for float (its gradient, SumOp, is
        float only). Supporting other formats is a todo item.
        """

        grad_ops, input_to_grad = GradientRegistry.GetBackwardPass(
            self._net.op[skip:], external_gradients
        )
        self._net.op.extend(grad_ops)
        return input_to_grad

    def RunAllOnGPU(self, gpu_id=0, use_cudnn=False):
        """A convenient function to run everything on the GPU."""
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = gpu_id
        self._net.device_option.CopyFrom(device_option)
        if use_cudnn:
            for op in self._net.op:
                op.engine = "CUDNN"

    def _CreateAndAddToSelf(self, op_type, inputs, outputs=None, **kwargs):
        """A helper function to create an operator and add it to self.
        """
        if outputs is None:
            # If we do not specify an output, we will assume that this op
            # produces one output in this case.
            outputs = self.NextName()
        elif type(outputs) is int:
            # In this case, we will auto-fill the given number of outputs
            # with auto-generated names.
            outputs = [self.NextName() for i in range(outputs)]
        op = CreateOperator(op_type, inputs, outputs, **kwargs)
        self._net.op.extend([op])
        if len(op.output) == 0:
            return
        elif len(op.output) == 1:
            return BlobReference(str(op.output[0]), self)
        else:
            return tuple(BlobReference(str(o), self) for o in op.output)


    def __getattr__(self, op_type):
        if not IsOperator(op_type):
            raise RuntimeError(
                'Method ' + op_type + ' is not a registered operator.'
            )
        return lambda *args, **kwargs: self._CreateAndAddToSelf(
                op_type, *args, **kwargs)


class ExecutionStep(object):
    def __init__(self, name):
        self._step = caffe2_pb2.ExecutionStep()
        self._step.name = name

    def __init__(self, name, nets, num_iter=None):
        self._step = caffe2_pb2.ExecutionStep()
        self._step.name = name
        if type(nets) is Net:
            nets = [nets]
        self._step.network.extend([str(n) for n in nets])
        if num_iter is not None:
            self._step.num_iter = num_iter

    def __str__(self):
        return self._step.name

    def Proto(self):
        return self._step

    def SetIter(self, num_iter):
        self._step.num_iter = num_iter

    def AddSubstep(self, substep):
        self._step.substep.add().CopyFrom(substep)

    def AddNet(self, net):
        self._step.network.add(str(net))


class Plan(object):
    def __init__(self, name):
        self._plan = caffe2_pb2.PlanDef()
        self._plan.name = name

    def __str__(self):
        return self._plan.name

    def Proto(self):
        return self._plan

    def AddNets(self, nets):
        for net in nets:
            self._plan.network.add().CopyFrom(net.Proto())

    def AddStep(self, step):
        self._plan.execution_step.add().CopyFrom(step.Proto())
