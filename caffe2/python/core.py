import atexit
import sys

try:
  from .libcaffe2_python import *
except ImportError as e:
  print(str(e))
  print('Pycaffe is not available. Exiting.')
  sys.exit(1)
# libcaffe2_python contains a global Workspace that we need to properly delete
# when exiting. Otherwise, cudart will cause segfaults sometimes.
atexit.register(OnModuleExit)

from caffe2.proto import caffe2_pb2
from collections import Counter, defaultdict
from caffe2.python import utils, workspace

_REGISTERED_OPERATORS = set(
    s.decode() for s in workspace.RegisteredOperators())

def IsOperator(op_type):
  return (op_type in _REGISTERED_OPERATORS)

def GetGradientName(name):
  """The function that returns the gradient name for a blob."""
  return name + '_grad'

def IsGradientName(name):
  return name.endswith('_grad')

def GetOriginalName(name):
  """THe function that returns the original name for a gradient blob."""
  if not name.endswith('_grad'):
    raise RuntimeError('The blob ' + name + ' is not a legal gradient name.')
  return name[:-5]

class BlobReference(object):
  """A wrapper around a blob in a net.

  BlobReference gives us a way to refer to the network that the blob is
  generated from. Note that blobs are, essentially, just strings in the current
  workspace.
  """
  def __init__(self, name, net):
    self._name = name
    self._from_net = net
    # meta allows helper functions to put whatever metainformation needed there.
    self.meta = {}

  def __str__(self):
    return self._name

  def __add__(self, other):
    if not isinstance(other, str):
      raise RuntimeError('Cannot add BlobReference to a non-string.')
    return self._name + other

  def Net(self):
    return self._from_net

  def Grad(self):
    return GetGradientName(self._name)

  def __getattr__(self, op_type):
    """A wrapper allowing one to initiate operators from a blob reference.

    Example: for a blob reference b that comes from network n, doing
        b.Relu(...)
    is equivalent to doing
        net.Relu([b], ...)
    """
    if not IsOperator(op_type):
      raise RuntimeError(
          'Method ' + op_type + ' is not a registered operator.')
    def _CreateAndAddToNet(inputs=[], *args, **kwargs):
      """Internal function that routes the operator generation to the network's
      __getattr__ function.
      """
      if isinstance(inputs, BlobReference) or isinstance(inputs, str):
        inputs = [inputs]
      # add self to the input list.
      inputs.insert(0, self)
      return self._from_net.__getattr__(op_type)(inputs, *args, **kwargs)
    return _CreateAndAddToNet


def CreateOperator(operator_type):
  """A function wrapper that allows one to create operators based on the
  operator type. The type should be a string corresponding to an operator
  registered with Caffe2.
  """
  def ReallyCreate(inputs, outputs, name='', device_option=None,
                   arg=None, engine=None, **kwargs):
    operator = caffe2_pb2.OperatorDef()
    operator.type = operator_type
    operator.name = name
    if type(inputs) is str or type(inputs) is BlobReference:
      inputs = [inputs]
    elif type(inputs) is not list:
      raise ValueError("Unknown input format: %s of type %s."
                       % (str(inputs), type(inputs)))
    if type(outputs) is str or type(outputs) is BlobReference:
      outputs = [outputs]
    elif type(outputs) is not list:
      raise ValueError("Unknown output format: %s of type %s."
                       % (str(outputs), type(outputs)))
    operator.input.extend([str(i) for i in inputs])
    operator.output.extend([str(o) for o in outputs])
    if device_option is not None:
      operator.device_option.CopyFrom(device_option)
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
  return ReallyCreate


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
  def GetGradientDefsCC(cls, op_def):
    grad_defs_str = cc_GetGradientDefs(op_def.SerializeToString())
    grad_defs = []
    for grad_def_str in grad_defs_str:
      grad_def = caffe2_pb2.OperatorDef();
      grad_def.ParseFromString(grad_def_str)
      grad_defs.append(grad_def)
    return grad_defs

  @classmethod
  def GetGradientDefs(cls, op):
    try:
      gradient_ops = cls.GetGradientDefsCC(op)
    except:
      # Not supported in C++; will try python registration next.
      try:
        gradient_ops = cls.gradient_registry_[op.type](op)
      except KeyError as err:
        raise KeyError('No gradient registered for op: %s' % op.type)
    if gradient_ops is None:
      return []
    if type(gradient_ops) is not list:
      gradient_ops = [gradient_ops]
    if op.HasField("device_option"):
      for gradient_op in gradient_ops:
        gradient_op.device_option.CopyFrom(op.device_option)
    if op.HasField("engine"):
      for gradient_op in gradient_ops:
        gradient_op.engine = op.engine
    return gradient_ops

  @classmethod
  def GetBackwardPass(cls, operators, external_gradients=[]):
    # (1) "Play" the forward pass of the network, so we know the version of any
    #     tensors that are being written multiple times.
    # After running this, we will have:
    # a) fwd_metadata: a list of [op, input_versions, output_versions]
    #    recording the input and the output version of the operator.
    # b) versioned_input_count: a dictionary specifying for each blob and each
    #    of its version, how many times it is used as input for another op.
    # c) current_versions: maintaining the current versions of the tensors we
    #    are having in the workspace. This is useful because if a gradient is
    #    trying to access an earlier version of a blob, we know that it is no
    #    longer there, and thus it should not be referred to at all.
    current_versions = defaultdict(int)
    versioned_input_count = defaultdict(lambda: defaultdict(int))
    fwd_metadata = []
    for op in operators:
      # For input, they are the current version in the dict.
      input_versions = dict()
      for s in op.input:
        input_versions[s] = current_versions[s]
        versioned_input_count[s][current_versions[s]] += 1
      # For output, they are the current version plus one. If this is a newly
      # created blob, its version starts with zero.
      output_versions = dict()
      for s in op.output:
        if s in current_versions:
          current_versions[s] += 1
        output_versions[s] = current_versions[s]
      fwd_metadata.append([op, input_versions, output_versions])

    # (2) Now, after having the virtual play above, we now play the operators
    # backwards, creating the gradients along the path. Note that although we
    # are playing it backwards, any value being overwritten can not be
    # recovered, and any reference to a blob already being overwritten would
    # trigger an error.

    all_gradient_ops = []
    current_gradient_versions = dict(
        (s, current_versions[GetOriginalName(s)]) for s in external_gradients)
    versioned_gradient_count = defaultdict(lambda: defaultdict(int))
    for forward_op, current_fwd_metadata in zip(operators[::-1], fwd_metadata[::-1]):
      gradient_ops = cls.GetGradientDefs(forward_op)
      # Now, the constraints for the inputs of the gradient operators are:
      #
      # (1) for inputs:
      # (1a) If it is a gradient name, it should match the version of the
      #      corresponding output.
      # (1b) If it is an output name, the current version should match the
      #      version when the operator was run.
      # (1c) If it is an input name, the current version should match the
      #      version when the operator was run.
      # (1d) If it is none of the above, it should be a blob that is generated
      #      locally by one of the previous gradient operators.
      #
      # (2) for outputs:
      # (2a) If it is a gradient name, it must be the gradient name of an input
      #      blob, and we will mark the gradient as being corresponding to the
      #      version of the input.
      # (2b) If it is anything else it is OK - we will simply "play" the op to
      #      update the current versions of blobs.
      locally_generated_blobs = []
      multiuse_input_ready_to_sum = []
      for grad_op in gradient_ops:
        for s in grad_op.input:
          if IsGradientName(s):
            if s not in current_gradient_versions:
              raise RuntimeError(
                  'Input gradient name "%s" is referred to but '
                  'is never generated.' % s)
            # This is a gradient name. We will need to check if this gradient is
            # produced already, and if this is the gradient we want.
            original_name = GetOriginalName(s)
            if original_name not in current_fwd_metadata[2]:
              raise RuntimeError(
                  'Input gradient name "%s" is not the gradient '
                  'of any of the op\'s output.' % s)
            if (current_fwd_metadata[2][original_name] !=
                current_gradient_versions[s]):
              raise RuntimeError(
                  'Gradient name "%s" is expected to correspond to '
                  'version %d of "%s", but currently we have version %d.' %
                  (s, current_fwd_metadata[2][s], original_name,
                   current_gradient_versions[s]))
          elif s in current_fwd_metadata[2]:
            if (current_versions[s] != current_fwd_metadata[2][s]):
              raise RuntimeError(
                  'Gradient operator needs output "%s" at version '
                  '%d, but currently we have version %d.' %
                  (s, current_fwd_metadata[2][s], current_versions[s]))
          elif s in current_fwd_metadata[1]:
            if (current_versions[s] != current_fwd_metadata[1][s]):
              raise RuntimeError(
                  'Gradient operator needs input "%s" at version '
                  '%d, but currently we have version %d.' %
                  (s, current_fwd_metadata[1][s], current_versions[s]))
          else:
            if s not in locally_generated_blobs:
              if s not in locally_generated_blobs:
                raise RuntimeError(
                    'Blob name "%s" not in the scope of operator: %s\nand is '
                    'not generated by any of the local gradient operators.' %
                    (s, str(current_fwd_metadata[0])))
        for idx, s in enumerate(grad_op.output):
          if IsGradientName(s):
            original_name = GetOriginalName(s)
            if original_name not in current_fwd_metadata[1]:
              raise RuntimeError(
                  'Output gradient name "%s" is not the '
                  'gradient of any of the op\'s input name.' % s)
            # Set the current gradient version.
            version = current_fwd_metadata[1][original_name]
            current_gradient_versions[s] = version
            # Now we should also check if the gradient we product is a multi-use
            # input, in which case we will automatically add split nodes.
            # TODO: Instead of adding split nodes, we can also choose to
            # sequentially compute and accumulate gradients. Maybe implement
            # that in the future.
            if versioned_input_count[original_name][version] > 1:
              grad_op.output[idx] = '_%s_autosplit_%d' % (
                  s, versioned_gradient_count[s][version])
              versioned_gradient_count[s][version] += 1
              assert (versioned_gradient_count[s][version] <=
                      versioned_input_count[original_name][version])
              if (versioned_gradient_count[s][version] ==
                  versioned_input_count[original_name][version]):
                # We have calculated all the autosplit gradients. Will need to
                # add a sum after this gradient computation.
                multiuse_input_ready_to_sum.append(
                    (s, versioned_gradient_count[s][version], grad_op))
          else:
            locally_generated_blobs.append(s)
      # If some of the multi use inputs are ready to be summed, we will do so.
      for s, count, source_op in multiuse_input_ready_to_sum:
        additional_sum_op = CreateOperator('Sum')(
            ['_%s_autosplit_%d' % (s, i) for i in range(count)], [s])
        if source_op.HasField('device_option'):
          additional_sum_op.device_option.CopyFrom(source_op.device_option)
        gradient_ops.append(additional_sum_op)
      # Now, for bookkeeping purposes, we will need to "play" the gradient
      # operators. The reason is that the gradient operators may (although in
      # most cases they shouldn't) change some of the existing blobs, in which
      # case this explicit bookkeeping is going to catch them.
      for op in gradient_ops:
        for s in op.output:
          if s in current_versions:
            current_versions[s] += 1
          output_versions[s] = current_versions[s]
      all_gradient_ops += gradient_ops
    # After we have done computation for each op, we now have the gradient
    # operators ready.
    return all_gradient_ops


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
          sum([list(op.input) for op in self._net.op], []) +
          sum([list(op.output) for op in self._net.op], []))
      prefix_len = len(self._net.name + '_blob_')
      autogen_indices = [int(name[prefix_len:]) for name in existing_names
                       if name.startswith(self._net.name + '_blob_')]
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

  def AddGradientOperators(self, skip=0, external_gradients=[]):
    """Add the gradient for operators in the net.

    Inputs:
      skip: skips the first n operators. This is provided mainly because a lot
          of nets may use the first few operators for data generation like stuff
          which really do not need to have gradients.

    Currently, this is hard-coded for float operators if there are branches
    (i.e. a blob is used as input to multiple operators). This is because the
    inserted SplitOp is hard-coded for float (its gradient, SumOp, is float
    only). Supporting other formats is a todo item.
    """
    grad_ops = GradientRegistry.GetBackwardPass(self._net.op[skip:])
    self._net.op.extend(grad_ops)
    return

  def RunAllOnGPU(self, gpu_id=0, use_cudnn=False):
    """A convenient function to run everything on the GPU."""
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.cuda_gpu_id = gpu_id
    self._net.device_option.CopyFrom(device_option)
    if use_cudnn:
      for op in self._net.op:
        op.engine = "CUDNN"

  def __getattr__(self, op_type):
    if not IsOperator(op_type):
      raise RuntimeError(
          'Method ' + op_type + ' is not a registered operator.')
    def _CreateAndAddToSelf(inputs, outputs=None, **kwargs):
      if outputs is None:
        # If we do not specify an output, we will assume that this operator
        # produces one output in this case.
        outputs = self.NextName()
      elif type(outputs) is int:
        # In this case, we will auto-fill the given number of outputs with
        # auto-generated names.
        outputs = [self.NextName() for i in range(outputs)]
      op = CreateOperator(op_type)(inputs, outputs, **kwargs)
      self._net.op.extend([op])
      if len(op.output) == 0:
        return
      elif len(op.output) == 1:
        return BlobReference(str(op.output[0]), self)
      else:
        return tuple(BlobReference(str(o), self) for o in op.output)
    return _CreateAndAddToSelf


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

