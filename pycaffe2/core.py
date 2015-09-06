from caffe2.proto import caffe2_pb2
from collections import Counter, defaultdict
from pycaffe2 import utils

def GetGradientName(name):
  """The function that returns the gradient name for a blob."""
  return name + '_grad'

class BlobReference(object):
  """A wrapper around a blob in a net.

  BlobReference gives us a way to refer to the network that the blob is
  generated from. Note that blobs are, essentially, just strings in the current
  workspace.
  """
  def __init__(self, name, net):
    self._name = name
    self._from_net = net

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
                   arg=None, **kwargs):
    operator = caffe2_pb2.OperatorDef()
    operator.type = operator_type
    operator.name = name
    if type(inputs) is str or type(inputs) is BlobReference:
      inputs = [inputs]
    elif type(inputs) is unicode:
      inputs = [str(inputs)]
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
    if device_option:
      operator.device_option.CopyFrom(device_option)
    # random seed is defined in the device option, so we need to do special
    # care.
    if 'random_seed' in kwargs:
      operator.device_option.random_seed = kwargs['random_seed']
      del kwargs['random_seed']
    # Add given arguments that do not need parsing
    if arg:
      operator.arg.extend(arg)
    # Add all other arguments
    for key, value in kwargs.iteritems():
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
  def GetGradient(cls, op):
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
    return gradient_ops


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

  def AddGradientOperators(self, skip=0):
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
    # (1) Make sure that the network is "legal" in terms of computing gradients:
    # for every blob there is only going to be one operator that generates it.
    all_outputs = sum([list(op.output) for op in self._net.op], [])
    if len(all_outputs) != len(set(all_outputs)):
      # There is some output that is produced by multiple operators. This is not
      # good.
      raise RuntimeError("Some blobs are produced multiple times. A count is "
                         "as follows: " + str(Counter(all_outputs)))
    # (2) For cases when a blob is being used by multiple operators, we will
    # need to take special care. Currently, we will ask the operators to compute
    # the gradients, and add aggregation operators to get the final gradient.
    input_counts = Counter(
        sum([list(op.input) for op in self._net.op[skip:]], []))
    multiple_use_blobs = set(
        [key for key in input_counts if input_counts[key] > 1])

    # Now, if there are multiple use blobs, it means that we are going to have
    # shared parameters, and we want to make special care for them. The
    # conventional strategy in Caffe is to insert a SplitLayer that splits
    # the input into multiple blobs, with the split layer automatically
    # accumulates gradients. This makes the AddGradientOperators stateful, in
    # the sense that it may change existing operators, which is not desired.
    # In this implementation we will keep the original operator, and instead
    # manually modify the gradient names.
    num_ops_before_grad = len(self._net.op)
    # Obtain the gradient operators that we need. Note that these gradient
    # operators are going to be refined, as we need to figure out shared
    # parameters.
    gradient_ops = sum(
        [GradientRegistry.GetGradient(self._net.op[i])
         for i in xrange(num_ops_before_grad - 1, skip - 1, -1)], [])
    if len(multiple_use_blobs) == 0:
      # There is no concern about shared parameters, so we can simply skip.
      self._net.op.extend(gradient_ops)
      return
    # Now, if there are multiple use operators, we will need to figure out
    # any gradients that are overwriting each other.
    # To do this, we do a first pass over the gradients to count the number
    # of occurences of the gradients.
    gradient_occurences = defaultdict(int)
    for blob_name in multiple_use_blobs:
      count = 0
      blob_gradient_name = GetGradientName(blob_name)
      for op in gradient_ops:
        for output_name in op.output:
          if output_name == blob_gradient_name:
            gradient_occurences[blob_gradient_name] += 1
    # For anything that only has one gradient blob generated, we don't need
    # to take any special care.
    for key in gradient_occurences.keys():
      if gradient_occurences[key] == 1:
        del gradient_occurences[key]
    # Now, we add the gradient ops back to the network, modifying the
    # gradient names on the fly.
    grad_encountered = defaultdict(int)
    for op in gradient_ops:
      additional_sum_ops = []
      for i, grad_name in enumerate(op.output):
        if grad_name in gradient_occurences:
          # rename the gradient to an intermediate name
          op.output[i] = (
              '_%s_autosplit_%d' % (grad_name, grad_encountered[grad_name]))
          grad_encountered[grad_name] += 1
          if grad_encountered[grad_name] == gradient_occurences[grad_name]:
            # We have encountered all gradient names; time to add a SumOp.
            additional_sum_ops.append(
                CreateOperator('Sum')(
                    ['_%s_autosplit_%d' % (grad_name, i)
                     for i in range(gradient_occurences[grad_name])],
                    [grad_name]))
      # After re-writing the outputs, we can safely add it to the network.
      self._net.op.extend([op])
      self._net.op.extend(additional_sum_ops)
    return

  def RunAllOnGPU(self, gpu_id=0):
    """A convenient function to run everything on the GPU."""
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.cuda_gpu_id = gpu_id
    self._net.device_option.CopyFrom(device_option)

  def __getattr__(self, operator_type):
    if operator_type in self.__class__.operator_registry_:
      # Not finished. Operator registry allows one to define custon functions,
      # but so far that functionality is not complete.
      return self.__class__.operator_registry_
    def _CreateAndAddToSelf(inputs, outputs=None, **kwargs):
      if outputs is None:
        # If we do not specify an output, we will assume that this operator
        # produces one output in this case.
        outputs = self.NextName()
      elif type(outputs) is int:
        # In this case, we will auto-fill the given number of outputs with
        # auto-generated names.
        outputs = [self.NextName() for i in range(outputs)]
      op = CreateOperator(operator_type)(inputs, outputs, **kwargs)
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

