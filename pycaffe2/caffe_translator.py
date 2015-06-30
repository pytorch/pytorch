from caffe2.proto import caffe2_pb2
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from pycaffe2 import utils


def _StateMeetsRule(state, rule):
  """A function that reproduces Caffe's StateMeetsRule functionality."""
  if rule.HasField('phase') and rule.phase != state.phase:
    return False
  if rule.HasField('min_level') and state.level < rule.min_level:
    return False
  if rule.HasField('max_level') and state.level > rule.max_lavel:
    return False
  curr_stages = set(list(state.stage))
  # all stages in rule.stages should be in, otherwise it's not a match.
  if len(rule.stage) and any([s not in curr_stages for s in rule.stage]):
    return False
  # none of the stage in rule.stages should be in, otherwise it's not a match.
  if len(rule.not_stage) and any([s in curr_stages for s in rule.not_stage]):
    return False
  # If none of the nonmatch happens, return True.
  return True


def _ShouldInclude(net_state, layer):
  """A function that reproduces Caffe's inclusion and exclusion rule."""
  ret = (len(layer.include) == 0)
  # check exclude rules: if any exclusion is met, we shouldn't include.
  ret &= not any([_StateMeetsRule(net_state, rule) for rule in layer.exclude])
  if len(layer.include):
    # check include rules: if any inclusion is met, we should include.
    ret |= any([_StateMeetsRule(net_state, rule) for rule in layer.include])
  return ret


def DeleteDropout(net):
  """A utility function that replaces all dropout operators with Alias.

  The reason for this is that Caffe involves Dropout in both training and
  testing, and uses a global mode to determine whether we are training or
  testing a model. Instead of that, what Caffe2 does is to remove that global
  mode, and explicitly require the network to NOT contain a dropout operator.
  In this function, we will simply replace all dropouts with an Alias operator.

  Inputs:
    net: a caffe2 net.
  Outputs:
    None. The function works by modifying net in-place.
  """
  for op in net.operators:
    if op.type == 'Dropout':
      op.type = 'Alias'
      del op.outputs[1]  # output 1 is the dropout mask, which is not needed.
      del op.args[:]  # args is used in Dropout but not needed in Alias.
  return


class CacaRegistry(object):
  registry_ = {}

  @classmethod
  def Register(cls, op_name):
    """A decorator for registering gradient mappings."""
    def Wrapper(func):
      cls.registry_[op_name] = func
      return func
    return Wrapper

  @classmethod
  def TranslateLayer(cls, layer, pretrained_blobs):
    try:
      caffe_ops, params = cls.registry_[layer.type](layer, pretrained_blobs)
    except KeyError as err:
      raise KeyError('No translator registered for layer: %s' % str(layer))
    if caffe_ops is None:
      return []
    if type(caffe_ops) is not list:
      caffe_ops = [caffe_ops]
    return caffe_ops, params

  @classmethod
  def TranslateModel(cls, caffe_net, pretrained_net,
                     net_state=caffe_pb2.NetState()):
    net = caffe2_pb2.NetDef()
    net.name = caffe_net.name
    net_params = caffe2_pb2.TensorProtos()
    if len(caffe_net.layer) == 0:
      raise ValueError('I think something is wrong. This translation script '
                       'only accepts new style layers that are stored in the '
                       'layer field.')
    for layer in caffe_net.layer:
      if not _ShouldInclude(net_state, layer):
        print 'Current net state does not need layer', layer.name
        continue
      print 'Translate layer', layer.name
      # Get pretrained one
      pretrained_layers = (
          [l for l in pretrained_net.layer if l.name == layer.name] +
          [l for l in pretrained_net.layers if l.name == layer.name])
      if len(pretrained_layers) > 1:
        raise ValueError('huh? more than one pretrained layer of one name?')
      elif len(pretrained_layers) == 1:
        pretrained_blobs = [utils.CaffeBlobToNumpyArray(blob)
                            for blob in pretrained_layers[0].blobs]
      else:
        # No pretrained layer for the given layer name. We'll just pass no
        # parameter blobs.
        # print 'No pretrained layer for layer', layer.name
        pretrained_blobs = []
      operators, params = cls.TranslateLayer(layer, pretrained_blobs)
      net.operators.extend(operators)
      net_params.protos.extend(params)
    return net, net_params


def TranslateModel(caffe_net, pretrained_net):
  return CacaRegistry.TranslateModel(caffe_net, pretrained_net)


def BaseTranslate(layer, caffe2_type):
  caffe2_op = caffe2_pb2.OperatorDef()
  caffe2_op.type = caffe2_type
  caffe2_op.inputs.extend(layer.bottom)
  caffe2_op.outputs.extend(layer.top)
  return caffe2_op


def AddArgument(op, key, value):
  """Makes an argument based on the value type."""
  op.args.extend([utils.MakeArgument(key, value)])


################################################################################
# Common translators for layers.
################################################################################

@CacaRegistry.Register("Convolution")
def TranslateConv(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "Conv")
  output = caffe_op.outputs[0]
  caffe_op.inputs.extend([output + '_w', output + '_b'])
  param = layer.convolution_param
  AddArgument(caffe_op, "stride", param.stride)
  AddArgument(caffe_op, "kernel", param.kernel_size)
  AddArgument(caffe_op, "pad", param.pad)
  AddArgument(caffe_op, "order", "NCHW")
  if param.group > 1:
    # Now, if the model is grouped convolution, let's do a backward hack and make
    # things working but in an efficient way by inserting zero parameters.
    n, c, h, w = pretrained_blobs[0].shape
    g = param.group
    og = int(n / g)
    if (og * g != n):
      raise ValueError("This should not happen")
    weight = np.zeros((n, c * g, h, w), dtype=np.float32)
    for i in range(param.group):
      weight[i * og : (i + 1) * og, i * c : (i+1) * c, :, :] = pretrained_blobs[0][i * og : (i + 1) * og]
  else:
    weight = pretrained_blobs[0]
  weight = utils.NumpyArrayToCaffe2Tensor(weight, output + '_w')
  bias = utils.NumpyArrayToCaffe2Tensor(
      pretrained_blobs[1].flatten(), output + '_b')
  # Todo: deal with parameters.
  return caffe_op, [weight, bias]

@CacaRegistry.Register("ReLU")
def TranslateRelu(layer, pretrained_blobs):
  return BaseTranslate(layer, "Relu"), []

@CacaRegistry.Register("Pooling")
def TranslatePool(layer, pretrained_blobs):
  param = layer.pooling_param
  if param.pool == caffe_pb2.PoolingParameter.MAX:
    caffe_op = BaseTranslate(layer, "MaxPool")
    caffe_op.outputs.extend(['_' + caffe_op.outputs[0] + '_maxid'])
  elif param.pool == caffe_pb2.PoolingParameter.AVE:
    caffe_op = BaseTranslate(layer, "AveragePool")
  AddArgument(caffe_op, "stride", int(param.stride))
  AddArgument(caffe_op, "kernel", int(param.kernel_size))
  AddArgument(caffe_op, "pad", int(param.pad))
  AddArgument(caffe_op, "order", "NCHW")
  return caffe_op, []

@CacaRegistry.Register("LRN")
def TranslateLRN(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "LRN")
  caffe_op.outputs.extend(['_' + caffe_op.outputs[0] + '_scale'])
  param = layer.lrn_param
  if param.norm_region != caffe_pb2.LRNParameter.ACROSS_CHANNELS:
    raise ValueError("Does not support norm region other than across channels.")
  AddArgument(caffe_op, "size", int(param.local_size))
  AddArgument(caffe_op, "alpha", float(param.alpha))
  AddArgument(caffe_op, "beta", float(param.beta))
  AddArgument(caffe_op, "bias", float(param.k))
  AddArgument(caffe_op, "order", "NCHW")
  return caffe_op, []

@CacaRegistry.Register("InnerProduct")
def TranslateInnerProduct(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "FC")
  output = caffe_op.outputs[0]
  caffe_op.inputs.extend([output + '_w', output + '_b'])
  weight = utils.NumpyArrayToCaffe2Tensor(
      pretrained_blobs[0][0,0], output + '_w')
  bias = utils.NumpyArrayToCaffe2Tensor(
      pretrained_blobs[1].flatten(), output + '_b')
  return caffe_op, [weight, bias]

@CacaRegistry.Register("Dropout")
def TranslateDropout(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "Dropout")
  caffe_op.outputs.extend(['_' + caffe_op.outputs[0] + '_mask'])
  param = layer.dropout_param
  AddArgument(caffe_op, "ratio", param.dropout_ratio)
  return caffe_op, []


@CacaRegistry.Register("Softmax")
def TranslateSoftmax(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "Softmax")
  return caffe_op, []
