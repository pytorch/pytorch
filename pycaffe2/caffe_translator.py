from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from pycaffe2 import core, utils


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
  for op in net.op:
    if op.type == 'Dropout':
      op.type = 'Alias'
      del op.output[1]  # output 1 is the dropout mask, which is not needed.
      del op.arg[:]  # args is used in Dropout but not needed in Alias.
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
      raise KeyError('No translator registered for layer: %s yet.' % str(layer))
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
      net.op.extend(operators)
      net_params.protos.extend(params)
    return net, net_params


def TranslateModel(caffe_net, pretrained_net):
  return CacaRegistry.TranslateModel(caffe_net, pretrained_net)


def BaseTranslate(layer, caffe2_type):
  caffe2_op = caffe2_pb2.OperatorDef()
  caffe2_op.type = caffe2_type
  caffe2_op.input.extend(layer.bottom)
  caffe2_op.output.extend(layer.top)
  return caffe2_op


def AddArgument(op, key, value):
  """Makes an argument based on the value type."""
  op.arg.extend([utils.MakeArgument(key, value)])


################################################################################
# Common translators for layers.
################################################################################

@CacaRegistry.Register("Convolution")
def TranslateConv(layer, pretrained_blobs):
  param = layer.convolution_param
  if param.group > 1:
    return TranslateConvWithGroups(layer, pretrained_blobs)
  # If there is no odd things, we will basically translate it to a standard
  # caffe2 op.
  caffe_op = BaseTranslate(layer, "Conv")
  output = caffe_op.output[0]
  caffe_op.input.extend([output + '_w', output + '_b'])
  AddArgument(caffe_op, "stride", param.stride)
  AddArgument(caffe_op, "kernel", param.kernel_size)
  AddArgument(caffe_op, "pad", param.pad)
  AddArgument(caffe_op, "order", "NCHW")
  weight = utils.NumpyArrayToCaffe2Tensor(pretrained_blobs[0], output + '_w')
  bias = utils.NumpyArrayToCaffe2Tensor(
      pretrained_blobs[1].flatten(), output + '_b')
  return caffe_op, [weight, bias]

def TranslateConvWithGroups(layer, pretrained_blobs):
  print ("Legacy warning: convolution with groups seem to be less and less " +
         "popular, so we no longer have it as a first-class citizen op. " +
         "Instead, we will simulate it with depth split followed by conv " +
         "followed by depth concat.")
  caffe_ops = []
  caffe_params = []
  param = layer.convolution_param
  weight, bias = pretrained_blobs
  bias = bias.flatten()
  n, c, h, w = weight.shape
  g = param.group  # group
  od = int(n / g)  # output dimension
  if (od * g != n):
    # This should not happen: n should always be divisible by g.
    raise ValueError("This should not happen.")
  output = layer.top[0]
  # first, depth_split
  depth_split_op = core.CreateOperator("DepthSplit")(
      layer.bottom[0],
      ['_' + output + '_gconv_split_' + str(i) for i in range(g)],
      dimensions=[c for i in range(g)],
      order="NCHW")
  caffe_ops.append(depth_split_op)
  # second, convolutions
  for i in range(g):
    # convolution layer i
    this_weight = utils.NumpyArrayToCaffe2Tensor(
        weight[i * od : (i + 1) * od], output + '_gconv_' + str(i) + '_w')
    this_bias = utils.NumpyArrayToCaffe2Tensor(
        bias[i * od : (i + 1) * od], output + '_gconv_' + str(i) + '_b')
    conv_op = core.CreateOperator("Conv")(
        [depth_split_op.output[i], this_weight.name, this_bias.name],
        ['_' + output + '_gconv_conv_' + str(i)],
        stride=param.stride,
        kernel=param.kernel_size,
        pad=param.pad,
        order="NCHW")
    caffe_ops.append(conv_op)
    caffe_params.extend([this_weight, this_bias])
  # third, depth concat
  depth_concat_op = core.CreateOperator("DepthConcat")(
      ['_' + output + '_gconv_conv_' + str(i) for i in range(g)],
      [output, '_' + output + '_gconv_concat_dims'],
      order="NCHW")
  caffe_ops.append(depth_concat_op)
  return caffe_ops, caffe_params


@CacaRegistry.Register("ReLU")
def TranslateRelu(layer, pretrained_blobs):
  return BaseTranslate(layer, "Relu"), []

@CacaRegistry.Register("Pooling")
def TranslatePool(layer, pretrained_blobs):
  param = layer.pooling_param
  if param.pool == caffe_pb2.PoolingParameter.MAX:
    caffe_op = BaseTranslate(layer, "MaxPool")
    caffe_op.output.extend(['_' + caffe_op.output[0] + '_maxid'])
  elif param.pool == caffe_pb2.PoolingParameter.AVE:
    caffe_op = BaseTranslate(layer, "AveragePool")
  AddArgument(caffe_op, "stride", int(param.stride))
  AddArgument(caffe_op, "kernel", int(param.kernel_size))
  AddArgument(caffe_op, "pad", int(param.pad))
  AddArgument(caffe_op, "order", "NCHW")
  AddArgument(caffe_op, "legacy_pad", caffe2_legacy_pb2.CAFFE_LEGACY_POOLING)
  return caffe_op, []

@CacaRegistry.Register("LRN")
def TranslateLRN(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "LRN")
  caffe_op.output.extend(['_' + caffe_op.output[0] + '_scale'])
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
  output = caffe_op.output[0]
  caffe_op.input.extend([output + '_w', output + '_b'])
  weight = utils.NumpyArrayToCaffe2Tensor(
      pretrained_blobs[0][0,0], output + '_w')
  bias = utils.NumpyArrayToCaffe2Tensor(
      pretrained_blobs[1].flatten(), output + '_b')
  return caffe_op, [weight, bias]

@CacaRegistry.Register("Dropout")
def TranslateDropout(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "Dropout")
  caffe_op.output.extend(['_' + caffe_op.output[0] + '_mask'])
  param = layer.dropout_param
  AddArgument(caffe_op, "ratio", param.dropout_ratio)
  return caffe_op, []


@CacaRegistry.Register("Softmax")
def TranslateSoftmax(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "Softmax")
  return caffe_op, []

@CacaRegistry.Register("Concat")
def TranslateConcat(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "DepthConcat")
  caffe_op.output.extend(['_' + caffe_op.output[0] + '_dims'])
  AddArgument(caffe_op, "order", "NCHW")
  return caffe_op, []