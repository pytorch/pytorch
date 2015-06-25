from caffe2.proto import caffe2_pb2
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from pycaffe2 import utils

MODE_TRAIN = 0
MODE_TEST = 1
__TRANSLATE_MODE__ = MODE_TRAIN

def SetTranslateMode(mode):
  global __TRANSLATE_MODE__
  __TRANSLATE_MODE__ = mode

def IsTraining():
  return (__TRANSLATE_MODE__ == MODE_TRAIN)

def IsTesting():
  return (__TRANSLATE_MODE__ == MODE_TEST)


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
  def TranslateModel(cls, caffe_net, pretrained_net):
    net = caffe2_pb2.NetDef()
    net.name = caffe_net.name
    net_params = []
    if len(caffe_net.layer) == 0:
      raise ValueError('I think something is wrong. This translation script '
                       'only accepts new style layers that are stored in the '
                       'layer field.')
    for layer in caffe_net.layer:
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
      net_params.extend(params)
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
  if IsTraining():
    caffe_op = BaseTranslate(layer, "Dropout")
    caffe_op.outputs.extend(['_' + caffe_op.outputs[0] + '_mask'])
    param = layer.dropout_param
    AddArgument(caffe_op, "ratio", param.dropout_ratio)
    return caffe_op, []
  else:
    return BaseTranslate(layer, "Alias"), []


@CacaRegistry.Register("Softmax")
def TranslateSoftmax(layer, pretrained_blobs):
  caffe_op = BaseTranslate(layer, "Softmax")
  return caffe_op, []
