from caffe2.python import core

class CNNModelHelper(object):
  """A helper model so we can write CNN models more easily, without having to
  manually define parameter initializations and operators separately.
  """
  def __init__(self, order="NCHW", name=None):
    if name is None:
      name = "CNN"
    self.net = core.Net(name)
    self.param_init_net = core.Net(name + '_init')
    self.params = []
    self.weights = []
    self.biases = []
    self.order = order
    if self.order != "NHWC" and self.order != "NCHW":
      raise ValueError("Cannot understand the CNN storage order %s."
                       % self.order)

  def Proto(self):
    return self.net.Proto()

  def Conv(self, blob_in, blob_out, dim_in, dim_out, kernel,
           weight_init, bias_init, **kwargs):
    """Convolution. We intentionally do not provide odd kernel/stride/pad
    settings in order to discourage the use of odd cases.
    """
    weight_shape = ([dim_out, dim_in, kernel, kernel] if self.order == "NCHW"
                    else [dim_out, kernel, kernel, dim_in])
    weight = self.param_init_net.__getattr__(weight_init[0])(
        [], blob_out + '_w', shape=weight_shape, **weight_init[1])
    bias = self.param_init_net.__getattr__(bias_init[0])(
        [], blob_out + '_b', shape=[dim_out,], **bias_init[1])
    self.params.extend([weight, bias])
    self.weights.append(weight)
    self.biases.append(bias)
    return self.net.Conv([blob_in, weight, bias], blob_out, kernel=kernel,
                         order=self.order, **kwargs)

  def GroupConv(self, blob_in, blob_out, dim_in, dim_out, kernel,
                weight_init, bias_init, group=1, **kwargs):
    """Convolution. We intentionally do not provide odd kernel/stride/pad
    settings in order to discourage the use of odd cases.
    """
    if dim_in % group:
      raise ValueError("dim_in should be divisible by group.")
    splitted_blobs = self.net.DepthSplit(
        blob_in,
        ['_' + blob_out + '_gconv_split_' + str(i) for i in range(group)],
        dimensions=[dim_in / group for i in range(group)],
        order=self.order)
    weight_shape = ([dim_out / group, dim_in / group, kernel, kernel]
                    if self.order == "NCHW"
                    else [dim_out / group, kernel, kernel, dim_in / group])
    conv_blobs = []
    for i in range(group):
      weight = self.param_init_net.__getattr__(weight_init[0])(
          [], blob_out + '_gconv_%d_w' % i, shape=weight_shape,
          **weight_init[1])
      bias = self.param_init_net.__getattr__(bias_init[0])(
          [], blob_out + '_gconv_%d_b' % i, shape=[dim_out / group],
          **bias_init[1])
      self.params.extend([weight, bias])
      self.weights.append(weight)
      self.biases.append(bias)
      conv_blobs.append(
          splitted_blobs[i].Conv([weight, bias], blob_out + '_gconv_%d' % i,
                                 kernel=kernel, order=self.order, **kwargs))
    concat, concat_dims = self.net.DepthConcat(
          conv_blobs, [blob_out, "_" + blob_out + "_concat_dims"],
          order=self.order)
    return concat

  def FC(self, blob_in, blob_out, dim_in, dim_out, weight_init, bias_init,
         **kwargs):
    """FC"""
    weight = self.param_init_net.__getattr__(weight_init[0])(
        [], blob_out + '_w', shape=[dim_out, dim_in], **weight_init[1])
    bias = self.param_init_net.__getattr__(bias_init[0])(
        [], blob_out + '_b', shape=[dim_out,], **bias_init[1])
    self.params.extend([weight, bias])
    return self.net.FC([blob_in, weight, bias], blob_out, **kwargs)

  def LRN(self, blob_in, blob_out, **kwargs):
    """LRN"""
    return self.net.LRN(blob_in, [blob_out, "_" + blob_out + "_scale"],
                        order=self.order, **kwargs)[0]

  def Dropout(self, blob_in, blob_out, **kwargs):
    """Dropout"""
    return self.net.Dropout(blob_in, [blob_out, "_" + blob_out + "_mask"],
                            **kwargs)[0]

  def MaxPool(self, blob_in, blob_out, **kwargs):
    """Max pooling"""
    return self.net.MaxPool(blob_in, blob_out,
                            order=self.order, **kwargs)
  def AveragePool(self, blob_in, blob_out, **kwargs):
    """Average pooling"""
    return self.net.AveragePool(blob_in, blob_out,
                                order=self.order, **kwargs)

  def DepthConcat(self, blobs_in, blob_out, **kwargs):
    """Depth Concat."""
    return self.net.DepthConcat(blobs_in, [blob_out, "_" + blob_out + "_condat_dims"],
                                order=self.order)[0]

  def Relu(self, blob_in, blob_out, **kwargs):
    """Relu."""
    return self.net.Relu(blob_in, blob_out, order=self.order, **kwargs)

  def AddGradientOperators(self):
    self.net.AddGradientOperators()

  def __getattr__(self, op_type):
    """Catch-all for all other operators, mostly those without params."""
    if not core.IsOperator(op_type):
      raise RuntimeError(
          'Method ' + op_type + ' is not a registered operator.')
    return self.net.__getattr__(op_type)


