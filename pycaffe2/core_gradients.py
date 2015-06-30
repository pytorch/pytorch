from caffe2.proto import caffe2_pb2
from pycaffe2.core import *  # I know, I know... will fix later

@GradientRegistry.RegisterGradient('FC')
def AddFCGradient(op):
  return CreateOperator('FCGradient')(
      list(op.inputs) + [GetGradientName(op.outputs[0])],
      [GetGradientName(name) for name in
          [op.inputs[1], op.inputs[2], op.inputs[0]]])

@GradientRegistry.RegisterGradient('SquaredL2Distance')
def AddSquaredL2DistanceGradient(op):
  return CreateOperator('SquaredL2DistanceGradient')(
      list(op.inputs) + [GetGradientName(op.outputs[0])],
      [GetGradientName(name) for name in op.inputs])

@GradientRegistry.RegisterGradient("LabelCrossEntropy")
def AddLabelCrossEntropyGradient(op):
  return CreateOperator('LabelCrossEntropyGradient')(
    list(op.inputs) + [GetGradientName(op.outputs[0])],
    [GetGradientName(op.inputs[0])])

@GradientRegistry.RegisterGradient("Softmax")
def AddSoftmaxGradient(op):
  return CreateOperator('SoftmaxGradient')(
    [op.outputs[0], GetGradientName(op.outputs[0])],
    [GetGradientName(op.inputs[0])])

@GradientRegistry.RegisterGradient("Flatten")
def AddFlattenGradient(op):
  return CreateOperator('ReshapeLike')(
    [GetGradientName(op.outputs[0]), op.inputs[0]],
    [GetGradientName(op.inputs[0])])

@GradientRegistry.RegisterGradient("AveragedLoss")
def CheckAveragedLossNaming(op):
  if op.outputs[1] != GetGradientName(op.inputs[0]):
    raise ValueError(
        "AveragedLoss output[1] should be named as the gradient of input[0]. "
        "Please name your output[1] to %s.", GetGradientName(op.inputs[0]))
  return


@GradientRegistry.RegisterGradient("TensorProtosDBInput")
@GradientRegistry.RegisterGradient("GaussianFill")
def NoGradientToCompute(op):
  return

@GradientRegistry.RegisterGradient("Accuracy")
@GradientRegistry.RegisterGradient("Print")
def UtilityOperatorsShouldNotBeAddedBeforeGradients(op):
  raise RuntimeError("Utility operators should be added after you add "
                     "gradient operators to a net.")


@GradientRegistry.RegisterGradient("Relu")
def AddReluGradient(op):
  return CreateOperator("ReluGradient")(
      [op.inputs[0], GetGradientName(op.outputs[0])],
      [GetGradientName(op.inputs[0])])

@GradientRegistry.RegisterGradient("MaxPool")
def AddMaxPoolGradient(op):
  return CreateOperator("MaxPoolGradient")(
      [op.inputs[0], GetGradientName(op.outputs[0]), op.outputs[1]],
      [GetGradientName(op.inputs[0])], args=op.args)


@GradientRegistry.RegisterGradient("AveragePool")
def AddAveragePoolGradient(op):
  return CreateOperator("AveragePoolGradient")(
      [op.inputs[0], GetGradientName(op.outputs[0])],
      [GetGradientName(op.inputs[0])], args=op.args)

@GradientRegistry.RegisterGradient('Conv')
def AddFCGradient(op):
  return CreateOperator('ConvGradient')(
      list(op.inputs) + [GetGradientName(op.outputs[0])],
      [GetGradientName(name) for name in
          [op.inputs[1], op.inputs[2], op.inputs[0]]],
      args=op.args)


@GradientRegistry.RegisterGradient('DepthSplit')
def AddDepthSplitGradient(op):
  return CreateOperator('DepthConcat')(
      [GetGradientName(name) for name in op.outputs],
      [GetGradientName(op.inputs[0]), '_' + GetGradientName(op.inputs[0]) + '_dims'],
      args = op.args)

@GradientRegistry.RegisterGradient('DepthConcat')
def AddDepthConcatGradient(op):
  return CreateOperator('DepthSplit')(
      [GetGradientName(op.outputs[0]), op.outputs[1]],
      [GetGradientName(name) for name in op.inputs],
      args = op.args)

@GradientRegistry.RegisterGradient('Dropout')
def AddDropoutGradient(op):
  return CreateOperator('DropoutGrad')(
      [GetGradientName(op.outputs[0]), op.outputs[1]],
      [GetGradientName(op.inputs[0])],
      args = op.args)

@GradientRegistry.RegisterGradient('LRN')
def AddLRNGradient(op):
  return CreateOperator('LRNGradient')(
      [op.inputs[0], op.outputs[0], op.outputs[1],
       GetGradientName(op.outputs[0])],
      [GetGradientName(op.inputs[0])],
      args = op.args)

@GradientRegistry.RegisterGradient('Split')
def AddSplitGradient(op):
  return CreateOperator('Sum')(
    [GetGradientName(name) for name in op.outputs],
    [GetGradientName(op.inputs[0])])

@GradientRegistry.RegisterGradient('Alias')
def AddAliasGradient(op):
  return CreateOperator('Alias')(
    [GetGradientName(name) for name in op.outputs],
    [GetGradientName(name) for name in op.inputs])