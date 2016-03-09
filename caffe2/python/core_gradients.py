from caffe2.proto import caffe2_pb2
from caffe2.python.core import *  # I know, I know... will fix later

"""
@GradientRegistry.RegisterGradient('FC')
def AddFCGradient(op):
  return CreateOperator('FCGradient')(
      [op.input[0], op.input[1], GetGradientName(op.output[0])],
      [GetGradientName(name) for name in
          [op.input[1], op.input[2], op.input[0]]])

@GradientRegistry.RegisterGradient('SquaredL2Distance')
def AddSquaredL2DistanceGradient(op):
  return CreateOperator('SquaredL2DistanceGradient')(
      list(op.input) + [GetGradientName(op.output[0])],
      [GetGradientName(name) for name in op.input])
"""

"""
@GradientRegistry.RegisterGradient("LabelCrossEntropy")
def AddLabelCrossEntropyGradient(op):
  return CreateOperator('LabelCrossEntropyGradient')(
    list(op.input) + [GetGradientName(op.output[0])],
    [GetGradientName(op.input[0])])
"""

"""
@GradientRegistry.RegisterGradient("Softmax")
def AddSoftmaxGradient(op):
  return CreateOperator('SoftmaxGradient')(
    [op.output[0], GetGradientName(op.output[0])],
    [GetGradientName(op.input[0])])
"""

"""
@GradientRegistry.RegisterGradient("Flatten")
def AddFlattenGradient(op):
  return CreateOperator('ReshapeLike')(
    [GetGradientName(op.output[0]), op.input[0]],
    [GetGradientName(op.input[0])])
"""

"""
@GradientRegistry.RegisterGradient("AveragedLoss")
def CheckAveragedLossNaming(op):
  return CreateOperator('AveragedLossGradient')(
    [op.input[0]], [GetGradientName(op.input[0])])

@GradientRegistry.RegisterGradient("WeightedSumLoss")
def CheckAveragedLossNaming(op):
  return CreateOperator('WeightedSumLossGradient')(
    [op.input[1]], [GetGradientName(op.input[0])])
"""

"""
@GradientRegistry.RegisterGradient("TensorProtosDBInput")
@GradientRegistry.RegisterGradient("GaussianFill")
@GradientRegistry.RegisterGradient("Iter")
def NoGradientToCompute(op):
  return
"""
"""
@GradientRegistry.RegisterGradient("Accuracy")
@GradientRegistry.RegisterGradient("Print")
def UtilityOperatorsShouldNotBeAddedBeforeGradients(op):
  raise RuntimeError("Utility operators should be added after you add "
                     "gradient operators to a net.")
"""

"""
@GradientRegistry.RegisterGradient("Relu")
def AddReluGradient(op):
  return CreateOperator("ReluGradient")(
      [op.output[0], GetGradientName(op.output[0])],
      [GetGradientName(op.input[0])])
"""
"""
@GradientRegistry.RegisterGradient("Clip")
def AddReluGradient(op):
  return CreateOperator("ClipGradient")(
      [op.output[0], GetGradientName(op.output[0])],
      [GetGradientName(op.input[0])], arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient("MaxPool")
def AddMaxPoolGradient(op):
  return CreateOperator("MaxPoolGradient")(
      [op.input[0], GetGradientName(op.output[0]), op.output[1]],
      [GetGradientName(op.input[0])], arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient("AveragePool")
def AddAveragePoolGradient(op):
  return CreateOperator("AveragePoolGradient")(
      [op.input[0], GetGradientName(op.output[0])],
      [GetGradientName(op.input[0])], arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient('Conv')
def AddConvGradient(op):
  return CreateOperator('ConvGradient')(
      [op.input[0], op.input[1], GetGradientName(op.output[0])],
      [GetGradientName(name) for name in
          [op.input[1], op.input[2], op.input[0]]],
      arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient('DepthSplit')
def AddDepthSplitGradient(op):
  return CreateOperator('DepthConcat')(
      [GetGradientName(name) for name in op.output],
      [GetGradientName(op.input[0]), '_' + GetGradientName(op.input[0]) + '_dims'],
      arg=op.arg)

@GradientRegistry.RegisterGradient('DepthConcat')
def AddDepthConcatGradient(op):
  return CreateOperator('DepthSplit')(
      [GetGradientName(op.output[0]), op.output[1]],
      [GetGradientName(name) for name in op.input],
      arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient('Dropout')
def AddDropoutGradient(op):
  return CreateOperator('DropoutGrad')(
      [GetGradientName(op.output[0]), op.output[1]],
      [GetGradientName(op.input[0])],
      arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient('LRN')
def AddLRNGradient(op):
  return CreateOperator('LRNGradient')(
      [op.input[0], op.output[0], op.output[1],
       GetGradientName(op.output[0])],
      [GetGradientName(op.input[0])],
      arg=op.arg)
"""

"""
@GradientRegistry.RegisterGradient('Split')
def AddSplitGradient(op):
  return CreateOperator('Sum')(
    [GetGradientName(name) for name in op.output],
    [GetGradientName(op.input[0])])
"""
"""
@GradientRegistry.RegisterGradient('Alias')
def AddAliasGradient(op):
  return CreateOperator('Alias')(
    [GetGradientName(name) for name in op.output],
    [GetGradientName(name) for name in op.input])
"""
