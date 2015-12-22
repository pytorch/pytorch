"""
Benchmark for common convnets.

Speed on Titan X, with 10 warmup steps and 10 main steps and with different
versions of cudnn, are as follows (time reported below is per-batch time,
forward / forward+backward):

                    CuDNN V3        CuDNN v4
AlexNet         32.5 / 108.0    27.4 /  90.1
OverFeat       113.0 / 342.3    91.7 / 276.5
Inception      134.5 / 485.8   125.7 / 450.6
VGG (batch 64) 200.8 / 650.0   164.1 / 551.7

Speed on Inception with varied batch sizes and CuDNN v4 is as follows:

Batch Size   Speed per batch     Speed per image
 16             22.8 /  72.7         1.43 / 4.54
 32             38.0 / 127.5         1.19 / 3.98
 64             67.2 / 233.6         1.05 / 3.65
128            125.7 / 450.6         0.98 / 3.52

(Note that these numbers involve a "full" backprop, i.e. the gradient
with respect to the input image is also computed.)

To get the numbers, simply run:

for MODEL in AlexNet OverFeat Inception; do
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py \
    --batch_size 128 --model $MODEL --forward_only True
done
for MODEL in AlexNet OverFeat Inception; do
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py \
    --batch_size 128 --model $MODEL
done
PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py \
  --batch_size 64 --model VGGA --forward_only True
PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py \
  --batch_size 64 --model VGGA

for BS in 16 32 64 128; do
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py \
    --batch_size $BS --model Inception --forward_only True
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py \
    --batch_size $BS --model Inception
done

Note that VGG needs to be run at batch 64 due to memory limit on the backward
pass.
"""

import argparse
import numpy as np
import time

from pycaffe2 import cnn, utils, workspace


def AlexNet(order):
  model = cnn.CNNModelHelper(order, name="alexnet")
  conv1 = model.Conv("data", "conv1", 3, 64, 11,
                    ('XavierFill', {}), ('ConstantFill', {}), stride=4, pad=2)
  relu1 = model.Relu(conv1, "conv1")
  pool1 = model.MaxPool(relu1, "pool1", kernel=3, stride=2)
  conv2 = model.Conv(pool1, "conv2", 64, 192, 5,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=2)
  relu2 = model.Relu(conv2, "conv2")
  pool2 = model.MaxPool(relu2, "pool2", kernel=3, stride=2)
  conv3 = model.Conv(pool2, "conv3", 192, 384, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu3 = model.Relu(conv3, "conv3")
  conv4 = model.Conv(relu3, "conv4", 384, 256, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu4 = model.Relu(conv4, "conv4")
  conv5 = model.Conv(relu4, "conv5", 256, 256, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu5 = model.Relu(conv5, "conv5")
  pool5 = model.MaxPool(relu5, "pool5", kernel=3, stride=2)
  fc6 = model.FC(pool5, "fc6", 256*6*6, 4096,
                    ('XavierFill', {}), ('ConstantFill', {}))
  relu6 = model.Relu(fc6, "fc6")
  fc7 = model.FC(relu6, "fc7", 4096, 4096,
                    ('XavierFill', {}), ('ConstantFill', {}))
  relu7 = model.Relu(fc7, "fc7")
  fc8 = model.FC(relu7, "fc8", 4096, 1000,
                    ('XavierFill', {}), ('ConstantFill', {}))
  pred = model.Softmax(fc8, "pred")
  xent = model.LabelCrossEntropy([pred, "label"], "xent")
  loss = model.AveragedLoss(xent, "loss")
  return model, 224


def OverFeat(order):
  model = cnn.CNNModelHelper(order, name="overfeat")
  conv1 = model.Conv("data", "conv1", 3, 96, 11,
                    ('XavierFill', {}), ('ConstantFill', {}), stride=4)
  relu1 = model.Relu(conv1, "conv1")
  pool1 = model.MaxPool(relu1, "pool1", kernel=2, stride=2)
  conv2 = model.Conv(pool1, "conv2", 96, 256, 5,
                    ('XavierFill', {}), ('ConstantFill', {}))
  relu2 = model.Relu(conv2, "conv2")
  pool2 = model.MaxPool(relu2, "pool2", kernel=2, stride=2)
  conv3 = model.Conv(pool2, "conv3", 256, 512, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu3 = model.Relu(conv3, "conv3")
  conv4 = model.Conv(relu3, "conv4", 512, 1024, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu4 = model.Relu(conv4, "conv4")
  conv5 = model.Conv(relu4, "conv5", 1024, 1024, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu5 = model.Relu(conv5, "conv5")
  pool5 = model.MaxPool(relu5, "pool5", kernel=2, stride=2)
  fc6 = model.FC(pool5, "fc6", 1024*6*6, 3072,
                 ('XavierFill', {}), ('ConstantFill', {}))
  relu6 = model.Relu(fc6, "fc6")
  fc7 = model.FC(relu6, "fc7", 3072, 4096,
                 ('XavierFill', {}), ('ConstantFill', {}))
  relu7 = model.Relu(fc7, "fc7")
  fc8 = model.FC(relu7, "fc8", 4096, 1000,
                    ('XavierFill', {}), ('ConstantFill', {}))
  pred = model.Softmax(fc8, "pred")
  xent = model.LabelCrossEntropy([pred, "label"], "xent")
  loss = model.AveragedLoss(xent, "loss")
  return model, 231


def VGGA(order):
  model = cnn.CNNModelHelper(order, name='vgg-a')
  conv1 = model.Conv("data", "conv1", 3, 64, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu1 = model.Relu(conv1, "conv1")
  pool1 = model.MaxPool(relu1, "pool1", kernel=2, stride=2)
  conv2 = model.Conv(pool1, "conv2", 64, 128, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu2 = model.Relu(conv2, "conv2")
  pool2 = model.MaxPool(relu2, "pool2", kernel=2, stride=2)
  conv3 = model.Conv(pool2, "conv3", 128, 256, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu3 = model.Relu(conv3, "conv3")
  conv4 = model.Conv(relu3, "conv4", 256, 256, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu4 = model.Relu(conv4, "conv4")
  pool4 = model.MaxPool(relu4, "pool4", kernel=2, stride=2)
  conv5 = model.Conv(pool4, "conv5", 256, 512, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu5 = model.Relu(conv5, "conv5")
  conv6 = model.Conv(relu5, "conv6", 512, 512, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu6 = model.Relu(conv6, "conv6")
  pool6 = model.MaxPool(relu6, "pool6", kernel=2, stride=2)
  conv7 = model.Conv(pool6, "conv7", 512, 512, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu7 = model.Relu(conv7, "conv7")
  conv8 = model.Conv(relu7, "conv8", 512, 512, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu8 = model.Relu(conv8, "conv8")
  pool8 = model.MaxPool(relu8, "pool8", kernel=2, stride=2)

  fcix = model.FC(pool8, "fcix", 512*7*7, 4096,
                 ('XavierFill', {}), ('ConstantFill', {}))
  reluix = model.Relu(fcix, "fcix")
  fcx = model.FC(reluix, "fcx", 4096, 4096,
                 ('XavierFill', {}), ('ConstantFill', {}))
  relux = model.Relu(fcx, "fcx")
  fcxi = model.FC(relux, "fcxi", 4096, 1000,
                    ('XavierFill', {}), ('ConstantFill', {}))
  pred = model.Softmax(fcxi, "pred")
  xent = model.LabelCrossEntropy([pred, "label"], "xent")
  loss = model.AveragedLoss(xent, "loss")
  return model, 231


def _InceptionModule(model, input_blob, input_depth, output_name,
                     conv1_depth, conv3_depths, conv5_depths, pool_depth):
  # path 1: 1x1 conv
  conv1 = model.Conv(input_blob, output_name + ":conv1",
                     input_depth, conv1_depth, 1,
                     ('XavierFill', {}), ('ConstantFill', {}))
  conv1 = model.Relu(conv1, conv1)
  # path 2: 1x1 conv + 3x3 conv
  conv3_reduce = model.Conv(input_blob, output_name + ":conv3_reduce",
                            input_depth, conv3_depths[0], 1,
                            ('XavierFill', {}), ('ConstantFill', {}))
  conv3_reduce = model.Relu(conv3_reduce, conv3_reduce)
  conv3 = model.Conv(conv3_reduce, output_name + ":conv3",
                     conv3_depths[0], conv3_depths[1], 3,
                     ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  conv3 = model.Relu(conv3, conv3)
  # path 3: 1x1 conv + 5x5 conv
  conv5_reduce = model.Conv(input_blob, output_name + ":conv5_reduce",
                            input_depth, conv5_depths[0], 1,
                            ('XavierFill', {}), ('ConstantFill', {}))
  conv5_reduce = model.Relu(conv5_reduce, conv5_reduce)
  conv5 = model.Conv(conv5_reduce, output_name + ":conv5",
                     conv5_depths[0], conv5_depths[1], 5,
                     ('XavierFill', {}), ('ConstantFill', {}), pad=2)
  conv5 = model.Relu(conv5, conv5)
  # path 4: pool + 1x1 conv
  pool = model.MaxPool(input_blob, output_name + ":pool",
                       kernel=3, stride=1, pad=1)
  pool_proj = model.Conv(pool, output_name + ":pool_proj",
                         input_depth, pool_depth, 1,
                         ('XavierFill', {}), ('ConstantFill', {}))
  pool_proj = model.Relu(pool_proj, pool_proj)
  output = model.DepthConcat([conv1, conv3, conv5, pool_proj], output_name)
  return output

def Inception(order):
  model = cnn.CNNModelHelper(order, name="inception")
  conv1 = model.Conv("data", "conv1", 3, 64, 7,
                    ('XavierFill', {}), ('ConstantFill', {}), stride=2, pad=3)
  relu1 = model.Relu(conv1, "conv1")
  pool1 = model.MaxPool(relu1, "pool1", kernel=3, stride=2, pad=1)
  conv2a = model.Conv(pool1, "conv2a", 64, 64, 1,
                    ('XavierFill', {}), ('ConstantFill', {}))
  conv2a = model.Relu(conv2a, conv2a)
  conv2 = model.Conv(conv2a, "conv2", 64, 192, 3,
                    ('XavierFill', {}), ('ConstantFill', {}), pad=1)
  relu2 = model.Relu(conv2, "conv2")
  pool2 = model.MaxPool(relu2, "pool2", kernel=3, stride=2, pad=1)
  # Inception modules
  inc3 = _InceptionModule(model, pool2, 192, "inc3",
                          64, [96, 128], [16, 32], 32)
  inc4 = _InceptionModule(model, inc3, 256, "inc4",
                          128, [128, 192], [32, 96], 64)
  pool5 = model.MaxPool(inc4, "pool5", kernel=3, stride=2, pad=1)
  inc5 = _InceptionModule(model, pool5, 480, "inc5",
                          192, [96, 208], [16, 48], 64)
  inc6 = _InceptionModule(model, inc5, 512, "inc6",
                          160, [112, 224], [24, 64], 64)
  inc7 = _InceptionModule(model, inc6, 512, "inc7",
                          128, [128, 256], [24, 64], 64)
  inc8 = _InceptionModule(model, inc7, 512, "inc8",
                          112, [144, 288], [32, 64], 64)
  inc9 = _InceptionModule(model, inc8, 528, "inc9",
                          256, [160, 320], [32, 128], 128)
  pool9 = model.MaxPool(inc9, "pool9", kernel=3, stride=2, pad=1)
  inc10 = _InceptionModule(model, pool9, 832, "inc10",
                           256, [160, 320], [32, 128], 128)
  inc11 = _InceptionModule(model, inc10, 832, "inc11",
                           384, [192, 384], [48, 128], 128)
  pool11 = model.AveragePool(inc11, "pool11", kernel=7, stride=1)
  fc = model.FC(pool11, "fc", 1024, 1000,
                ('XavierFill', {}), ('ConstantFill', {}))
  # It seems that Soumith's benchmark does not have softmax on top
  # for Inception. We will add it anyway so we can have a proper
  # backward pass.
  pred = model.Softmax(fc, "pred")
  xent = model.LabelCrossEntropy([pred, "label"], "xent")
  loss = model.AveragedLoss(xent, "loss")
  return model, 224


def Benchmark(model_gen, arg):
  model, input_size = model_gen(arg.order)

  # In order to be able to run everything without feeding more stuff, let's
  # add the data and label blobs to the parameter initialization net as well.
  if arg.order == "NCHW":
    input_shape = [arg.batch_size, 3, input_size, input_size]
  else:
    input_shape = [arg.batch_size, input_size, input_size, 3]
  model.param_init_net.GaussianFill([], "data", shape=input_shape,
                                    mean=0.0, std=1.0)
  model.param_init_net.UniformIntFill([], "label", shape=[arg.batch_size,],
                                      min=0, max=999)

  # Note: even when we are running things on CPU, adding a few engine related
  # argument will not hurt since the CPU operator registy will simply ignore
  # these options and go the default path.
  for op in model.net.Proto().op:
    if op.type == 'Conv':
      op.engine = 'CUDNN'
      #op.arg.add().CopyFrom(utils.MakeArgument('ws_nbytes_limit', arg.cudnn_limit))
      op.arg.add().CopyFrom(utils.MakeArgument('exhaustive_search', 1))
      op.arg.add().CopyFrom(utils.MakeArgument('shared_ws_name', 'cudnn_workspace'))
    elif op.type in ['MaxPool', 'AveragePool', 'Relu', 'Softmax']:
      op.engine = 'CUDNN'
  if arg.forward_only:
    print arg.model, ': running forward only.'
  else:
    print arg.model, ': running forward-backward.'
    model.AddGradientOperators()
    if arg.order == 'NHWC':
      print ('==WARNING==\n'
             'NHWC order with CuDNN may not be supported yet, so I might\n'
             'exit suddenly.')

  if not arg.cpu:
    model.param_init_net.RunAllOnGPU()
    model.net.RunAllOnGPU()

  workspace.RunNetOnce(model.param_init_net)
  workspace.CreateNet(model.net)
  for i in range(arg.warmup_iterations):
    workspace.RunNet(model.net.Proto().name)

  start = time.time()
  for i in range(arg.iterations):
    workspace.RunNet(model.net.Proto().name)
  print 'Spent: ', (time.time() - start) / arg.iterations
  if arg.layer_wise_benchmark:
    print 'Layer-wise benchmark.'
    workspace.BenchmarkNet(
        model.net.Proto().name, 1, arg.iterations, True)
  # Writes out the pbtxt for benchmarks on e.g. Android
  with open("{0}_init_batch_{1}.pbtxt".format(arg.model, arg.batch_size),
            "w") as fid:
    fid.write(str(model.param_init_net.Proto()))
  with open("{0}.pbtxt".format(arg.model, arg.batch_size),
            "w") as fid:
    fid.write(str(model.net.Proto()))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Caffe2 benchmark.")
  parser.add_argument("--batch_size", type=int, default=128,
                      help="The batch size.")
  parser.add_argument("--model", type=str,
                      help="The model to benchmark.")
  parser.add_argument("--order", type=str, default="NCHW",
                      help="The order to evaluate.")
  parser.add_argument("--cudnn_ws", type=int, default=-1,
                      help="The cudnn workspace size.")
  parser.add_argument("--iterations", type=int, default=10,
                      help="Number of iterations to run the network.")
  parser.add_argument("--warmup_iterations", type=int, default=10,
                      help="Number of warm-up iterations before benchmarking.")
  parser.add_argument("--forward_only", type=bool, default=False,
                      help="If set, only run the forward pass.")
  parser.add_argument("--layer_wise_benchmark", type=bool, default=False,
                      help="If True, run the layer-wise benchmark as well.")
  parser.add_argument("--cpu", type=bool, default=False,
                      help="If True, run testing on CPU instead of GPU.")
  args = parser.parse_args()
  if (not args.batch_size or not args.model or not args.order or not args.cudnn_ws):
    parser.print_help()

  workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
  model_map = {
      'AlexNet': AlexNet,
      'OverFeat': OverFeat,
      'VGGA': VGGA,
      'Inception': Inception
      }
  Benchmark(model_map[args.model], args)
