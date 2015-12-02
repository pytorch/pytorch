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

  fcix = model.FC(pool8, "fcix", 512*7*7, 3072,
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





def Benchmark(model_gen, order, batch_size, cudnn_limit, forward_only,
              iterations):
  model, input_size = model_gen(order)
  for op in model.net._net.op:
    if op.type == 'Conv':
      op.engine = 'CUDNN'
      op.arg.add().CopyFrom(utils.MakeArgument('ws_nbytes_limit', cudnn_limit))
      op.arg.add().CopyFrom(utils.MakeArgument('shared_ws_name', 'cudnn_workspace'))
  if forward_only:
    print 'Running forward only.'
  else:
    print 'Running forward-backward.'
    model.AddGradientOperators()
    if order == 'NHWC':
      print ('==WARNING==\n'
             'NHWC order with CuDNN may not be supported yet, so I might\n'
             'exit suddenly.')
  model.param_init_net.RunAllOnGPU()
  model.net.RunAllOnGPU()

  workspace.ResetWorkspace()
  if order == 'NCHW':
    data_shape = (batch_size, 3, input_size, input_size)
  else:
    data_shape = (batch_size, input_size, input_size, 3)
  device_option = model.net.Proto().device_option
  workspace.FeedBlob("data", np.random.randn(*data_shape).astype(np.float32),
                     device_option)
  workspace.FeedBlob("label", np.asarray(range(batch_size)).astype(np.int32),
                     device_option)

  workspace.RunNetOnce(model.param_init_net)
  workspace.CreateNet(model.net)
  workspace.RunNet(model.net.Proto().name)
  start = time.time()
  for i in range(iterations):
    workspace.RunNet(model.net.Proto().name)
  print 'Spent: ', (time.time() - start) / iterations


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Caffe2 benchmark.")
  parser.add_argument("--batch_size", type=int, help="The batch size.")
  parser.add_argument("--model", type=str, help="The model to benchmark.")
  parser.add_argument("--order", type=str, help="The order to evaluate.")
  parser.add_argument("--cudnn_ws", type=int, help="The cudnn workspace size.")
  parser.add_argument("--iterations", type=int, default=100,
                      help="Number of iterations to run the network.")
  parser.add_argument("--forward_only", type=bool, default=False,
                      help="If set, only run the forward pass.")
  args = parser.parse_args()
  if (not args.batch_size or not args.model or not args.order or not args.cudnn_ws):
    parser.print_help()

  workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
  model_map = {'AlexNet': AlexNet, 'OverFeat': OverFeat, 'VGGA': VGGA}
  Benchmark(model_map[args.model], args.order, args.batch_size, args.cudnn_ws,
            args.forward_only, args.iterations)
