from pycaffe2 import core
from pycaffe2 import core_gradients

init_net = core.Net("init")
filter1 = init_net.XavierFill([], "filter1", shape=[20, 5, 5, 1])
bias1 = init_net.ConstantFill([], "bias1", shape=[20,], value=0.0)
filter2 = init_net.XavierFill([], "filter2", shape=[50, 5, 5, 20])
bias2 = init_net.ConstantFill([], "bias2", shape=[50,], value=0.0)
W3 = init_net.XavierFill([], "W3", shape=[500, 800])
B3 = init_net.ConstantFill([], "B3", shape=[500], value=0.0)
W4 = init_net.XavierFill([], "W4", shape=[10, 500])
B4 = init_net.ConstantFill([], "B4", shape=[10], value=0.0)

LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
DECAY = init_net.ConstantFill([], "DECAY", shape=[1], value=0.999)

train_net = core.Net("train")
data, label = train_net.TensorProtosDBInput(
    [], ["data", "label"], batch_size=64,
    db="gen/data/mnist/mnist-train-nhwc-minidb", db_type="minidb")

# For an unnamed version, do the following:
pool1 = (data.Conv([filter1, bias1], kernel=5, pad=0, stride=1, order="NHWC")
             .MaxPool([], kernel=2, stride=2, order="NHWC"))
pool2 = (pool1.Conv([filter2, bias2], kernel=5, pad=0, stride=1, order="NHWC")
              .MaxPool([], kernel=2, stride=2, order="NHWC"))
softmax = pool2.Flatten().FC([W3, B3]).Relu().FC([W4, B4]).Softmax()

# Cross entropy, and accuracy
xent = softmax.LabelCrossEntropy([label], "xent")
# The loss function.
loss = xent.AveragedLoss([], ["loss"])
# Get gradient, skipping the input and flatten layers.
train_net.AddGradientOperators(skip=1)
accuracy = softmax.Accuracy([label], "accuracy")
# parameter update.
for param in [filter1, bias1, filter2, bias2, W3, B3, W4, B4]:
  train_net.WeightedSum([param, ONE, param.Grad(), LR], param)
LR = train_net.Mul([LR, DECAY], "LR")
train_net.Print([accuracy], [])
train_net._net.net_type = 'dag'
train_net._net.num_workers = 8

# CPU version
plan = core.Plan("mnist_lenet")
plan.AddNets([init_net, train_net])
plan.AddStep(core.ExecutionStep("init", init_net))
plan.AddStep(core.ExecutionStep("train", train_net, 1000))
with open('mnist_lenet_nhwc.pbtxt', 'w') as fid:
  fid.write(str(plan.Proto()))

# GPU version
init_net.RunAllOnGPU()
train_net.RunAllOnGPU()
plan = core.Plan("mnist_lenet")
plan.AddNets([init_net, train_net])
plan.AddStep(core.ExecutionStep("init", init_net))
plan.AddStep(core.ExecutionStep("train", train_net, 1000))
with open('mnist_lenet_nhwc_gpu.pbtxt', 'w') as fid:
  fid.write(str(plan.Proto()))

