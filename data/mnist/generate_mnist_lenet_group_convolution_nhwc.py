from pycaffe2 import core
from pycaffe2 import core_gradients

init_net = core.Net("init")
filter1a = init_net.XavierFill([], "filter1a", shape=[10, 5, 5, 1])
bias1a = init_net.ConstantFill([], "bias1a", shape=[10,], value=0.0)
filter1b = init_net.XavierFill([], "filter1b", shape=[10, 5, 5, 1])
bias1b = init_net.ConstantFill([], "bias1b", shape=[10,], value=0.0)
filter2 = init_net.XavierFill([], "filter2", shape=[50, 5, 5, 20])
bias2 = init_net.ConstantFill([], "bias2", shape=[50,], value=0.0)
W3 = init_net.XavierFill([], "W3", shape=[500, 800])
B3 = init_net.ConstantFill([], "B3", shape=[500], value=0.0)
W4 = init_net.XavierFill([], "W4", shape=[10, 500])
B4 = init_net.ConstantFill([], "B4", shape=[10], value=0.0)

params = [filter1a, bias1a, filter1b, bias1b, filter2, bias2, W3, B3, W4, B4]

LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
DECAY = init_net.ConstantFill([], "DECAY", shape=[1], value=0.999)

train_net = core.Net("train")
data, label = train_net.TensorProtosDBInput(
    [], ["data", "label"], batch_size=64,
    db="gen/data/mnist/mnist-train-nhwc-minidb", db_type="minidb")

pool1a, _ = (data.Conv([filter1a, bias1a], kernel=5, pad=0, stride=1, order="NHWC")
                .MaxPool(outputs=2, kernel=2, stride=2, order="NHWC"))
pool1b, _ = (data.Conv([filter1b, bias1b], kernel=5, pad=0, stride=1, order="NHWC")
                .MaxPool(outputs=2, kernel=2, stride=2, order="NHWC"))
pool1, _ = pool1a.DepthConcat([pool1b], outputs=2, order="NHWC", channels=[10, 10])
pool2, _ = (pool1.Conv([filter2, bias2], kernel=5, pad=0, stride=1, order="NHWC")
                 .MaxPool(outputs=2, kernel=2, stride=2, order="NHWC"))
softmax = pool2.Flatten().FC([W3, B3]).Relu().FC([W4, B4]).Softmax()

# Cross entropy, and accuracy
xent = softmax.LabelCrossEntropy([label], "xent")
# The loss function.
loss = xent.AveragedLoss([], ["loss"])
# Get gradient
train_net.AddGradientOperators()
accuracy = softmax.Accuracy([label], "accuracy")
# parameter update.
for param in params:
  train_net.WeightedSum([param, ONE, param.Grad(), LR], param)
LR = train_net.Mul([LR, DECAY], "LR")
train_net.Print([accuracy], [])

# Run all on GPU.
#init_net.RunAllOnGPU()
#train_net.RunAllOnGPU()

plan = core.Plan("mnist_lenet_gc")
plan.AddNets([init_net, train_net])
plan.AddStep(core.ExecutionStep("init", init_net))
plan.AddStep(core.ExecutionStep("train", train_net, 1000))

with open('mnist_lenet_group_convolution_nhwc.pbtxt', 'w') as fid:
  fid.write(str(plan.Proto()))
