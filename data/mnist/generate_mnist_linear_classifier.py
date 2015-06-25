from pycaffe2 import core
from pycaffe2 import core_gradients

init_net = core.Net("init")
W = init_net.UniformFill([], "W", shape=[10, 784], min=-0.1, max=0.1)
B = init_net.ConstantFill([], "B", shape=[10], value=0.0)
LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
DECAY = init_net.ConstantFill([], "DECAY", shape=[1], value=0.999)

train_net = core.Net("train")
data, label = train_net.TensorProtosDBInput(
    [], ["data", "label"], batch_size=64,
    db="gen/data/mnist/mnist-train-minidb", db_type="minidb")
softmax = data.Flatten([], "data_flatten").FC([W, B], "pred").Softmax([], "softmax")
xent = softmax.LabelCrossEntropy([label], "xent")
loss, xent_grad = xent.AveragedLoss([], ["loss", xent.Grad()])
# Get gradient
train_net.AddGradientOperators()
accuracy = softmax.Accuracy([label], "accuracy")
# parameter update.
W = train_net.WeightedSum([W, ONE, "W_grad", LR], "W")
B = train_net.WeightedSum([B, ONE, "B_grad", LR], "B")
LR = train_net.Mul([LR, DECAY], "LR")
train_net.Print([loss, accuracy, LR], [])

# Run all on GPU.
# init_net.RunAllOnGPU()
# train_net.RunAllOnGPU()

plan = core.Plan("mnist_train")
plan.AddNets([init_net, train_net])
plan.AddStep(core.ExecutionStep("init", init_net))
plan.AddStep(core.ExecutionStep("train", train_net, 1000))

with open('linear_classifier_plan.pbtxt', 'w') as fid:
  fid.write(str(plan.Proto()))