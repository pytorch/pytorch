# This prototxt shows how we do a linear regression. The data X is generated
# from a random 2D Gaussian distribution, and the target Y is computed as
#     y = 2.0 * x[0] + 1.5 * x[1] + 0.5
# What we do here is to simply run a SGD without momentum, and a fixed learning
# rate. After 100 iterations, the weight and bias should reach a
# Note(Yangqing): This protobuffer txt is written when there is no python
# scripts yet, so some formats might be different from the (yet to be written)
# script format.

# Note(Yangqing): Since we are minimizing the loss, it becomes odd that we are
# using a negative learning rate - this definitely does not sound very intuitive
# so maybe we want to revisit it in the future.

# Note(Yangqing): The training network contains a loopy blob reference: the
# parameters are being referred to during computation, and then being written
# to during parameter update. I haven't got a good thought on how dependency
# should be figured out yet.

from pycaffe2 import core
from pycaffe2 import core_gradients

init_net = core.Net("init")
W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
W_gt = init_net.GivenTensorFill([], "W_gt", shape=[1, 2], values=[2.0, 1.5])
B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)

train_net = core.Net("train")
X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0)
Y_gt = X.FC([W_gt, B_gt], "Y_gt")
Y_pred = X.FC([W, B], "Y_pred")
dist = train_net.SquaredL2Distance([Y_gt, Y_pred], "dist")
loss = dist.AveragedLoss([], ["loss"])
# Get gradients for all the computations above. Note that in fact we don't need
# to get the gradient the Y_gt computation, but we'll just leave it there. In
# many cases, I am expecting one to load X and Y from the disk, so there is
# really no operator that will calculate the Y_gt input.
train_net.AddGradientOperators(skip=2)
# updates
train_net.WeightedSum([W, ONE, "W_grad", LR], W)
train_net.WeightedSum([B, ONE, "B_grad", LR], B)
train_net.Print([loss, W, B], [])

# Run all on GPU.
#init_net.RunAllOnGPU()
#train_net.RunAllOnGPU()


plan = core.Plan("toy_regression")
plan.AddNets([init_net, train_net])
plan.AddStep(core.ExecutionStep("init", init_net))
plan.AddStep(core.ExecutionStep("train", train_net, 100))

with open('toy_regression.pbtxt', 'w') as fid:
  fid.write(str(plan.Proto()))

