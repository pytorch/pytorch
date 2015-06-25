from pycaffe2 import core
from pycaffe2 import core_gradients

init_net = core.Net("init")
params = []

LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
DECAY = init_net.ConstantFill([], "DECAY", shape=[1], value=0.999)

order = "NCHW"

test_net = core.Net("train")
data, label = test_net.ImageInput(
    [], ["data", "label"], batch_size=64,
    db="/media/data/jiayq/Data/ILSVRC12/caffe2-train-lmdb", db_type='lmdb',
    mean=128., std=128., scale=256, crop=227, mirror=1)
if order == "NCHW":
  data = data.NHWC2NCHW()

filter1 = init_net.XavierFill([], "filter1", shape=[96, 3, 11, 11])
bias1 = init_net.ConstantFill([], "bias1", shape=[96,], value=0.0)
pool1, _ = (data.Conv([filter1, bias1], kernel=11, pad=0, stride=4, order=order)
                .Relu()
                .LRN(outputs=2, size=5, alpha=0.0001, beta=0.75, order=order)[0]
                .MaxPool(outputs=2, kernel=3, stride=2, order=order))
pool1a, pool1b = pool1.DepthSplit(outputs=2, channels=[48, 48], order=order)
filter2a = init_net.XavierFill([], "filter2a", shape=[128, 48, 5, 5])
bias2a = init_net.ConstantFill([], "bias2a", shape=[128,], value=0.0)
filter2b = init_net.XavierFill([], "filter2b", shape=[128, 48, 5, 5])
bias2b = init_net.ConstantFill([], "bias2b", shape=[128,], value=0.0)
pool2a, _ = (pool1a.Conv([filter2a, bias2a], kernel=5, pad=2, order=order)
                   .Relu()
                   .LRN(outputs=2,size=5, alpha=0.0001, beta=0.75, order=order)[0]
                   .MaxPool(outputs=2, kernel=3, stride=2, order=order))
pool2b, _ = (pool1b.Conv([filter2b, bias2b], kernel=5, pad=2, order=order)
                   .Relu()
                   .LRN(outputs=2, size=5, alpha=0.0001, beta=0.75, order=order)[0]
                   .MaxPool(outputs=2, kernel=3, stride=2, order=order))
pool2 = pool2a.DepthConcat([pool2b], channels=[128, 128], order=order)
filter3 = init_net.XavierFill([], "filter3", shape=[384, 256, 3, 3])
bias3 = init_net.ConstantFill([], "bias3", shape=[384,], value=0.0)
conv3a, conv3b = (pool2.Conv([filter3, bias3], kernel=3, pad=1, order=order).Relu()
                       .DepthSplit(outputs=2, channels=[192, 192], order=order))
filter4a = init_net.XavierFill([], "filter4a", shape=[192, 192, 3, 3])
bias4a = init_net.ConstantFill([], "bias4a", shape=[192,], value=0.0)
filter4b = init_net.XavierFill([], "filter4b", shape=[192, 192, 3, 3])
bias4b = init_net.ConstantFill([], "bias4b", shape=[192,], value=0.0)
conv4a = conv3a.Conv([filter4a, bias4a], kernel=3, pad=1, order=order).Relu()
conv4b = conv3b.Conv([filter4b, bias4b], kernel=3, pad=1, order=order).Relu()

filter5a = init_net.XavierFill([], "filter5a", shape=[128, 192, 3, 3])
bias5a = init_net.ConstantFill([], "bias5a", shape=[128,], value=0.0)
filter5b = init_net.XavierFill([], "filter5b", shape=[128, 192, 3, 3])
bias5b = init_net.ConstantFill([], "bias5b", shape=[128,], value=0.0)
conv5a = conv4a.Conv([filter5a, bias5a], kernel=3, pad=1, order=order)
conv5b = conv4b.Conv([filter5b, bias5b], kernel=3, pad=1, order=order)
pool5_flatten = (conv5a.DepthConcat([conv5b], channels=[128, 128], order=order)
                       .MaxPool(outputs=2, kernel=3, stride=2, order=order)[0]
                       .Relu().Flatten())
W6 = init_net.XavierFill([], "W6", shape=[4096, 6 * 6 * 256])
B6 = init_net.ConstantFill([], "B6", shape=[4096], value=0.0)
W7 = init_net.XavierFill([], "W7", shape=[4096, 4096])
B7 = init_net.ConstantFill([], "B7", shape=[4096], value=0.0)
W8 = init_net.ConstantFill([], "W8", shape=[1000, 4096])
B8 = init_net.ConstantFill([], "B8", shape=[1000], value=0.0)
pred = (pool5_flatten.FC([W6, B6]).Relu().Dropout(outputs=2)[0]
                   .FC([W7, B7]).Relu().Dropout(outputs=2)[0]
                   .FC([W8, B8]).Softmax())
xent = pred.LabelCrossEntropy([label], "xent")
# The loss function.
loss, xent_grad = xent.AveragedLoss([], ["loss", xent.Grad()])
test_net.AddGradientOperators(first=2)
test_net.Print([loss], [])

dump_net = core.Net("dump")
for blob in [data, pool1, pool1a, pool1b, pool2, conv3a, conv3b, conv4a, conv4b,
             conv5a, conv5b, pool5_flatten]:
  dump_net.SaveFloatTensor([blob], [], file=str(blob))

init_net.RunAllOnGPU()
test_net.RunAllOnGPU()
dump_net.RunAllOnGPU()


plan = core.Plan("alexnet")
plan.AddNets([init_net, test_net, dump_net])
plan.AddStep(core.ExecutionStep("init", init_net))
plan.AddStep(core.ExecutionStep("first_run", test_net))
#plan.AddStep(core.ExecutionStep("subsequent_run", test_net, 10))
plan.AddStep(core.ExecutionStep("dump", dump_net))

with open('alexnet.pbtxt', 'w') as fid:
  fid.write(str(plan.Proto()))
