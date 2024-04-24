# Using TensorBoard in ifbpy #

## Simple Example ##

```lang=py

import caffe2.contrib.tensorboard.tensorboard as tb
import caffe2.contrib.tensorboard.tensorboard_exporter as tb_exporter
from caffe2.python import brew, core, model_helper

model = model_helper.ModelHelper(name="overfeat")
data, label = brew.image_input(
    model, ["db"], ["data", "label"], is_test=0
)
with core.NameScope("conv1"):
    conv1 = brew.conv(model, data, "conv1", 3, 96, 11, stride=4)
    relu1 = brew.relu(model, conv1, conv1)
    pool1 = brew.max_pool(model, relu1, "pool1", kernel=2, stride=2)
with core.NameScope("conv2"):
    conv2 = brew.conv(model, pool1, "conv2", 96, 256, 5)
    relu2 = brew.relu(model, conv2, conv2)
    pool2 = brew.max_pool(model, relu2, "pool2", kernel=2, stride=2)
with core.NameScope("conv3"):
    conv3 = brew.conv(model, pool2, "conv3", 256, 512, 3, pad=1)
    relu3 = brew.relu(model, conv3, conv3)
with core.NameScope("conv4"):
    conv4 = brew.conv(model, relu3, "conv4", 512, 1024, 3, pad=1)
    relu4 = brew.relu(model, conv4, conv4)
with core.NameScope("conv5"):
    conv5 = brew.conv(model, relu4, "conv5", 1024, 1024, 3, pad=1)
    relu5 = brew.relu(model, conv5, conv5)
    pool5 = brew.max_pool(model, relu5, "pool5", kernel=2, stride=2)
with core.NameScope("fc6"):
    fc6 = brew.fc(model, pool5, "fc6", 1024*6*6, 3072)
    relu6 = brew.relu(model, fc6, "fc6")
with core.NameScope("fc7"):
    fc7 = brew.fc(model, relu6, "fc7", 3072, 4096)
    relu7 = brew.relu(model, fc7, "fc7")
with core.NameScope("classifier"):
    fc8 = brew.fc(model, relu7, "fc8", 4096, 1000)
    pred = brew.softmax(model, fc8, "pred")
    xent = model.LabelCrossEntropy([pred, label], "xent")
    loss = model.AveragedLoss(xent, "loss")
model.net.RunAllOnGPU()
model.param_init_net.RunAllOnGPU()
model.AddGradientOperators([loss], skip=1)

tb.Config.HEIGHT = 700
tb.visualize_cnn(model)
```
