from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.python.models.resnet as resnet


def gen_forward_pass_builder_fun(self, model, dataset, is_train):
    def create_resnet50_model_ops(model, loss_scale):
        [softmax, loss] = resnet.create_resnet50(
            model,
            "data",
            num_input_channels=3,
            num_labels=1000,
            label="label",
        )
        model.Accuracy([softmax, "label"], "accuracy")

        my_loss_scale = 1. / self.opts['distributed']['num_xpus'] / \
            self.opts['distributed']['num_shards']

        loss = model.Scale(loss, scale=my_loss_scale)

        return [loss]
    return create_resnet50_model_ops
