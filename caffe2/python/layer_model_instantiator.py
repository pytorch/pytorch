from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import InstantiationContext
from caffe2.python.layers.tags import Tags

import itertools


def generate_predict_net(model):
    predict_net = core.Net('predict_net')

    for layer in model.layers:
        if Tags.TRAIN_ONLY not in layer.tags:
            layer.add_operators(
                predict_net, context=InstantiationContext.PREDICTION)
    return predict_net


def generate_eval_net(model):
    eval_net = core.Net('eval_net')

    for layer in model.layers:
        layer.add_operators(
            eval_net, context=InstantiationContext.PREDICTION)

    input_schema = model.input_feature_schema + model.trainer_extra_schema
    output_schema = model.output_schema + model.metrics_schema
    eval_net.set_input_record(input_schema)
    eval_net.set_output_record(output_schema)
    return eval_net


def _generate_training_net_only(model):
    train_net = core.Net('train_net')
    train_init_net = model.create_init_net('train_init_net')

    for layer in model.layers:
        layer.add_operators(train_net, train_init_net)
    return train_init_net, train_net


def generate_training_nets_forward_only(model):
    train_init_net, train_net = _generate_training_net_only(model)
    train_net.set_input_record(model.input_feature_schema)
    return train_init_net, train_net


def generate_training_nets(model):
    train_init_net, train_net = _generate_training_net_only(model)

    loss = model.loss
    grad_map = train_net.AddGradientOperators(loss.field_blobs())
    for param, optimizer in model.param_to_optim.items():
        if not optimizer:
            optimizer = model.default_optimizer
        optimizer(train_net, train_init_net, param, grad_map[str(param)])

    trainer_schema = schema.Struct(
        *itertools.chain(
            model.trainer_extra_schema.get_children(),
            model.input_feature_schema.get_children(),
        )
    )

    train_net.set_input_record(trainer_schema)
    return train_init_net, train_net
