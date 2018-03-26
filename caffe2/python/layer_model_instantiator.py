## @package layer_model_instantiator
# Module caffe2.python.layer_model_instantiator
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import InstantiationContext
from caffe2.python.layers.tags import Tags


def _filter_layers(layers, include_tags):
    if include_tags is None:
        return layers
    include_tags = set(include_tags)
    return [l for l in layers if not include_tags.isdisjoint(l.tags)]


def shrink_output_schema(net, out_schema):
    if len(out_schema.field_names()) <= 1:
        return out_schema
    exists = [net.BlobIsDefined(blob) for blob in out_schema.field_blobs()]
    return schema.from_column_list(
        [
            col_name for ok, col_name in
            zip(exists, out_schema.field_names()) if ok
        ],
        [
            col_type for ok, col_type in
            zip(exists, out_schema.field_types()) if ok
        ],
        [
            col_blob for ok, col_blob in
            zip(exists, out_schema.field_blobs()) if ok
        ],
        [
            col_meta for ok, col_meta in
            zip(exists, out_schema.field_metadata()) if ok
        ]
    )


def generate_predict_net(model, include_tags=None):
    predict_net = core.Net('predict_net')

    for layer in _filter_layers(model.layers, include_tags):
        if Tags.EXCLUDE_FROM_PREDICTION not in layer.tags:
            layer.add_operators(
                predict_net, context=InstantiationContext.PREDICTION)

    predict_net.set_input_record(model.input_feature_schema.clone())
    output_schema = shrink_output_schema(
        predict_net, model.output_schema.clone()
    )
    predict_net.set_output_record(output_schema)
    return predict_net


def generate_eval_net(model, include_tags=None):
    eval_net = core.Net('eval_net')

    for layer in _filter_layers(model.layers, include_tags):
        if Tags.EXCLUDE_FROM_EVAL not in layer.tags:
            layer.add_operators(eval_net, context=InstantiationContext.EVAL)

    input_schema = model.input_feature_schema + model.trainer_extra_schema
    eval_net.set_input_record(input_schema)
    output_schema = shrink_output_schema(
        eval_net, model.output_schema + model.metrics_schema
    )
    eval_net.set_output_record(output_schema)
    return eval_net


def _generate_training_net_only(model, include_tags=None):
    train_net = core.Net('train_net')
    train_init_net = model.create_init_net('train_init_net')

    for layer in _filter_layers(model.layers, include_tags):
        if Tags.EXCLUDE_FROM_TRAIN not in layer.tags:
            layer.add_operators(train_net, train_init_net)

    input_schema = model.input_feature_schema + model.trainer_extra_schema
    train_net.set_input_record(input_schema)
    output_schema = shrink_output_schema(
        train_net, model.output_schema + model.metrics_schema
    )
    train_net.set_output_record(output_schema)
    return train_init_net, train_net


def generate_training_nets_forward_only(model, include_tags=None):
    train_init_net, train_net = _generate_training_net_only(model, include_tags)
    return train_init_net, train_net


def generate_training_nets(model, include_tags=None):
    train_init_net, train_net = _generate_training_net_only(model, include_tags)

    model.apply_regularizers_on_loss(train_net, train_init_net)
    if not model.has_loss():
        return train_init_net, train_net
    loss = model.loss
    grad_map = train_net.AddGradientOperators(loss.field_blobs())
    model.apply_post_grad_net_modifiers(train_net, train_init_net, grad_map)
    model.apply_optimizers(train_net, train_init_net, grad_map)
    model.apply_regularizers_after_optimizer(train_net, train_init_net, grad_map)
    model.apply_final_net_modifiers(train_net, train_init_net, grad_map)

    return train_init_net, train_net
