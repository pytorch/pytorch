from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
logging.basicConfig()
log = logging.getLogger("AnyExp")
log.setLevel(logging.DEBUG)

# For more depths, add the block config here
BLOCK_CONFIG = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 32, 36, 3),
    264: (3, 64, 36, 3),
    284: (3, 32, 64, 3),
}


def gen_forward_pass_builder_fun(self, model, dataset, is_train):
    split = 'train' if is_train else 'test'
    opts = self.opts

    def model_creator(model, loss_scale):
        model, softmax, loss = resnet_imagenet_create_model(
            model=model,
            data='data',
            labels='label',
            split=split,
            opts=opts,
            dataset=dataset,
        )
        return [loss]
    return model_creator


def resnet_imagenet_create_model(model, data, labels, split, opts, dataset):
    model_helper = ResNetModelHelper(model, split, opts)
    opts_depth = opts['model_param']['num_layer']
    engine = opts['model_param']['engine']
    log.info(' | ResNet-{} Imagenet'.format(opts_depth))
    assert opts_depth in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth. Please check.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[opts_depth]

    num_features = 2048
    residual_block = model_helper.bottleneck_block
    if opts_depth in [18, 34]:
        num_features = 512
        residual_block = model_helper.basic_block

    num_classes = 1000
    conv_blob = model.Conv(
        data, 'conv1', 3, 64, 7, stride=2, pad=3, weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=0, engine=engine
    )
    test_mode = False
    if split in ['test', 'val']:
        test_mode = True
    bn_blob = model.SpatialBN(
        conv_blob, 'res_conv1_bn', 64,
        # does not appear to affect test_loss performance
        # epsilon=1e-3,
        epsilon=opts['model_param']['bn_epsilon'],
        # momentum=0.1,
        momentum=opts['model_param']['bn_momentum'],
        is_test=test_mode,
    )
    relu_blob = model.Relu(bn_blob, bn_blob)
    max_pool = model.MaxPool(relu_blob, 'pool1', kernel=3, stride=2, pad=1)

    # TODO: This can be further optimized by passing dim_in, dim_out = features,
    # dim_out = features * 4
    if opts_depth in [50, 101, 152, 200, 264, 284]:
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, max_pool, 64, 256, stride=1, num_blocks=n1,
            prefix='res2', dim_inner=64
        )
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n2,
            prefix='res3', dim_inner=128
        )
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, blob_in, dim_in, 1024, stride=2, num_blocks=n3,
            prefix='res4', dim_inner=256
        )
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, blob_in, dim_in, 2048, stride=2, num_blocks=n4,
            prefix='res5', dim_inner=512
        )
    elif opts_depth in [18, 34]:
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, max_pool, 64, 64, stride=1, num_blocks=n1,
            prefix='res2',
        )
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, blob_in, dim_in, 128, stride=2, num_blocks=n2,
            prefix='res3',
        )
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, blob_in, dim_in, 256, stride=2, num_blocks=n3,
            prefix='res4',
        )
        blob_in, dim_in = model_helper.residual_layer(
            residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n4,
            prefix='res5',
        )

    pool_blob = model.AveragePool(blob_in, 'pool5', kernel=7, stride=1)

    loss_scale = 1. / opts['distributed']['num_xpus'] / \
        opts['distributed']['num_shards']

    loss = None

    fc_blob = model.FC(
        pool_blob, 'pred', num_features, num_classes,
        # does not appear to affect test_loss performance
        # weight_init=('GaussianFill', {'std': opts.fc_init_std}),
        # bias_init=('ConstantFill', {'value': 0.})
        weight_init=None,
        bias_init=None)
    softmax, loss = model.SoftmaxWithLoss(
        [fc_blob, labels],
        ['softmax', 'loss'],
        scale=loss_scale)
    model.Accuracy(['softmax', labels], 'accuracy')
    return model, softmax, loss


class ResNetModelHelper():

    def __init__(self, model, split, opts):
        self.model = model
        self.split = split
        self.opts = opts
        self.engine = opts['model_param']['engine']


    # shortcut type B
    def add_shortcut(self, blob_in, dim_in, dim_out, stride, prefix):
        if dim_in == dim_out:
            return blob_in
        conv_blob = self.model.Conv(
            blob_in, prefix, dim_in, dim_out, kernel=1,
            stride=stride,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1, engine=self.engine
        )
        test_mode = False
        if self.split in ['test', 'val']:
            test_mode = True
        bn_blob = self.model.SpatialBN(
            conv_blob, prefix + "_bn", dim_out,
            # epsilon=1e-3,
            # momentum=0.1,
            epsilon=self.opts['model_param']['bn_epsilon'],
            momentum=self.opts['model_param']['bn_momentum'],
            is_test=test_mode,
        )
        return bn_blob

    def conv_bn(
        self, blob_in, dim_in, dim_out, kernel, stride, prefix, group=1, pad=1,
    ):
        conv_blob = self.model.Conv(
            blob_in, prefix, dim_in, dim_out, kernel, stride=stride,
            pad=pad, group=group,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1, engine=self.engine
        )
        test_mode = False
        if self.split in ['test', 'val']:
            test_mode = True
        bn_blob = self.model.SpatialBN(
            conv_blob, prefix + "_bn", dim_out,
            epsilon=self.opts['model_param']['bn_epsilon'],
            momentum=self.opts['model_param']['bn_momentum'],
            is_test=test_mode,
        )
        return bn_blob

    def conv_bn_relu(
        self, blob_in, dim_in, dim_out, kernel, stride, prefix, pad=1, group=1,
    ):
        bn_blob = self.conv_bn(
            blob_in, dim_in, dim_out, kernel, stride, prefix, group=group,
            pad=pad
        )
        return self.model.Relu(bn_blob, bn_blob)

    # 3(a)this block uses multi-way group conv implementation that splits blobs
    def multiway_bottleneck_block(
        self, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group
    ):
        blob_out = self.conv_bn_relu(
            blob_in, dim_in, dim_inner, 1, 1, prefix + "_branch2a", pad=0,
        )

        conv_blob = self.model.GroupConv_Deprecated(
            blob_out, prefix + "_branch2b", dim_inner, dim_inner, kernel=3,
            stride=stride, pad=1, group=group, weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1, engine=self.engine
        )
        test_mode = False
        if self.split in ['test', 'val']:
            test_mode = True
        bn_blob = self.model.SpatialBN(
            conv_blob, prefix + "_branch2b_bn", dim_out,
            epsilon=self.opts['model_param']['bn_epsilon'],
            momentum=self.opts['model_param']['bn_momentum'], is_test=test_mode,
        )
        relu_blob = self.model.Relu(bn_blob, bn_blob)

        bn_blob = self.conv_bn(
            relu_blob, dim_inner, dim_out, 1, 1, prefix + "_branch2c", pad=0
        )
        if self.opts['model_param']['custom_bn_init']:
            self.model.param_init_net.ConstantFill(
                [bn_blob + '_s'], bn_blob + '_s',
                value=self.opts['model_param']['bn_init_gamma'])

        sc_blob = self.add_shortcut(
            blob_in, dim_in, dim_out, stride, prefix=prefix + "_branch1"
        )
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + "_sum")
        return self.model.Relu(sum_blob, sum_blob)

    # 3(c) this block uses cudnn group conv op
    def group_bottleneck_block(
        self, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group
    ):
        blob_out = self.conv_bn_relu(
            blob_in, dim_in, dim_inner, 1, 1, prefix + "_branch2a", pad=0,
        )
        blob_out = self.conv_bn_relu(
            blob_out, dim_inner, dim_inner, 3, stride, prefix + "_branch2b",
            group=group
        )
        bn_blob = self.conv_bn(
            blob_out, dim_inner, dim_out, 1, 1, prefix + "_branch2c", pad=0
        )
        if self.opts['model_param']['custom_bn_init']:
            self.model.param_init_net.ConstantFill(
                [bn_blob + '_s'], bn_blob + '_s',
                value=self.opts['model_param']['bn_init_gamma'])

        sc_blob = self.add_shortcut(
            blob_in, dim_in, dim_out, stride, prefix=prefix + "_branch1"
        )
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + "_sum")
        return self.model.Relu(sum_blob, sum_blob)

    # bottleneck residual layer for 50, 101, 152 layer networks
    def bottleneck_block(
        self, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group=None
    ):
        blob_out = self.conv_bn_relu(
            blob_in, dim_in, dim_inner, 1, 1, prefix + "_branch2a", pad=0,
        )
        blob_out = self.conv_bn_relu(
            blob_out, dim_inner, dim_inner, 3, stride, prefix + "_branch2b",
        )
        bn_blob = self.conv_bn(
            blob_out, dim_inner, dim_out, 1, 1, prefix + "_branch2c", pad=0
        )
        if self.opts['model_param']['custom_bn_init']:
            self.model.param_init_net.ConstantFill(
                [bn_blob + '_s'], bn_blob + '_s',
                value=self.opts['model_param']['bn_init_gamma'])

        sc_blob = self.add_shortcut(
            blob_in, dim_in, dim_out, stride, prefix=prefix + "_branch1"
        )
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + "_sum")
        return self.model.Relu(sum_blob, sum_blob)

    # basic layer for the 18 and 34 layer networks and the CIFAR data netwrorks
    def basic_block(
        self, blob_in, dim_in, dim_out, stride, prefix, dim_inner=None,
        group=None,
    ):
        blob_out = self.conv_bn_relu(
            blob_in, dim_in, dim_out, 3, stride, prefix + "_branch2a"
        )
        bn_blob = self.conv_bn(
            blob_out, dim_out, dim_out, 3, 1, prefix + "_branch2b", pad=1
        )
        sc_blob = self.add_shortcut(
            blob_in, dim_in, dim_out, stride, prefix=prefix + "_branch1"
        )
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + "_sum")
        return self.model.Relu(sum_blob, sum_blob)

    def residual_layer(
        self, block_fn, blob_in, dim_in, dim_out, stride, num_blocks, prefix,
        dim_inner=None, group=None
    ):
        # prefix is something like: res2, res3, etc.
        # each res layer has num_blocks stacked
        for idx in range(num_blocks):
            block_prefix = "{}_{}".format(prefix, idx)
            block_stride = 2 if (idx == 0 and stride == 2) else 1
            blob_in = block_fn(
                blob_in, dim_in, dim_out, block_stride, block_prefix, dim_inner,
                group
            )
            dim_in = dim_out
        return blob_in, dim_in
