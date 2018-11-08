## @package resnet
# Module caffe2.python.models.resnet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import brew
import logging

'''
Utility for creating ResNe(X)t
"Deep Residual Learning for Image Recognition" by He, Zhang et. al. 2015
"Aggregated Residual Transformations for Deep Neural Networks" by Xie et. al. 2016
'''


class ResNetBuilder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(
        self,
        model,
        prev_blob,
        no_bias,
        is_test,
        bn_epsilon=1e-5,
        bn_momentum=0.9,
    ):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.no_bias = 1 if no_bias else 0

    def add_conv(
        self,
        in_filters,
        out_filters,
        kernel,
        stride=1,
        group=1,
        pad=0,
    ):
        self.comp_idx += 1
        self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            group=group,
            pad=pad,
            no_bias=self.no_bias,
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            'comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx),
            num_filters,
            epsilon=self.bn_epsilon,
            momentum=self.bn_momentum,
            is_test=self.is_test,
        )
        return self.prev_blob

    '''
    Add a "bottleneck" component as decribed in He et. al. Figure 3 (right)
    '''

    def add_bottleneck(
        self,
        input_filters,   # num of feature maps from preceding layer
        base_filters,    # num of filters internally in the component
        output_filters,  # num of feature maps to output
        stride=1,
        group=1,
        spatial_batch_norm=True,
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 1x1
        self.add_conv(
            input_filters,
            base_filters,
            kernel=1,
            stride=1,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)

        self.add_relu()

        # 3x3 (note the pad, required for keeping dimensions)
        self.add_conv(
            base_filters,
            base_filters,
            kernel=3,
            stride=stride,
            group=group,
            pad=1,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)
        self.add_relu()

        # 1x1
        last_conv = self.add_conv(base_filters, output_filters, kernel=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(output_filters)

        # Summation with input signal (shortcut)
        # When the number of feature maps mismatch between the input
        # and output (this usually happens when the residual stage
        # changes), we need to do a projection for the short cut
        if output_filters != input_filters:
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                output_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=stride,
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    output_filters,
                    epsilon=self.bn_epsilon,
                    momentum=self.bn_momentum,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1

        return output_filters

    def add_simple_block(
        self,
        input_filters,
        num_filters,
        down_sampling=False,
        spatial_batch_norm=True
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 3x3
        self.add_conv(
            input_filters,
            num_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(num_filters, num_filters, kernel=3, pad=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                num_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1


def create_resnet_32x32(
    model, data, num_input_channels, num_groups, num_labels, is_test=False
):
    '''
    Create residual net for smaller images (sec 4.2 of He et. al (2015))
    num_groups = 'n' in the paper
    '''
    # conv1 + maxpool
    brew.conv(
        model, data, 'conv1', num_input_channels, 16, kernel=3, stride=1
    )
    brew.spatial_bn(
        model, 'conv1', 'conv1_spatbn', 16, epsilon=1e-3, is_test=is_test
    )
    brew.relu(model, 'conv1_spatbn', 'relu1')

    # Number of blocks as described in sec 4.2
    filters = [16, 32, 64]

    builder = ResNetBuilder(model, 'relu1', no_bias=0, is_test=is_test)
    prev_filters = 16
    for groupidx in range(0, 3):
        for blockidx in range(0, 2 * num_groups):
            builder.add_simple_block(
                prev_filters if blockidx == 0 else filters[groupidx],
                filters[groupidx],
                down_sampling=(True if blockidx == 0 and
                               groupidx > 0 else False))
        prev_filters = filters[groupidx]

    # Final layers
    brew.average_pool(
        model, builder.prev_blob, 'final_avg', kernel=8, stride=1
    )
    brew.fc(model, 'final_avg', 'last_out', 64, num_labels)
    softmax = brew.softmax(model, 'last_out', 'softmax')
    return softmax


RESNEXT_BLOCK_CONFIG = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}

RESNEXT_STRIDES = [1, 2, 2, 2]

logging.basicConfig()
log = logging.getLogger("resnext_builder")
log.setLevel(logging.DEBUG)


# The conv1 and final_avg kernel/stride args provide a basic mechanism for
# adapting resnet50 for different sizes of input images.
def create_resnext(
    model,
    data,
    num_input_channels,
    num_labels,
    num_layers,
    num_groups,
    num_width_per_group,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=1,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
    log=None,
    bn_epsilon=1e-5,
    bn_momentum=0.9,
):
    if num_layers not in RESNEXT_BLOCK_CONFIG:
        log.error("{}-layer is invalid for resnext config".format(num_layers))

    num_blocks = RESNEXT_BLOCK_CONFIG[num_layers]
    strides = RESNEXT_STRIDES
    num_filters = [64, 256, 512, 1024, 2048]

    if num_layers in [18, 34]:
        num_filters = [64, 64, 128, 256, 512]

    # the number of features before the last FC layer
    num_features = num_filters[-1]

    # conv1 + maxpool
    conv_blob = brew.conv(
        model,
        data,
        'conv1',
        num_input_channels,
        num_filters[0],
        weight_init=("MSRAFill", {}),
        kernel=conv1_kernel,
        stride=conv1_stride,
        pad=3,
        no_bias=no_bias
    )

    bn_blob = brew.spatial_bn(
        model,
        conv_blob,
        'conv1_spatbn_relu',
        num_filters[0],
        epsilon=bn_epsilon,
        momentum=bn_momentum,
        is_test=is_test
    )
    relu_blob = brew.relu(model, bn_blob, bn_blob)
    max_pool = brew.max_pool(model, relu_blob, 'pool1', kernel=3, stride=2, pad=1)

    # Residual blocks...
    builder = ResNetBuilder(model, max_pool, no_bias=no_bias,
                            is_test=is_test, bn_epsilon=1e-5, bn_momentum=0.9)

    inner_dim = num_groups * num_width_per_group

    # 4 different kinds of residual blocks
    for residual_idx in range(4):
        residual_num = num_blocks[residual_idx]
        residual_stride = strides[residual_idx]
        dim_in = num_filters[residual_idx]

        for blk_idx in range(residual_num):
            dim_in = builder.add_bottleneck(
                dim_in,
                inner_dim,
                num_filters[residual_idx + 1],  # dim out
                stride=residual_stride if blk_idx == 0 else 1,
                group=num_groups,
            )

        inner_dim *= 2

    # Final layers
    final_avg = brew.average_pool(
        model,
        builder.prev_blob,
        'final_avg',
        kernel=final_avg_kernel,
        stride=1,
        global_pooling=True,
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = brew.fc(
        model, final_avg, 'last_out_L{}'.format(num_labels), num_features, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, last_out, "softmax")


# The conv1 and final_avg kernel/stride args provide a basic mechanism for
# adapting resnet50 for different sizes of input images.
def create_resnet50(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
):
    # resnet50 is a special case for ResNeXt50-1x64d
    return create_resnext(
        model,
        data,
        num_input_channels,
        num_labels,
        num_layers=50,
        num_groups=1,
        num_width_per_group=64,
        label=label,
        is_test=is_test,
        no_loss=no_loss,
        no_bias=no_bias,
        conv1_kernel=conv1_kernel,
        conv1_stride=conv1_stride,
        final_avg_kernel=final_avg_kernel,
    )
