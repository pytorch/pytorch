from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


'''
Utility for creating ResNets
See "Deep Residual Learning for Image Recognition" by He, Zhang et. al. 2015
'''


class ResNetBuilder():
    '''
    Helper class for constructing residual blocks.
    '''
    def __init__(self, model, prev_blob):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob

    def add_conv(self, in_filters, out_filters, kernel, stride=1, pad=0):
        self.comp_idx += 1
        self.prev_blob = self.model.Conv(
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            pad=pad
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = self.model.Relu(
            self.prev_blob,
            'comp_%d_relu_%d' % (self.comp_count, self.comp_idx)
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters):
        self.prev_blob = self.model.SpatialBN(
            self.prev_blob,
            'comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx),
            num_filters,
            epsilon=1e-3
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
        down_sampling=False,
        spatial_batch_norm=True,
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 1x1
        self.add_conv(
            input_filters,
            base_filters,
            kernel=1,
            stride=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)

        self.add_relu()

        # 3x3 (note the pad, required for keeping dimensions)
        self.add_conv(
            base_filters,
            base_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)
        self.add_relu()

        # 1x1
        last_conv = self.add_conv(base_filters, output_filters, kernel=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(output_filters)

        # Summation with input signal (shortcut)
        # If we need to increase dimensions (feature maps), need to
        # do do a projection for the short cut
        if (output_filters > input_filters):
            shortcut_blob = self.model.Conv(
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                output_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2)
            )
            if spatial_batch_norm:
                shortcut_blob = self.model.SpatialBN(
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    output_filters,
                    epsilon=1e-3,
                )

        self.prev_blob = self.model.Sum(
            [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1

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
            shortcut_blob = self.model.Conv(
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                num_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
            )
            if spatial_batch_norm:
                shortcut_blob = self.model.SpatialBN(
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    num_filters,
                    epsilon=1e-3
                )

        self.prev_blob = self.model.Sum(
            [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1


def create_resnet50(model, data, num_input_channels, num_labels, label=None):
    # conv1 + maxpool
    model.Conv(data, 'conv1', num_input_channels, 64, weight_init=("MSRAFill", {}), kernel=7, stride=2, pad=3)
    model.SpatialBN('conv1', 'conv1_spatbn', 64, epsilon=1e-3)
    model.Relu('conv1_spatbn', 'relu1')
    model.MaxPool('relu1', 'pool1', kernel=3, stride=2)

    # Residual blocks...
    builder = ResNetBuilder(model, 'pool1')

    # conv2_x (ref Table 1 in He et al. (2015))
    builder.add_bottleneck(64, 64, 256)
    builder.add_bottleneck(256, 64, 256)
    builder.add_bottleneck(256, 64, 256)

    # conv3_x
    builder.add_bottleneck(256, 128, 512, down_sampling=True)
    for i in range(1, 4):
        builder.add_bottleneck(512, 128, 512)

    # conv4_x
    builder.add_bottleneck(512, 256, 1024, down_sampling=True)
    for i in range(1, 6):
        builder.add_bottleneck(1024, 256, 1024)

    # conv5_x
    builder.add_bottleneck(1024, 512, 2048, down_sampling=True)
    builder.add_bottleneck(2048, 512, 2048)
    builder.add_bottleneck(2048, 512, 2048)

    # Final layers
    model.AveragePool(builder.prev_blob, 'final_avg', kernel=7, stride=1)

    # Final dimension of the "image" is reduced to 7x7
    model.FC('final_avg', 'pred', 2048, num_labels)

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            ["pred", label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax("pred", "softmax")

def create_resnet_32x32(
    model, data, num_input_channels, num_groups, num_labels
):
    '''
    Create residual net for smaller images (sec 4.2 of He et. al (2015))
    num_groups = 'n' in the paper
    '''
    # conv1 + maxpool
    model.Conv(data, 'conv1', num_input_channels, 16, kernel=3, stride=1)
    model.SpatialBN('conv1', 'conv1_spatbn', 16, epsilon=1e-3)
    model.Relu('conv1_spatbn', 'relu1')

    # Number of blocks as described in sec 4.2
    filters = [16, 32, 64]

    builder = ResNetBuilder(model, 'relu1')
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
    model.AveragePool(builder.prev_blob, 'final_avg', kernel=8, stride=1)
    model.FC('final_avg', 'pred', 64, num_labels)
    softmax = model.Softmax('pred', 'softmax')
    return softmax
