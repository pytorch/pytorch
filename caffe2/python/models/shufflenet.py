# Module caffe2.python.models.shufflenet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import brew

"""
Utilitiy for creating ShuffleNet
"ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design" by Ma et. al. 2018
"""

SCALE_CHANNELS = {
    '0.5x': [24, 48, 96, 192, 1024],
    '1.0x': [24, 116, 232, 464, 1024],
    '1.5x': [24, 176, 352, 704, 1024],
    '2.0x': [24, 244, 488, 976, 2048],
}

STRIDE_1_REPEAT = [3, 7, 3]
STRIDE_2_REPEAT = [1, 1, 1]


class ShuffleNetV2Builder():
    def __init__(
        self,
        model,
        data,
        num_input_channels,
        num_labels,
        num_groups=2,
        output_channels=SCALE_CHANNELS['1.0x'],
        stride_1_repeat_times=STRIDE_1_REPEAT,
        stride_2_repeat_times=STRIDE_2_REPEAT,
        is_test=False,
        detection=False,
        use_bn=True,
    ):
        self.model = model
        self.data = data
        self.num_input_channels = num_input_channels
        self.num_labels = num_labels
        self.num_groups = num_groups
        self.output_channels = output_channels
        self.stride_1_repeat_times = stride_1_repeat_times
        self.stride_2_repeat_times = stride_2_repeat_times
        self.is_test = is_test
        self.detection = detection
        self.use_bn = use_bn

    def create(self):
        """Build a shuffle net.
            If model built for detection, an additional 3x3
            depthwise convolution would be added before the first pointwise
            convolution in each building block.
        """

        dim_in = self.output_channels[0]
        dim_out = self.output_channels[0]

        s = brew.conv(self.model, self.data, 'conv_1',
                      self.num_input_channels, dim_out,
                      kernel=3, stride=2)
        s = brew.max_pool(self.model, s, 'pool_1', kernel=3, stride=2)

        for idx, (dim_out, n_stride_1, n_stride_2) in enumerate(
            zip(self.output_channels[1:4],
                self.stride_1_repeat_times, self.stride_2_repeat_times)):

            for i in range(n_stride_2):
                s, dim_in = self.add_block_stride_2(
                    'stage_' + str(idx + 2) + '_stride2_' + str(i + 1),
                    s, dim_in, dim_out)

            for i in range(n_stride_1):
                s, dim_in = self.add_block_stride_1(
                    'stage_' + str(idx + 2) + '_stride1_' + str(i + 1),
                    s, dim_in, dim_out)

        s = brew.conv(self.model, s, 'conv_5', dim_in,
                      self.output_channels[4], kernel=1)
        s = self.model.AveragePool(s, 'avg_pooled', kernel=7)
        s = brew.fc(self.model, s, 'last_out_L{}'.format(self.num_labels),
                    self.output_channels[4],
                    self.num_labels)
        print('Build shufflenet_v2 successfully!')
        return s, self.num_labels

    def add_block_stride_2(self, prefix, blob_in, dim_in, dim_out):
        dim_out = int(dim_out / 2)
        right = left = blob_in

        # Enlarge the receptive field for detection task
        if self.detection:
            right = brew.conv(self.model, right, prefix + '_right_conv_d',
                              dim_in, dim_in, kernel=3, group=dim_in, pad=1)
            right = brew.spatial_bn(self.model, right, right + '_bn',
                                    dim_in, epsilon=1e-3,
                                    is_test=self.is_test)

        right = brew.conv(self.model, right, prefix + '_right_conv_1',
                          dim_in, dim_in, kernel=1)
        if self.use_bn:
            right = brew.spatial_bn(self.model, right, right + '_bn', dim_in,
                                    epsilon=1e-3,
                                    is_test=self.is_test)
        right = brew.relu(self.model, right, right)

        right = brew.conv(self.model, right, prefix + '_right_dwconv', dim_in,
                          dim_in, kernel=3, stride=2, group=dim_in, pad=1)
        if self.use_bn:
            right = brew.spatial_bn(self.model, right, right + '_nb', dim_in,
                                    epsilon=1e-3,
                                    is_test=self.is_test)

        right = brew.conv(self.model, right, prefix + '_right_conv_3', dim_in,
                          dim_out, kernel=1)
        if self.use_bn:
            right = brew.spatial_bn(self.model, right, right + '_bn', dim_out,
                                    epsilon=1e-3,
                                    is_test=self.is_test)
        right = brew.relu(self.model, right, right)

        if self.detection:
            left = brew.conv(self.model, left, prefix + '_left_conv_d', dim_in,
                             dim_in, kernel=3, group=dim_in, pad=1)
            if self.use_bn:
                left = brew.spatial_bn(self.model, left, right + '_bn', dim_in,
                                       epsilon=1e-3,
                                       is_test=self.is_test)

        left = brew.conv(self.model, left, prefix + '_left_dwconv', dim_in,
                         dim_in, kernel=3, stride=2, group=dim_in, pad=1)
        if self.use_bn:
            left = brew.spatial_bn(self.model, left, left + '_bn', dim_in,
                                   epsilon=1e-3,
                                   is_test=self.is_test)

        left = brew.conv(self.model, left, prefix + '_left_conv_1', dim_in,
                         dim_out, kernel=1)
        if self.use_bn:
            left = brew.spatial_bn(self.model, left, left + '_bn', dim_out,
                                   epsilon=1e-3,
                                   is_test=self.is_test)
        left = brew.relu(self.model, left, left)

        concated = brew.concat(self.model, [right, left], prefix + '_concated')

        shuffled = self.model.net.ChannelShuffle(
            concated, prefix + '_shuffled',
            group=self.num_groups, kernel=1
        )
        return shuffled, dim_out * 2

    def add_block_stride_1(self, prefix, blob_in, dim_in, dim_out):
        dim_in = int(dim_in / 2)
        dim_out = int(dim_out / 2)
        self.model.net.Split(blob_in, [prefix + '_left', prefix + '_right'])
        right = prefix + '_right'

        if self.detection:
            right = brew.conv(self.model, right, prefix + '_right_conv_d',
                              dim_in, dim_in, kernel=3, group=dim_in, pad=1)
            if self.use_bn:
                right = brew.spatial_bn(self.model, right, right + '_bn',
                                        dim_in, epsilon=1e-3,
                                        is_test=self.is_test)

        right = brew.conv(self.model, right, prefix + '_right_conv_1', dim_in,
                          dim_in, kernel=1)
        if self.use_bn:
            right = brew.spatial_bn(self.model, right, right + '_bn', dim_in,
                                    epsilon=1e-3, is_test=self.is_test)
        right = brew.relu(self.model, right, right)

        right = brew.conv(self.model, right, prefix + '_right_dwcon', dim_in,
                          dim_in, kernel=3, stride=1, group=dim_in, pad=1)
        if self.use_bn:
            right = brew.spatial_bn(self.model, right, right + '_bn', dim_in,
                                    epsilon=1e-3, is_test=False)
        right = brew.conv(self.model, right, prefix + '_right_conv_3', dim_in,
                          dim_out, kernel=1)

        if self.use_bn:
            right = brew.spatial_bn(self.model, right, right + '_bn', dim_out,
                                    epsilon=1e-3,
                                    is_test=self.is_test)
        right = brew.relu(self.model, right, right)

        concated = brew.concat(self.model, [right, prefix + '_left'],
                               prefix + '_concated')

        shuffled = self.model.net.ChannelShuffle(
            concated, prefix + '_shuffled',
            group=self.num_groups, kernel=1
        )
        return shuffled, dim_out * 2


def create_shufflenet(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
):
    builder = ShuffleNetV2Builder(model, data, num_input_channels,
                                  num_labels,
                                  is_test=is_test)
    last_out, dim_out = builder.create()

    if no_loss:
        return last_out

    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )
        return (softmax, loss)
