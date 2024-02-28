# Module caffe2.python.models.shufflenet






from caffe2.python import brew

"""
Utilitiy for creating ShuffleNet
"ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design" by Ma et. al. 2018
"""

OUTPUT_CHANNELS = {
    '0.5x': [24, 48, 96, 192, 1024],
    '1.0x': [24, 116, 232, 464, 1024],
    '1.5x': [24, 176, 352, 704, 1024],
    '2.0x': [24, 244, 488, 976, 2048],
}


class ShuffleNetV2Builder():
    def __init__(
        self,
        model,
        data,
        num_input_channels,
        num_labels,
        num_groups=2,
        width='1.0x',
        is_test=False,
        detection=False,
        bn_epsilon=1e-5,
    ):
        self.model = model
        self.prev_blob = data
        self.num_input_channels = num_input_channels
        self.num_labels = num_labels
        self.num_groups = num_groups
        self.output_channels = OUTPUT_CHANNELS[width]
        self.stage_repeats = [3, 7, 3]
        self.is_test = is_test
        self.detection = detection
        self.bn_epsilon = bn_epsilon

    def create(self):
        in_channels = self.output_channels[0]

        self.prev_blob = brew.conv(self.model, self.prev_blob, 'stage1_conv',
                                   self.num_input_channels, in_channels,
                                   weight_init=("MSRAFill", {}),
                                   kernel=3, stride=2)
        self.prev_blob = brew.max_pool(self.model, self.prev_blob,
                                       'stage1_pool', kernel=3, stride=2)

        # adds stage#{2,3,4}; see table 5 of the ShufflenetV2 paper.
        for idx, (out_channels, n_repeats) in enumerate(zip(
            self.output_channels[1:4], self.stage_repeats
        )):
            prefix = 'stage{}_stride{}'.format(idx + 2, 2)
            self.add_spatial_ds_unit(prefix, in_channels, out_channels)
            in_channels = out_channels
            for i in range(n_repeats):
                prefix = 'stage{}_stride{}_repeat{}'.format(
                    idx + 2, 1, i + 1
                )
                self.add_basic_unit(prefix, in_channels)

        self.last_conv = brew.conv(self.model, self.prev_blob, 'conv5',
                                   in_channels, self.output_channels[4],
                                   kernel=1)
        self.avg_pool = self.model.AveragePool(self.last_conv, 'avg_pool',
                                               kernel=7)
        self.last_out = brew.fc(self.model,
                                self.avg_pool,
                                'last_out_L{}'.format(self.num_labels),
                                self.output_channels[4],
                                self.num_labels)

    # spatial down sampling unit with stride=2
    def add_spatial_ds_unit(self, prefix, in_channels, out_channels, stride=2):
        right = left = self.prev_blob
        out_channels = out_channels // 2

        # Enlarge the receptive field for detection task
        if self.detection:
            left = self.add_detection_unit(left, prefix + '_left_detection',
                                           in_channels, in_channels)

        left = self.add_dwconv3x3_bn(left, prefix + 'left_dwconv',
                                     in_channels, stride)
        left = self.add_conv1x1_bn(left, prefix + '_left_conv1', in_channels,
                                   out_channels)

        if self.detection:
            right = self.add_detection_unit(right, prefix + '_right_detection',
                                            in_channels, in_channels)

        right = self.add_conv1x1_bn(right, prefix + '_right_conv1',
                                    in_channels, out_channels)
        right = self.add_dwconv3x3_bn(right, prefix + '_right_dwconv',
                                      out_channels, stride)
        right = self.add_conv1x1_bn(right, prefix + '_right_conv2',
                                    out_channels, out_channels)

        self.prev_blob = brew.concat(self.model, [right, left],
                                     prefix + '_concat')
        self.prev_blob = self.model.net.ChannelShuffle(
            self.prev_blob, prefix + '_ch_shuffle',
            group=self.num_groups, kernel=1
        )

    # basic unit with stride=1
    def add_basic_unit(self, prefix, in_channels, stride=1):
        in_channels = in_channels // 2
        left = prefix + '_left'
        right = prefix + '_right'
        self.model.net.Split(self.prev_blob, [left, right])

        if self.detection:
            right = self.add_detection_unit(right, prefix + '_right_detection',
                                            in_channels, in_channels)

        right = self.add_conv1x1_bn(right, prefix + '_right_conv1',
                                    in_channels, in_channels)
        right = self.add_dwconv3x3_bn(right, prefix + '_right_dwconv',
                                      in_channels, stride)
        right = self.add_conv1x1_bn(right, prefix + '_right_conv2',
                                    in_channels, in_channels)

        self.prev_blob = brew.concat(self.model, [right, left],
                                     prefix + '_concat')

        self.prev_blob = self.model.net.ChannelShuffle(
            self.prev_blob, prefix + '_ch_shuffle',
            group=self.num_groups, kernel=1
        )

    # helper functions to create net's units
    def add_detection_unit(self, prev_blob, prefix, in_channels, out_channels,
                           kernel=3, pad=1):
        out_blob = brew.conv(self.model, prev_blob, prefix + '_conv',
                             in_channels, out_channels, kernel=kernel,
                             weight_init=("MSRAFill", {}),
                             group=in_channels, pad=pad)
        out_blob = brew.spatial_bn(self.model, out_blob, prefix + '_bn',
                                   out_channels, epsilon=self.bn_epsilon,
                                   is_test=self.is_test)
        return out_blob

    def add_conv1x1_bn(self, prev_blob, blob, in_channels, out_channels):
        prev_blob = brew.conv(self.model, prev_blob, blob, in_channels,
                              out_channels, kernel=1,
                              weight_init=("MSRAFill", {}))
        prev_blob = brew.spatial_bn(self.model, prev_blob, prev_blob + '_bn',
                                    out_channels,
                                    epsilon=self.bn_epsilon,
                                    is_test=self.is_test)
        prev_blob = brew.relu(self.model, prev_blob, prev_blob)
        return prev_blob

    def add_dwconv3x3_bn(self, prev_blob, blob, channels, stride):
            prev_blob = brew.conv(self.model, prev_blob, blob, channels,
                                  channels, kernel=3,
                                  weight_init=("MSRAFill", {}),
                                  stride=stride, group=channels, pad=1)
            prev_blob = brew.spatial_bn(self.model, prev_blob,
                                        prev_blob + '_bn',
                                        channels,
                                        epsilon=self.bn_epsilon,
                                        is_test=self.is_test)
            return prev_blob


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
    builder.create()

    if no_loss:
        return builder.last_out

    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [builder.last_out, label],
            ["softmax", "loss"],
        )
        return (softmax, loss)
