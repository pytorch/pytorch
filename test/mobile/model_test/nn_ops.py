# https://pytorch.org/docs/stable/nn.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNOpsModule(torch.nn.Module):
    def __init__(self):
        super(NNOpsModule, self).__init__()

        self.input1d = torch.randn(20, 16, 50)
        self.module1d = nn.ModuleList(
            [
                nn.Conv1d(16, 33, 3, stride=2),
                nn.ConvTranspose1d(16, 33, 3, stride=2),
                nn.Fold(output_size=(4, 5), kernel_size=(2, 2)),
                # pooling
                nn.MaxPool1d(3, stride=2),
                nn.AvgPool1d(3, stride=2),
                nn.LPPool1d(2, 3, stride=2),
                nn.AdaptiveMaxPool1d(3),
                nn.AdaptiveAvgPool1d(3),
                # padding
                nn.ReflectionPad1d(2),
                nn.ReplicationPad1d(2),
                nn.ConstantPad1d(2, 3.5),
                # normalization
                nn.BatchNorm1d(16),
                nn.InstanceNorm1d(16),
            ]
        )

        self.input2d = torch.randn(20, 16, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                ),
                nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)),
                nn.Unfold(kernel_size=(2, 3)),
                # pooling
                nn.MaxPool2d((3, 2), stride=(2, 1)),
                nn.AvgPool2d((3, 2), stride=(2, 1)),
                nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5)),
                nn.LPPool2d(2, 3, stride=(2, 1)),
                nn.AdaptiveMaxPool2d((5, 7)),
                nn.AdaptiveAvgPool2d((7)),
                # padding
                nn.ReflectionPad2d(2),
                nn.ReplicationPad2d(2),
                nn.ZeroPad2d(2),
                nn.ConstantPad2d(2, 3.5),
                # normalization
                nn.BatchNorm2d(16),
                nn.GroupNorm(4, 16),
                nn.InstanceNorm2d(16),
                nn.LayerNorm([16, 30, 10]),
                nn.LocalResponseNorm(2),
            ]
        )

        self.input3d = torch.randn(10, 16, 10, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)),
                nn.ConvTranspose3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2)
                ),
                # pooling
                nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2)),
                nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2)),
                nn.FractionalMaxPool3d(3, output_size=(13, 12, 11)),
                nn.AdaptiveMaxPool3d((5, 7, 9)),
                nn.AdaptiveAvgPool3d((5, 7, 9)),
                # padding
                nn.ReflectionPad3d(1),
                nn.ReplicationPad3d(3),
                nn.ConstantPad3d(3, 3.5),
                # normalization
                nn.BatchNorm3d(16),
                nn.InstanceNorm3d(16),
                nn.ChannelShuffle(2),
            ]
        )

        self.poolingModule2d = nn.ModuleList(
            [
                nn.MaxPool2d(2, stride=2),
                # nn.MaxUnpool2d(2, stride=2),
            ]
        )

        self.activations = nn.ModuleList(
            [
                nn.ELU(),
                nn.Hardshrink(),
                nn.Hardsigmoid(),
                nn.Hardtanh(),
                nn.Hardswish(),
                nn.LeakyReLU(),
                nn.LogSigmoid(),
                # nn.MultiheadAttention(),
                nn.PReLU(),
                nn.ReLU(),
                nn.ReLU6(),
                nn.RReLU(),
                nn.SELU(),
                nn.CELU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.SiLU(),
                nn.Mish(),
                nn.Softplus(),
                nn.Softshrink(),
                nn.Softsign(),
                nn.Tanh(),
                nn.Tanhshrink(),
                # nn.Threshold(0.1, 20),
                nn.GLU(),
                nn.Softmin(),
                nn.Softmax(),
                nn.Softmax2d(),
                nn.LogSoftmax(),
                # nn.AdaptiveLogSoftmaxWithLoss(),
            ]
        )

        self.rnn = nn.ModuleList(
            [
                nn.RNN(10, 20, 2),
                nn.RNNCell(10, 20),
            ]
        )

        self.gru = nn.ModuleList([nn.GRU(10, 20, 2), nn.GRUCell(10, 20)])

        self.lstm = nn.ModuleList(
            [
                nn.LSTM(10, 20, 2),
                nn.LSTMCell(10, 20),
            ]
        )

        self.transformers = nn.ModuleList(
            [
                nn.Transformer(
                    d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1
                ),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=4, nhead=2), num_layers=1
                ),
                nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=4, nhead=2), num_layers=1
                ),
            ]
        )

        self.linears = nn.ModuleList(
            [
                nn.Identity(54),
                nn.Linear(20, 20),
                nn.Bilinear(20, 20, 40),
                # nn.LazyLinear(20, 30),
            ]
        )

        self.shuffle = nn.ChannelShuffle(2)

    def forward(self):
        return [
            self.convolution_pooling_padding_ops(),
            self.activation_ops(),
            self.recurrent_ops(),
            self.transformer_ops(),
            self.linear_ops(),
            self.dropout_functions(),
            self.sparse_functions(),
            self.distance_functions(),
            # self.loss_functions(),
            self.vision_functions(),
            self.shuffle_functions(),
        ]

    def convolution_pooling_padding_ops(self):
        for i, module in enumerate(self.module1d):
            x = module(self.input1d)
        for i, module in enumerate(self.module2d):
            x = module(self.input2d)
        for i, module in enumerate(self.module3d):
            x = module(self.input3d)
        return x

    def activation_ops(self):
        input = torch.randn(2, 3, 4)
        for i, module in enumerate(self.activations):
            x = module(input)
        return x

    def recurrent_ops(self):
        input = torch.randn(5, 3, 10)
        h = torch.randn(2, 3, 20)
        c = torch.randn(2, 3, 20)
        x = self.rnn[0](input, h)
        x = self.rnn[1](input[0], h[0])
        x = self.gru[0](input, h)
        x = self.gru[1](input[0], h[0])
        x = self.lstm[0](input, (h, c))
        x = self.lstm[1](input[0], (h[0], c[0]))
        return x

    def transformer_ops(self):
        input = torch.rand(10, 16, 4)
        tgt = torch.rand((20, 16, 4))
        return [
            self.transformers[0](input, tgt),
            self.transformers[1](input),
            self.transformers[2](input, tgt),
        ]

    def linear_ops(self):
        input = torch.randn(32, 20)
        return [
            self.linears[0](input),
            self.linears[1](input),
            self.linears[2](input, input),
        ]

    def dropout_functions(self):
        a = torch.randn(8, 4)
        b = torch.randn(8, 4, 4, 4)
        c = torch.randn(8, 4, 4, 4, 4)
        return [
            F.dropout(a),
            F.dropout2d(b),
            F.dropout3d(c),
            F.alpha_dropout(a),
            F.feature_alpha_dropout(c),
        ]

    def sparse_functions(self):
        input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        input2 = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        embedding_matrix = torch.rand(10, 3)
        offsets = torch.tensor([0, 4])
        return [
            F.embedding(input, embedding_matrix),
            F.embedding_bag(input2, embedding_matrix, offsets),
            F.one_hot(torch.arange(0, 5) % 3, num_classes=5),
        ]

    def distance_functions(self):
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        return [
            F.pairwise_distance(a, b),
            F.cosine_similarity(a, b),
            F.pdist(a),
        ]

    def loss_functions(self):
        a = torch.randn(3, 2)
        b = torch.rand(3, 2)
        c = torch.rand(3)
        # log_probs = torch.randn(5, 3, 2).log_softmax(2).detach()
        return [
            # F.binary_cross_entropy(a, b),
            # F.binary_cross_entropy_with_logits(a, b),
            F.poisson_nll_loss(a, b),
            F.cosine_embedding_loss(a, b, c),
            F.cross_entropy(a, b),
            # F.ctc_loss(log_probs, a, c, c),
            # F.gaussian_nll_loss(a, b, torch.ones(5, 1)),
            # F.hinge_embedding_loss(a, b),
            F.kl_div(a, b),
            F.l1_loss(a, b),
            F.mse_loss(a, b),
            F.margin_ranking_loss(c, c, c),
            F.multilabel_margin_loss(a, b),
            F.multilabel_soft_margin_loss(a, b),
            F.multi_margin_loss(a, b, -b),
            F.nll_loss(a, b),
            F.huber_loss(a, b),
            F.smooth_l1_loss(a, b),
            F.soft_margin_loss(a, b),
            F.triplet_margin_loss(a, b, -b),
            # F.triplet_margin_with_distance_loss(a, b, -b),
        ]

    def vision_functions(self):
        return []

    def shuffle_functions(self):
        return self.shuffle(torch.randn(1, 4, 2, 2))
