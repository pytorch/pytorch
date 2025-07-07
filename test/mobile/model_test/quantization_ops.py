import torch
import torch.nn as nn


class GeneralQuantModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.ao.nn.quantized.Embedding(
            num_embeddings=10, embedding_dim=12
        )
        self.embedding_input = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        self.func = torch.ao.nn.quantized.QFunctional()
        self.conv1 = torch.ao.nn.quantized.ConvTranspose1d(16, 33, 3, stride=2)
        self.conv2 = torch.ao.nn.quantized.ConvTranspose2d(16, 33, 3, stride=2)
        self.conv3 = torch.ao.nn.quantized.ConvTranspose3d(16, 33, 3, stride=2)

    def forward(self):
        a = torch.quantize_per_tensor(torch.tensor([3.0]), 1.0, 0, torch.qint32)
        b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        c = torch.quantize_per_tensor(
            torch.tensor([3.0]), torch.tensor(1.0), torch.tensor(0), torch.qint32
        )
        input1 = torch.randn(1, 16, 4)
        input2 = torch.randn(1, 16, 4, 4)
        return len(
            self.func.add(a, b),
            self.func.cat((a, a), 0),
            self.func.mul(a, b),
            self.func.add_relu(a, b),
            self.func.add_scalar(a, b),
            self.func.mul_scalar(a, b),
            self.embedding(self.embedding_input),
            self.conv1(
                torch.quantize_per_tensor(
                    input1, scale=1.0, zero_point=0, dtype=torch.quint8
                )
            ),
            self.conv2(
                torch.quantize_per_tensor(
                    input2, scale=1.0, zero_point=0, dtype=torch.quint8
                )
            ),
            c,
            # self.conv3(torch.quantize_per_tensor(input3, scale=1.0, zero_point=0, dtype=torch.quint8)), # failed on iOS
        )


class DynamicQuantModule:
    def __init__(self) -> None:
        super().__init__()
        self.module = self.M()

    def getModule(self):
        return torch.ao.quantization.quantize_dynamic(self.module, dtype=torch.qint8)

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super(DynamicQuantModule.M, self).__init__()
            self.rnn = nn.RNN(4, 8, 2)
            self.rnncell = nn.RNNCell(4, 8)
            self.gru = nn.GRU(4, 8, 2)
            self.grucell = nn.GRUCell(4, 8)
            self.lstm = nn.LSTM(4, 8, 2)
            self.lstmcell = nn.LSTMCell(4, 8)
            self.linears = nn.ModuleList(
                [
                    nn.Identity(54),
                    nn.Linear(20, 20),
                    nn.Bilinear(20, 20, 40),
                ]
            )
            self.transformers = nn.ModuleList(
                [
                    nn.Transformer(
                        d_model=2, nhead=2, num_encoder_layers=1, num_decoder_layers=1
                    ),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=2, nhead=2), num_layers=1
                    ),
                    nn.TransformerDecoder(
                        nn.TransformerDecoderLayer(d_model=2, nhead=2), num_layers=1
                    ),
                ]
            )
            # self.a = torch.nn.utils.rnn.pad_sequence([torch.tensor([1,2,3]), torch.tensor([3,4])], batch_first=True)

        def forward(self):
            input = torch.randn(5, 3, 4)
            h = torch.randn(2, 3, 8)
            c = torch.randn(2, 3, 8)
            linear_input = torch.randn(32, 20)
            trans_input = torch.randn(1, 16, 2)
            tgt = torch.rand(1, 16, 2)

            return len(
                (
                    self.rnn(input, h),
                    self.rnncell(input[0], h[0]),
                    self.gru(input, h),
                    self.grucell(input[0], h[0]),
                    self.lstm(input, (h, c)),
                    # self.lstm(torch.nn.utils.rnn.pack_padded_sequence(self.a, lengths=torch.tensor([3,2,1])), (h, c)),
                    self.lstmcell(input[0], (h[0], c[0])),
                    self.transformers[0](trans_input, tgt),
                    self.transformers[1](trans_input),
                    self.transformers[2](trans_input, tgt),
                    self.linears[0](linear_input),
                    self.linears[1](linear_input),
                    self.linears[2](linear_input, linear_input),
                )
            )


class StaticQuantModule:
    def getModule(self):
        model_fp32 = self.M()
        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        return model_int8

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super(StaticQuantModule.M, self).__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.input1d = torch.randn(4, 2, 2)
            self.input2d = torch.randn((4, 2, 4, 4))
            self.input3d = torch.randn(4, 2, 2, 4, 4)
            self.linear_input = torch.randn(32, 20)

            self.layer1 = nn.Sequential(
                nn.Conv1d(2, 2, 1), nn.InstanceNorm1d(1), nn.Hardswish()
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(2, 2, 1),
                nn.BatchNorm2d(2),
                nn.InstanceNorm2d(1),
                nn.LeakyReLU(),
            )
            self.layer3 = nn.Sequential(
                nn.Conv3d(2, 2, 1), nn.BatchNorm3d(2), nn.InstanceNorm3d(1), nn.ReLU()
            )
            self.layer4 = nn.Sequential(nn.Linear(4, 3))
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self):
            x = self.quant(self.input1d)
            x = self.layer1(x)
            x = self.dequant(x)

            y = self.input2d
            y = self.quant(y)
            y = self.layer2(y)
            y = self.layer4(y)
            y = self.dequant(y)

            z = self.quant(self.input3d)
            z = self.layer3(z)
            z = self.dequant(z)

            return (x, y, z)


class FusedQuantModule:
    def getModule(self):
        model_fp32 = self.M()
        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_fused = torch.ao.quantization.fuse_modules(
            model_fp32,
            [
                ["conv1d", "relu1"],
                ["conv2d", "relu2"],
                ["conv3d", "relu3"],
                ["linear", "relu4"],
            ],
        )
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        return model_int8

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super(FusedQuantModule.M, self).__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.input1d = torch.randn(4, 2, 2)
            self.input2d = torch.randn((4, 2, 4, 4))
            self.input3d = torch.randn(4, 2, 2, 4, 4)
            self.conv1d = nn.Conv1d(2, 2, 1)
            self.conv2d = nn.Conv2d(2, 2, 1)
            self.conv3d = nn.Conv3d(2, 2, 1)
            self.linear = nn.Linear(4, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self):
            x = self.input1d
            y = self.input2d
            z = self.input3d

            x = self.quant(x)
            x = self.conv1d(x)
            x = self.relu1(x)
            x = self.dequant(x)

            y = self.quant(y)
            y = self.conv2d(y)
            y = self.relu2(y)
            y = self.dequant(y)

            z = self.quant(z)
            z = self.conv3d(z)
            z = self.relu3(z)
            z = self.linear(z)
            z = self.relu4(z)
            z = self.dequant(z)

            return (x, y, z)
