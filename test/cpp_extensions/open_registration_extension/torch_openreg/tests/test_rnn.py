# Owner(s): ["module: PrivateUse1"]

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLSTM(TestCase):
    def test_lstm_basic(self):
        input_size, hidden_size, num_layers = 10, 20, 1
        seq_len, batch_size = 5, 2

        lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        lstm_openreg = lstm.to("openreg")
        x = torch.randn(seq_len, batch_size, input_size, device="openreg")

        output, (h_n, c_n) = lstm_openreg(x)

        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(h_n.device.type, "openreg")
        self.assertEqual(c_n.device.type, "openreg")
        self.assertEqual(output.shape, torch.Size([seq_len, batch_size, hidden_size]))
        self.assertEqual(h_n.shape, torch.Size([num_layers, batch_size, hidden_size]))
        self.assertEqual(c_n.shape, torch.Size([num_layers, batch_size, hidden_size]))

    def test_lstm_multi_layer(self):
        input_size, hidden_size, num_layers = 10, 20, 3
        seq_len, batch_size = 5, 2

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to("openreg")
        x = torch.randn(seq_len, batch_size, input_size, device="openreg")

        output, (h_n, c_n) = lstm(x)

        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, torch.Size([seq_len, batch_size, hidden_size]))
        self.assertEqual(h_n.shape, torch.Size([num_layers, batch_size, hidden_size]))
        self.assertEqual(c_n.shape, torch.Size([num_layers, batch_size, hidden_size]))

    def test_lstm_bidirectional(self):
        input_size, hidden_size, num_layers = 10, 20, 2
        seq_len, batch_size = 5, 2
        num_directions = 2

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        ).to("openreg")
        x = torch.randn(seq_len, batch_size, input_size, device="openreg")

        output, (h_n, c_n) = lstm(x)

        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(
            output.shape,
            torch.Size([seq_len, batch_size, hidden_size * num_directions]),
        )
        self.assertEqual(
            h_n.shape,
            torch.Size([num_layers * num_directions, batch_size, hidden_size]),
        )
        self.assertEqual(
            c_n.shape,
            torch.Size([num_layers * num_directions, batch_size, hidden_size]),
        )

    def test_lstm_batch_first(self):
        input_size, hidden_size = 10, 20
        seq_len, batch_size = 5, 2

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        ).to("openreg")
        x = torch.randn(batch_size, seq_len, input_size, device="openreg")

        output, (h_n, c_n) = lstm(x)

        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, torch.Size([batch_size, seq_len, hidden_size]))

    def test_lstm_with_initial_hidden(self):
        input_size, hidden_size, num_layers = 10, 20, 1
        seq_len, batch_size = 5, 2

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to("openreg")
        x = torch.randn(seq_len, batch_size, input_size, device="openreg")
        h0 = torch.randn(num_layers, batch_size, hidden_size, device="openreg")
        c0 = torch.randn(num_layers, batch_size, hidden_size, device="openreg")

        output, (h_n, c_n) = lstm(x, (h0, c0))

        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(h_n.device.type, "openreg")
        self.assertEqual(c_n.device.type, "openreg")

    def test_lstm_packed_sequence(self):
        input_size, hidden_size = 10, 20

        lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size).to("openreg")

        sequences = [
            torch.randn(5, input_size),
            torch.randn(3, input_size),
            torch.randn(1, input_size),
        ]
        packed = nn.utils.rnn.pack_sequence(sequences, enforce_sorted=True)
        packed_openreg = packed.to("openreg")

        output_packed, (h_n, c_n) = lstm(packed_openreg)

        self.assertEqual(output_packed.data.device.type, "openreg")
        self.assertEqual(h_n.device.type, "openreg")
        self.assertEqual(c_n.device.type, "openreg")

    def test_lstm_cpu_parity(self):
        input_size, hidden_size, num_layers = 8, 16, 2
        seq_len, batch_size = 4, 3

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )

        x_cpu = torch.randn(seq_len, batch_size, input_size)
        output_cpu, (h_cpu, c_cpu) = lstm(x_cpu)

        lstm_openreg = lstm.to("openreg")
        x_openreg = x_cpu.to("openreg")
        output_openreg, (h_openreg, c_openreg) = lstm_openreg(x_openreg)

        self.assertEqual(output_cpu, output_openreg.cpu())
        self.assertEqual(h_cpu, h_openreg.cpu())
        self.assertEqual(c_cpu, c_openreg.cpu())


if __name__ == "__main__":
    run_tests()
