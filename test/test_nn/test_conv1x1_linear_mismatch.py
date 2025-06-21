import torch
import torch.nn as nn

def test_conv1x1_linear_mismatch():
    print("✅ Test file is running...")

    conv = nn.Conv2d(4, 5, 1)
    linear = nn.Linear(4, 5)

    linear.weight.data = conv.weight.data.view(5, 4)
    linear.bias.data = conv.bias.data

    x = torch.randn(2, 4, 1, 1)

    out_conv = conv(x).view(2, 5)
    out_linear = linear(x.view(2, 4))

    print("Conv output:\n", out_conv)
    print("Linear output:\n", out_linear)

    torch.testing.assert_close(out_conv, out_linear, rtol=1e-5, atol=1e-7)

