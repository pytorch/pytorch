import torch


@torch._dynamo.config.patch(allow_rnn=True)
def main():
    class LSTM(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.lstm = torch.nn.LSTM(2, 3, num_layers=2, bias=False)

        def forward(self, x, h0, c0):
            out, (_, _) = self.lstm(x, (h0, c0))
            return out

    torch._dynamo.config.recompile_limit = 1
    device = "cuda"
    lstm = LSTM(2).to(device)
    comp_lstm = torch.compile(lstm)

    x = torch.rand(3, 2).to(device)
    torch._dynamo.mark_dynamic(x, 0)
    h_0 = torch.rand(2, 3).to(device)
    c_0 = torch.rand(2, 3).to(device)

    base = lstm(x, h_0, c_0)
    output = comp_lstm(x, h_0, c_0)

    assert torch.all(torch.isclose(base, output, rtol=1e-7, atol=1e-7)), (
        f"Failed: base size {base.size()} vs output size {output.size()}; "
        + f"base checksum: {base.sum().item()} vs output sum: {output.sum().item()}"
    )
    print("Passed, yay!")
    x2 = torch.rand(8, 2).to(device)  # now try with different seqlen
    assert torch.all(
        torch.isclose(lstm(x2, h_0, c_0), comp_lstm(x2, h_0, c_0), rtol=1e-7, atol=1e-7)
    )


if __name__ == "__main__":
    main()
