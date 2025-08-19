import torch


@torch._dynamo.config.patch(allow_rnn=True)
def main():
    class LSTM(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.lstm = torch.nn.LSTM(h, h)

        def forward(self, x, h0, c0):
            out, (_, _) = self.lstm(x, (h0, c0))
            return out

    torch._dynamo.config.recompile_limit = 1
    lstm = LSTM(32)
    comp_lstm = torch.compile(lstm)

    x = torch.rand(5, 32)
    torch._dynamo.mark_dynamic(x, 0)
    h_0 = torch.rand(1, 32)
    c_0 = torch.rand(1, 32)

    base = lstm(x, h_0, c_0)
    output = comp_lstm(x, h_0, c_0)

    assert torch.all(torch.isclose(base, output, rtol=1e-4, atol=1e-7)), (
        f"Failed: base size {base.size()} vs output size {output.size()}; base checksum: {base.sum().item()} vs output sum: {output.sum().item()}"
    )
    print("Passed, yay!")
    x2 = torch.rand(8, 32)  # now try with different seqlen
    assert torch.all(
        torch.isclose(lstm(x2, h_0, c_0), comp_lstm(x2, h_0, c_0), rtol=1e-4, atol=1e-7)
    )


if __name__ == "__main__":
    main()
