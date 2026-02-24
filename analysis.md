# Analysis for Issue #175520: Question about torch._dynamo.export and nn.GRU support

**Issue**: https://github.com/pytorch/pytorch/issues/175520
**Category**: question
**Summary**: User asks if torch._dynamo.export not supporting nn.GRU is an intentional limitation or a bug

## Answer

This is a known, intentional limitation. `torch._dynamo.export` does not support `nn.RNN`, `nn.GRU`, or `nn.LSTM` modules. The error message explicitly states this: "Dynamo does not support RNN, GRU, or LSTM."

Note that `torch._dynamo.export` is a legacy/deprecated API. The recommended modern alternative is `torch.export.export`, which does support GRU and other RNN modules. You can use it like this:

```python
import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=80,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        out, h = self.gru(x)
        return out, h

model = SimpleGRU()
sample_input = torch.randn(2, 5, 80)

exported = torch.export.export(model, (sample_input,))
```

If you must use `torch._dynamo.export` for some reason, you can work around this by decomposing the GRU into basic tensor operations manually, but using `torch.export.export` is strongly recommended instead.

## Repro Code

```python
import torch
import torch.nn as nn
import torch._dynamo as dynamo

class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=80,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        out, h = self.gru(x)
        return out, h

model = SimpleGRU()
sample_input = torch.randn(2, 5, 80)

exported = dynamo.export(model)(sample_input)
```
