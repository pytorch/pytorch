import torch
import torch.nn as nn
import random

def test_unicode_error_on_extreme_values():
    class BuggyModel(nn.Module):
        def forward(self, x):
            if self.training:
                mask = torch.rand_like(x) < 0.1
                extreme_vals = torch.empty_like(x[mask])
                for i in range(extreme_vals.numel()):
                    extreme_vals[i] = random.choice([
                        float("nan"), float("inf"), float("-inf")
                    ])
                x[mask] = extreme_vals
            return x

    model = BuggyModel().train()
    x = torch.randn(1, 1, 32, 32)

    try:
        compiled = torch.compile(model)
        _ = compiled(x)
    except UnicodeDecodeError as e:
        assert False, f"torch.compile raised UnicodeDecodeError: {e}"
