# Owner(s): ["module: unknown"]

import torch

x = torch.ones((3, 3), requires_grad=True)
(3 * x).sum().backward()
