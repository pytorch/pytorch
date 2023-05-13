# Gradient Formula

```python3

import torch

def attn(q, k, v):
    x = torch.matmul(q, k.transpose(1, 0))
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a

q = torch.rand(2, 3)
k = torch.rand(2, 3)
v = torch.rand(2, 4)
o, a = attn(q, k, v)
```

$$
\newcommand{\PD}[2]{ \frac{ \partial #1 }{ \partial #2 }}
\newcommand{\D}[2] {\frac{ \partial #1 }{ \partial #2 }}
$$

We know $ \PD{L}{o} $.
We need to find $ \PD{L}{a} \PD{L}{v} \PD{L}{x} \PD{L}{q} \PD{L}{k} $

$$
\PD{L}{a_{i, j}} = \sum_k \PD{L}{o_{i, k}} * b_{j, k} \\

\PD{L}{v_{i, j}} = \sum_k \PD{L}{o_{k, j}} * a_{k, i}
$$

$$
\D{tanh(x)}{x} = 1 / cosh^2(x) \\

\PD{L}{x} = \D{tanh(x)}{x} (x) * \D{L}{a} \\
    = \D{L}{a} * 1 / cosh^2(x)
$$

Now, we know $\PD{L}{x}$ which can be used to compute $\PD{L}{q}, \PD{L}{kt}$ where $kt$ is `k.transpose()`.
$$
\PD{L}{q_{i, j}} = \sum_k \PD{L}{x_{i, k}} * kt_{j, k} \\
\PD{L}{kt_{i, j}} = \sum_k \PD{L}{x_{k, j}} * q_{k, i} \\
\PD{L}{k_{i, j}} = \PD{L}{kt_{j, i}}
$$

# In psuedocode

```python
import torch
import torch.nn as nn

def attn(q, k, v):
    x = torch.matmul(q, k.transpose(1, 0))
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a

q = torch.rand(2, 3, requires_grad=True)
k = torch.rand(2, 3, requires_grad=True)
v = torch.rand(2, 4, requires_grad=True)

x = torch.matmul(q, k.transpose(1, 0))
a = torch.tanh(x)
o = torch.matmul(a, v)
# o, a = attn(q, k, v)

loss_fn = nn.MSELoss()

y_true = torch.zeros_like(o)
loss = loss_fn(o, y_true)

loss.backward()

dL_do = 2 * (o - y_true) / o.numel()

dL_da = dL_do.mm(v.transpose(1, 0))

dL_dv = a.transpose(1, 0).mm(dL_do)

dL_dx = dL_da / ((torch.cosh(x)) ** 2)

dL_dq = dL_dx.mm(k)
dL_dk = q.transpose(1, 0).mm(dL_dx).transpose(1, 0)
```