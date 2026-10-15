# Autograd lab: Derive gradient formula

## Forward

```python
def attn(q, k, v):
    x = torch.matmul(q, k.transpose(0, 1))
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a
```

## Backward

```
grad_v = a^T @ grad_o
grad_x = (grad_a + grad_o @ v^T) * (1 - a^2)
grad_q = grad_x @ k
grad_k = grad_x^T @ q
```
