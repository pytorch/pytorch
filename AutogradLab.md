## Autograd Lab Derivation

Initial function provided is:
```
def attn(Q, K, V):
  x = torch.matmul(Q, K.t())
  a = torch.tanh(x)
  o = torch.matmul(a, V)

  return o, a
```

`a` is a useful signal, giving an intermediate output, but for autograd purposes should be ignored.

Taking the standard derivatives of `matmul`:
```
y = matmul(A, B)

grad_A = matmul(grad_y, B^T)
grad_B = matmul(A^T, grad_y)
```

We can step through the initial function op-by-op - first `o = torch.matmul(a, V)`:
```
# For the next op in line
grad_a = matmul(grad_o, V^T)
# one of my input gradients
grad_V = matmul(a^T, grad_o)
```

Next we need to differentiate through `a = torch.tanh(x)` - if `f = tanh(x)` then `f' = 1 - x^2`, so applying the chain rule:
```
grad_x = grad_a * (1 - a**2)
```

`grad_x` here is the final intermediate gradient we need - we can now use this to compute the final derivatives, `grad_Q` and `grad_K` which are originally computed via. `x = matmul(Q, K^T)`:
```
grad_Q = matmul(grad_x, (K^T)^T)
       = matmul(grad_x, K^T)

grad_(K^T) = matmul(Q^T, grad_x)
grad_K = [matmul(Q^T, grad_x)^T]
```


