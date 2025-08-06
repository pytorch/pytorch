We have the following forward function declaration:

```
def forward(q, k, v):
	x = torch.matmul(q, k.transpose(0, 1))
	a = torch.tanh(x)
	o = torch.matmul(a, v)
	return o, a
```

Per https://docs.pytorch.org/docs/stable/notes/extending.html, we will have the following backward function definition. Since the outputs are (o, a), we should receive (grad_o, grad_a) as inputs. Since the inputs are (q, k, v), we should have (grad_q, grad_k, grad_v) as outputs. 

```
def backward(grad_o, grad_a):
	...
	return grad_q, grad_k, grad_v
```

I have derived the following gradients
Note: Although grad_a is passed in as a parameter, we still must compute it to get the local gradient contribution.

```
o = av 
	=> grad_a = grad_o @ v^T # dL/da
	=> grad_v = grad_o^T @ a # dL/dv

a = tanh(x)
	=> grad_x = grad_a * (1 - a^2) # dL/dx

x = qk^T
	=> grad_q = grad_x @ k # dL/dq
	=> grad_k = grad_x^T @ q # dL/dk
```

