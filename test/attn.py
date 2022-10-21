from torch import autograd
import torch

# Inherit from Function
class AttentionFunction(autograd.Function):
    @staticmethod
    def attn(ctx, q, k, v):
        a = torch.tanh(torch.matmul(q, k.T))
        ctx.save_for_backward(q, k, v, a)
        o = torch.matmul(a, v)
        return o, a

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, q, k, v):
        assert len(q.size()) == 2, f"input q must be 2D but instead has {len(q.size())} dims"
        assert len(k.size()) == 2, f"input k must be 2D but instead has {len(k.size())} dims"
        assert len(v.size()) == 2, f"input v must be 2D but instead has {len(v.size())} dims"
        assert q.size(0) == k.size(0) and k.size(0) == v.size(0), "all inputs must have the same first dimension"
        assert q.size(1) == k.size(1), "q and k must share the same size for their second dimension"
        ctx.set_materialize_grads(False)
        return AttentionFunction.attn(ctx, q, k, v)

    # This function has TWO outputs, so it gets TWO gradients :D
    @staticmethod
    def backward(ctx, grad_o, grad_a):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        q, k, v, a = ctx.saved_tensors
        total_q = total_k = total_v = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        i = grad_o @ v.T
        j = 1 - a ** 2
        if ctx.needs_input_grad[0]:
            total_q = (i * j) @ k + (grad_a * j) @ k
        if ctx.needs_input_grad[1]:
            total_k = (i * j).T @ q + (grad_a * j).T @ q
        if ctx.needs_input_grad[2]:
            total_v = a.T @ grad_o

        return total_q, total_k, total_v

'''
For o + q: ((grad_o @ v^T) * (1 - tanh^2(x))) @ k
For o + k: ((grad_o @ v^T) * (1 - tanh^2(x)))^T @ q   ---> how come i don't need to transpose this at the end? nvm i get it
For o + v: grad_o^T @ a --> this would give a transposed result! so gotta transpose once more, or rewrite based on AB = (B^T A^T)^T

For a + q: (grad_a * (1 - tanh^2(x))) @ k
For a + k: (grad_a * ((1 - tanh^2(x))^T @ q     ---> how come i don't need to transpose this at the end? nvm i get it
For a + v: 0
'''

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
q = torch.rand(2, 3, requires_grad=True, dtype=torch.float64)
k = torch.rand(2, 3, requires_grad=True, dtype=torch.float64)
v = torch.rand(2, 4, requires_grad=True, dtype=torch.float64)
input = (q, k, v)
test = autograd.gradcheck(AttentionFunction.apply, input)
print(test)
