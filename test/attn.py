from torch import autograd
import torch

# Inherit from Function
class AttentionFunction(autograd.Function):
    @staticmethod
    def attn(ctx, q, k, v):
        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        ctx.save_for_backward(q, k, v, x, a)
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
        return AttentionFunction.attn(ctx, q, k, v)

    # This function has TWO outputs, so it gets TWO gradients :D
    @staticmethod
    def backward(ctx, grad_o, grad_a):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        q, k, v, x, a = ctx.saved_tensors
        total_q = total_k = total_v = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_oq = ((grad_o @ v.transpose(0, 1)) * (1 - torch.tanh(x) ** 2)) @ k
            grad_aq = (grad_a * (1 - torch.tanh(x) ** 2)) @ k
            total_q = grad_oq + grad_aq
        if ctx.needs_input_grad[1]:
            # I used to have these be transposed at the end but that didn't pass the tests
            grad_ok = ((grad_o @ v.transpose(0, 1)) * (1 - torch.tanh(x) ** 2)).transpose(0, 1) @ q
            grad_ak = (grad_a * (1 - torch.tanh(x) ** 2)).transpose(0, 1) @ q
            total_k = grad_ok + grad_ak
        if ctx.needs_input_grad[2]:
            total_v = a.transpose(0, 1) @ grad_o

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
test = autograd.gradcheck(AttentionFunction.apply, input, eps=1e-6, atol=1e-4)
print(test)
