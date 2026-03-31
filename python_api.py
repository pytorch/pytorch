import torch
from torch.autograd import gradcheck

class attn(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, q, k, v):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache tensors for
        use in the backward pass using the ``ctx.save_for_backward`` method. Other
        objects can be stored directly as attributes on the ctx object, such as
        ``ctx.my_object = my_object``. Check out `Extending torch.autograd <https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>`_
        for further details.
        """
        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)
        ctx.save_for_backward(q, k, v, a)
        return o, a

    @staticmethod
    def backward(ctx, grad_o, grad_a_extra):
        """
        In the backward implementation we receive the downstream gradients and deduce
        dL/dq, dL/dk, dL/dv from the received gradients using the chain rule plus reuse 
        of the intermediate value for `a`.
        """
        # 1. Retrieve saved tensors
        q, k, v, a = ctx.saved_tensors

        # 2. Gradient of o = matmul(a, v)
        # dL/da = grad_o @ v.T
        grad_a = torch.matmul(grad_o, v.transpose(0, 1))
        # dL/dv = a.T @ grad_o
        grad_v = torch.matmul(a.transpose(0, 1), grad_o)

        # 3. Combine with gradient from the second output 'a' (if any)
        if grad_a_extra is not None:
            grad_a = grad_a + grad_a_extra

        # 4. Gradient of a = tanh(x)
        # dL/dx = grad_a * (1 - a^2)
        grad_x = grad_a * (1 - a * a)

        # 5. Gradient of x = matmul(q, k.T)
        # dL/dq = grad_x @ k
        grad_q = torch.matmul(grad_x, k)
        # dL/dk = grad_x.T @ q
        grad_k = torch.matmul(grad_x.transpose(0, 1), q)

        return grad_q, grad_k, grad_v


if __name__ == "__main__":
    q = torch.rand(2, 3, dtype=torch.double, requires_grad=True)
    k = torch.rand(2, 3, dtype=torch.double, requires_grad=True)
    v = torch.rand(2, 4, dtype=torch.double, requires_grad=True)
    inputs = (q, k, v)
    test = gradcheck(attn.apply, inputs, eps=1e-6, atol=1e-4)
    print(f"Gradcheck passed: {test}")
