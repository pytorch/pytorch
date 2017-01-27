import torch

from ..function import Function, InplaceFunction


# TODO: no need to save all args if the grad w.r.t. some of them is not needed
class _BlasBase(InplaceFunction):

    def __init__(self, alpha=1, beta=1, inplace=False):
        super(_BlasBase, self).__init__(inplace)
        self.alpha = alpha
        self.beta = beta

    def _get_output(self, arg):
        if self.inplace:
            self.mark_dirty(arg)
            return arg
        else:
            return arg.new().resize_as_(arg)


class Addmm(_BlasBase):

    def forward(self, add_matrix, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        output = self._get_output(add_matrix)
        return torch.addmm(self.alpha, add_matrix, self.beta,
                           matrix1, matrix2, out=output)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_add_matrix = grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_add_matrix = grad_output
            if self.alpha != 1:
                grad_add_matrix = grad_add_matrix.mul(self.alpha)

        if self.needs_input_grad[1]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())
            if self.beta != 1:
                grad_matrix1 *= self.beta

        if self.needs_input_grad[2]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)
            if self.beta != 1:
                grad_matrix2 *= self.beta

        return grad_add_matrix, grad_matrix1, grad_matrix2


class Addbmm(_BlasBase):

    def forward(self, add_matrix, batch1, batch2):
        self.save_for_backward(batch1, batch2)
        output = self._get_output(add_matrix)
        return torch.addbmm(self.alpha, add_matrix, self.beta,
                            batch1, batch2, out=output)

    def backward(self, grad_output):
        batch1, batch2 = self.saved_tensors
        grad_add_matrix = grad_batch1 = grad_batch2 = None

        if self.needs_input_grad[0]:
            grad_add_matrix = grad_output
            if self.alpha != 1:
                grad_add_matrix = grad_add_matrix.mul(self.alpha)

        if any(self.needs_input_grad[1:]):
            batch_grad_output = (grad_output
                                 .unsqueeze(0)
                                 .expand(batch1.size(0), batch1.size(1), batch2.size(2)))

        if self.needs_input_grad[1]:
            grad_batch1 = torch.bmm(batch_grad_output, batch2.transpose(1, 2))
            if self.beta != 1:
                grad_batch1 *= self.beta

        if self.needs_input_grad[2]:
            grad_batch2 = torch.bmm(batch1.transpose(1, 2), batch_grad_output)
            if self.beta != 1:
                grad_batch2 *= self.beta

        return grad_add_matrix, grad_batch1, grad_batch2


class Baddbmm(_BlasBase):

    def forward(self, add_batch, batch1, batch2):
        self.save_for_backward(batch1, batch2)
        output = self._get_output(add_batch)
        return torch.baddbmm(self.alpha, add_batch, self.beta,
                             batch1, batch2, out=output)

    def backward(self, grad_output):
        batch1, batch2 = self.saved_tensors
        grad_add_batch = grad_batch1 = grad_batch2 = None

        if self.needs_input_grad[0]:
            grad_add_batch = grad_output
            if self.alpha != 1:
                grad_add_batch = grad_add_batch.mul(self.alpha)

        if self.needs_input_grad[1]:
            grad_batch1 = torch.bmm(grad_output, batch2.transpose(1, 2))
            if self.beta != 1:
                grad_batch1 *= self.beta

        if self.needs_input_grad[2]:
            grad_batch2 = torch.bmm(batch1.transpose(1, 2), grad_output)
            if self.beta != 1:
                grad_batch2 *= self.beta

        return grad_add_batch, grad_batch1, grad_batch2


class Addmv(_BlasBase):

    def forward(self, add_vector, matrix, vector):
        self.save_for_backward(matrix, vector)
        output = self._get_output(add_vector)
        return torch.addmv(self.alpha, add_vector, self.beta,
                           matrix, vector, out=output)

    def backward(self, grad_output):
        matrix, vector = self.saved_tensors
        grad_add_vector = grad_matrix = grad_vector = None

        if self.needs_input_grad[0]:
            grad_add_vector = grad_output
            if self.alpha != 1:
                grad_add_vector = grad_add_vector.mul(self.alpha)

        if self.needs_input_grad[1]:
            grad_matrix = torch.ger(grad_output, vector)
            if self.beta != 1:
                grad_matrix *= self.beta

        if self.needs_input_grad[2]:
            grad_vector = torch.mv(matrix.t(), grad_output)
            if self.beta != 1:
                grad_vector *= self.beta

        return grad_add_vector, grad_matrix, grad_vector


class Addr(_BlasBase):

    def forward(self, add_matrix, vector1, vector2):
        self.save_for_backward(vector1, vector2)
        output = self._get_output(add_matrix)
        return torch.addr(self.alpha, add_matrix, self.beta,
                          vector1, vector2, out=output)

    def backward(self, grad_output):
        vector1, vector2 = self.saved_tensors
        grad_add_matrix = grad_vector1 = grad_vector2 = None

        if self.needs_input_grad[0]:
            grad_add_matrix = grad_output
            if self.alpha != 1:
                grad_add_matrix = grad_add_matrix.mul(self.alpha)

        if self.needs_input_grad[1]:
            grad_vector1 = torch.mv(grad_output, vector2)
            if self.beta != 1:
                grad_vector1 *= self.beta

        if self.needs_input_grad[2]:
            # TODO: maybe it's better to do transpose + mv + transpose
            grad_vector2 = torch.mm(vector1.unsqueeze(0), grad_output)
            if self.beta != 1:
                grad_vector2 *= self.beta

        return grad_add_matrix, grad_vector1, grad_vector2


class Dot(Function):

    def forward(self, vector1, vector2):
        self.save_for_backward(vector1, vector2)
        return vector1.new((vector1.dot(vector2),))

    def backward(self, grad_output):
        vector1, vector2 = self.saved_tensors
        grad_vector1 = grad_vector2 = None

        if self.needs_input_grad[0]:
            grad_vector1 = vector2.mul(grad_output[0])

        if self.needs_input_grad[1]:
            grad_vector2 = vector1.mul(grad_output[0])

        return grad_vector1, grad_vector2


# TODO: cross
# TODO: diag
# TODO: trace
# TODO: tril
# TODO: triu
