import torch
from torch.autograd import Variable

from .module import Module


class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=-1,
            max_norm=None, norm_type=2, scale_grad_by_freq=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        weight_t = torch.DoubleTensor(num_embeddings, embedding_dim)
        self.weight = Variable(weight_t)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input):
        return self._backend.Embedding(self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq)(input, self.weight)


# TODO: SparseLinear

