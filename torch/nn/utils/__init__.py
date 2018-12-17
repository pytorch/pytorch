from . import rnn
from .clip_grad import clip_grad_norm, clip_grad_norm_, clip_grad_value_
from .weight_norm import weight_norm, remove_weight_norm
from .convert_parameters import parameters_to_vector, vector_to_parameters
from .spectral_norm import spectral_norm, remove_spectral_norm
from torch._C import _infer_size, _add_docstr, _nn

one_hot = _add_docstr(_nn.one_hot, r"""
one_hot(tensor, num_classes=0) -> LongTensor

Takes LongTensor with index values of shape ``(*)`` and returns a tensor
of shape ``(*, num_classes)`` that have zeros everywhere except where the
index of last dimension matches the corresponding value of the input tensor,
in which case it will be 1.

See also `One-hot on Wikipedia`_ .

.. _One-hot on Wikipedia:
    https://en.wikipedia.org/wiki/One-hot

Arguments:
    tensor (LongTensor): class values of any shape.
    num_classes (int):  Total number of classes. If set to -1, the actually
        number of classes will be inferred as the largest class value in the
        input tensor.

Returns:
    Tensor: LongTensor that has one more dimension with 1 values at the
        index of last dimension indicated by the input, and 0 everywhere
        else.

Examples::
    >>> torch.one_hot(torch.arange(0, 5) % 3)
    tensor([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]])
    >>> torch.one_hot(torch.arange(0, 5) % 3, num_classes=5)
    tensor([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]])
    >>> torch.one_hot(torch.arange(0, 6).view(3,2) % 3)
    tensor([[[1, 0, 0],
             [0, 1, 0]],

            [[0, 0, 1],
             [1, 0, 0]],

            [[0, 1, 0],
             [0, 0, 1]]])
""")
