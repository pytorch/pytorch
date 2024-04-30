import torch
from torch.library import register_fake


@register_fake("aten::sparse_dim")
def sparse_dim_abstract(x):
    if x.layout is torch.sparse_coo:
        return x._indices().shape[0]
    if x.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        return 2
    return 0


@register_fake("aten::dense_dim")
def dense_dim_abstract(x):
    if x.layout is torch.sparse_coo:
        return x._values().ndim - 1
    if x.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        return x.values().ndim - 1
    return x.ndim
