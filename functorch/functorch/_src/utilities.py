import torch

def _prod(x):
    s = 1
    for i in x:
        s *= i
    return s


def _size_of(metadata):
    sizes = {
        torch.float: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int: 4,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.bool: 1,
    }

    numel = _prod(metadata.shape)
    dtype = metadata.dtype

    if dtype not in sizes:
        raise NotImplementedError("Don't know the size of dtype ", dtype)

    return numel * sizes[dtype]


def draw_nx_graph(nx_graph, filename = "fig.svg"):
    import matplotlib.pyplot as pyplot  # pylint: disable=import-error
    import networkx as nx 
    labels = nx.get_edge_attributes(nx_graph,'capacity')
    pos=nx.planar_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels = True)
    nx.draw_networkx_edge_labels(nx_graph,pos,edge_labels=labels)
    pyplot.savefig(filename)    