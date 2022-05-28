import torch

from .base_sparsifier import BaseSparsifier


class NearlyDiagonalSparsifier(BaseSparsifier):
    r"""Nearly Diagonal Sparsifier

    This sparsifier creates a nearly diagonal mask to be applied to the weight matrix. 
    Nearly Diagonal Matrix is a matrix that contains non-zero elements near the diagonal and the rest are zero. 
    An example of a nearly diagonal matrix with degree (or nearliness) 3 and 5 are follows respectively.
    1 1 0 0       1 1 1 0
    0 1 1 0       1 1 1 1
    0 1 1 1       1 1 1 1
    0 0 1 1       0 1 1 1
    Note that a nearly diagonal matrix with degree 1 is just a matrix with main diagonal populated

    This sparsifier is controlled by one variables:
    1. `nearliness` defines the number of non-zero diagonal lines that are closest to the main diagonal. 
        Currently - supports only odd number

    Args:
        nearliness: The degree of nearliness

    """
    def __init__(self, nearliness: int = 1):
        defaults = {'nearliness': nearliness}
        super().__init__(defaults=defaults)

    def update_mask(self, layer, nearliness: int,
                    **kwargs):
        mask = layer.parametrizations.weight[0].mask
        mask.data = torch.zeros_like(mask)
        if nearliness <= 0:
            return

        weight = layer.weight
        height, width = weight.shape

        if nearliness % 2 == 0:
            raise ValueError("nearliness can only be an odd number")
        dist_to_diagonal = nearliness // 2
        # check
        if dist_to_diagonal >= min(height, width):
            raise ValueError("nearliness cannot be larger than the "
                            "dimensions of weight matrix.")
        # create mask
        for row in range(0, height):
            for col in range(0, width):
                if abs(row - col) <= dist_to_diagonal:
                    mask[row, col] = 1
