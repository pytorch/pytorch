from .base_sparsifier import BaseSparsifier

def _flat_idx_to_2d(idx, shape):
    rows = idx // shape[1]
    cols = idx % shape[1]
    return rows, cols

class WeightNormSparsifier(BaseSparsifier):
    def __init__(self, config, sparsity_level=0.5, sparsity_pattern=(1, 4)):
        defaults = {
            'sparsity_level': sparsity_level,
            'sparsity_pattern': sparsity_pattern
        }
        super().__init__(config, defaults)

    def update_mask(self, layer, sparsity_level, sparsity_pattern, **kwargs):
        if sparsity_level <= 0:
            layer.mask = torch.ones(layer.weight.shape)
        elif sparsity_level >= 1.0:
            layer.mask = torch.zeros(layer.weight.shape)
        else:
            ww = layer.weight * layer.weight
            ww = ww.reshape(1, *ww.shape)
            ww_pool = F.avg_pool(ww, kernel_size=sparsity_pattern,
                                   stride=sparsity_pattern, ceil_mode=True)
            ww_pool_flat = ww_pool.flatten()
            _, sorted_idx = torch.sort(ww_pool_flat)
            threshold_idx = int(round(sparsity_level * len(sorted_idx)))
            sorted_idx = sorted_idx[:threshold_idx]
            rows, cols = _flat_idx_to_2d(sorted_idx, ww_pool.shape[1:])
            rows *= sparsity_pattern[0]
            cols *= sparsity_pattern[1]

            new_mask = torch.ones(ww.shape)
            for row, col in zip(rows, cols):
                new_mask[row:row+sparsity_pattern[0], col:col+sparsity_pattern[1]] = 0
            layer.mask *= new_mask
