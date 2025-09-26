import random

class Tensor:
    def __init__(self, size, stride, dtype, device, supported_ops, requires_grad=None):
        self.size = size
        self.stride = stride
        self.dtype = dtype
        self.device = device
        self.supported_ops = supported_ops
        self.requires_grad = requires_grad  # None means use default behavior, True/False override
        # Add optional attributes for cat operation - allow int values
        self._cat_dim = None  # type: int | None
        self._cat_sizes = None  # type: tuple | None
        # For view/sum/fill_diagonal
        self._view_shape = None  # type: tuple | None
        self._sum_dim = None  # type: int | tuple | str | None
        self._fill_diag_dims = None  # type: tuple | None

    def decompose(self):
        from operators import AddOperator, CatOperator

        candidates = []
        for op in self.supported_ops:
            if op.can_produce(self):
                candidates.append(op)
        if not candidates:
            return []
        candidate = random.choice(candidates)
        # --- CHANGED: pass random number of inputs for Add/Cat ---
        if isinstance(candidate, AddOperator) or isinstance(candidate, CatOperator):
            min_inputs = 2
            max_inputs = 5
            num_inputs = random.randint(min_inputs, max_inputs)
            return candidate.decompose(self, num_inputs=num_inputs)
        return candidate.decompose(self)
