from typing import NamedTuple

import numpy as np
import torch


__all__ = ["Fuzzer", "FuzzedParameter", "FuzzedTensor"]


_DISTRIBUTIONS = (
    "loguniform",
    "uniform",
)

class FuzzedParameter(object):
    def __init__(self, name, minval=None, maxval=None, distribution=None, constraint=None):
        self._name = name
        self._minval = minval
        self._maxval = maxval
        self._distribution = self._check_distribution(distribution)
        self._contraint = constraint

    @property
    def name(self):
        return self._name

    def sample(self, state):
        if self._distribution == "loguniform":
            return self._loguniform(state)

        if self._distribution == "uniform":
            return self._uniform(state)

        if isinstance(self._distribution, dict):
            return self._custom_distribution(state)

    def satisfies_constraints(self, params):
        return self._contraint is None or self._contraint(**params)

    def _check_distribution(self, distribution):
        if not isinstance(distribution, dict):
            assert distribution in _DISTRIBUTIONS
        else:
            assert sum(distribution.values()) == 1, "Distribution is not normalized"
            assert self._minval is None
            assert self._maxval is None

        return distribution

    def _loguniform(self, state):
        output = int(2 ** state.uniform(
            low=np.log2(self._minval),
            high=np.log2(self._maxval)
        ))
        if output < self._minval:
            return self._minval
        if output > self._maxval:
            return self._maxval
        return output

    def _uniform(self, state):
        if isinstance(self._minval, int) and isinstance(self._maxval, int):
            return int(state.randint(low=self._minval, high=self._maxval + 1))
        return state.uniform(low=self._minval, high=self._maxval)

    def _custom_distribution(self, state):
        return state.choice(tuple(self._distribution.keys()), p=tuple(self._distribution.values()))


class FuzzedTensor(object):
    def __init__(self, name, size, probability_contiguous=0.5,
                 min_elements=None, max_elements=None,
                 dim_parameter=None, roll_parameter=None,
                 tensor_constructor=None):
        self._name = name
        self._size = size
        self._probability_contiguous = probability_contiguous
        self._min_elements = min_elements
        self._max_elements = max_elements
        self._dim_parameter = dim_parameter
        self._roll_parameter = roll_parameter
        self._tensor_constructor = tensor_constructor

    @property
    def name(self):
        return self._name

    def _make_tensor(self, params, state):
        size = self._get_concrete_size(params)
        dim = len(size)
        if self._tensor_constructor is None:
            tensor = torch.rand(size=size)
        else:
            tensor = self._tensor_constructor(size=size, **params)

        if self._roll_parameter is not None:
            assert params[self._roll_parameter] < dim
            rolled_order = tuple(np.roll(np.arange(dim), params[self._roll_parameter]))
            tensor = tensor.permute(rolled_order).contiguous()

        # Randomly permute the Tensor and call `.contiguous()` to force re-ordering
        # of the memory, and then permute it back to the original shape.
        order = np.arange(dim)
        if state.rand() > self._probability_contiguous:
            while dim > 1 and np.all(order == np.arange(dim)):
                order = state.permutation(tensor.dim())

            tensor = tensor.permute(tuple(order)).contiguous().permute(tuple(np.argsort(order)))

        properties = {
            "numel": int(tensor.numel()),
            "order": order,
            "is_contiguous": tensor.is_contiguous(),
            "dtype": str(tensor.dtype),
        }

        return tensor, properties

    def _get_concrete_size(self, params):
        size = tuple(self._size)
        if self._dim_parameter is not None:
            dim = params[self._dim_parameter]
            if len(size) > dim:
                size = size[:dim]
            if len(size) < dim:
                size = size + tuple(1 for _ in range(dim - len(size)))
        return tuple(params.get(i, i) for i in size)

    def satisfies_constraints(self, params):
        size = self._get_concrete_size(params)
        num_elements = np.prod(size)
        if self._max_elements is not None and num_elements > self._max_elements:
            return False
        if self._min_elements is not None and num_elements < self._min_elements:
            return False
        return True


class Fuzzer(object):
    def __init__(self, parameters, tensors, seed=None):
        if seed is None:
            seed = np.random.RandomState().randint(0, 2**63)
        self._seed = seed
        self._parameters = parameters
        self._tensors = tensors
        assert not len({p.name for p in parameters}.intersection({t.name for t in tensors}))

        self._rejections = 0
        self._total_generated = 0

    def take(self, n):
        state = np.random.RandomState(self._seed)
        torch.manual_seed(state.randint(low=0, high=2 ** 63))
        for _ in range(n):
            params = self._generate(state)
            tensors = {}
            tensor_properties = {}
            for t in self._tensors:
                tensor, properties = t._make_tensor(params, state)
                tensors[t.name] = tensor
                tensor_properties[t.name] = properties
            yield tensors, tensor_properties, params

    @property
    def rejection_rate(self):
        if not self._total_generated:
            return 0.
        return self._rejections / self._total_generated

    def _generate(self, state):
        for _ in range(1000):
            candidate_params = {}
            for p in self._parameters:
                candidate_params[p.name] = p.sample(state)

            self._total_generated += 1
            if not all(p.satisfies_constraints(candidate_params) for p in self._parameters):
                self._rejections += 1
                continue

            if not all(t.satisfies_constraints(candidate_params) for t in self._tensors):
                self._rejections += 1
                continue

            return candidate_params
        raise ValueError("Failed to generate a set of valid parameters.")
