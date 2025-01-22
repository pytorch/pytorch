# mypy: allow-untyped-defs
import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


__all__ = [
    "Fuzzer",
    "FuzzedParameter", "ParameterAlias",
    "FuzzedTensor",
]


_DISTRIBUTIONS = (
    "loguniform",
    "uniform",
)


class FuzzedParameter:
    """Specification for a parameter to be generated during fuzzing."""
    def __init__(
        self,
        name: str,
        minval: Optional[Union[int, float]] = None,
        maxval: Optional[Union[int, float]] = None,
        distribution: Optional[Union[str, Dict[Any, float]]] = None,
        strict: bool = False,
    ):
        """
        Args:
            name:
                A string name with which to identify the parameter.
                FuzzedTensors can reference this string in their
                specifications.
            minval:
                The lower bound for the generated value. See the description
                of `distribution` for type behavior.
            maxval:
                The upper bound for the generated value. Type behavior is
                identical to `minval`.
            distribution:
                Specifies the distribution from which this parameter should
                be drawn. There are three possibilities:
                    - "loguniform"
                        Samples between `minval` and `maxval` (inclusive) such
                        that the probabilities are uniform in log space. As a
                        concrete example, if minval=1 and maxval=100, a sample
                        is as likely to fall in [1, 10) as it is [10, 100].
                    - "uniform"
                        Samples are chosen with uniform probability between
                        `minval` and `maxval` (inclusive). If either `minval`
                        or `maxval` is a float then the distribution is the
                        continuous uniform distribution; otherwise samples
                        are constrained to the integers.
                    - dict:
                        If a dict is passed, the keys are taken to be choices
                        for the variables and the values are interpreted as
                        probabilities. (And must sum to one.)
                If a dict is passed, `minval` and `maxval` must not be set.
                Otherwise, they must be set.
            strict:
                If a parameter is strict, it will not be included in the
                iterative resampling process which Fuzzer uses to find a
                valid parameter configuration. This allows an author to
                prevent skew from resampling for a given parameter (for
                instance, a low size limit could inadvertently bias towards
                Tensors with fewer dimensions) at the cost of more iterations
                when generating parameters.
        """
        self._name = name
        self._minval = minval
        self._maxval = maxval
        self._distribution = self._check_distribution(distribution)
        self.strict = strict

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

    def _check_distribution(self, distribution):
        if not isinstance(distribution, dict):
            assert distribution in _DISTRIBUTIONS
        else:
            assert not any(i < 0 for i in distribution.values()), "Probabilities cannot be negative"
            assert abs(sum(distribution.values()) - 1) <= 1e-5, "Distribution is not normalized"
            assert self._minval is None
            assert self._maxval is None

        return distribution

    def _loguniform(self, state):
        import numpy as np
        output = int(2 ** state.uniform(
            low=np.log2(self._minval) if self._minval is not None else None,
            high=np.log2(self._maxval) if self._maxval is not None else None,
        ))
        if self._minval is not None and output < self._minval:
            return self._minval
        if self._maxval is not None and output > self._maxval:
            return self._maxval
        return output

    def _uniform(self, state):
        if isinstance(self._minval, int) and isinstance(self._maxval, int):
            return int(state.randint(low=self._minval, high=self._maxval + 1))
        return state.uniform(low=self._minval, high=self._maxval)

    def _custom_distribution(self, state):
        import numpy as np
        # If we directly pass the keys to `choice`, numpy will convert
        # them to numpy dtypes.
        index = state.choice(
            np.arange(len(self._distribution)),
            p=tuple(self._distribution.values()))
        return list(self._distribution.keys())[index]


class ParameterAlias:
    """Indicates that a parameter should alias the value of another parameter.

    When used in conjunction with a custom distribution, this allows fuzzed
    tensors to represent a broader range of behaviors. For example, the
    following sometimes produces Tensors which broadcast:

    Fuzzer(
        parameters=[
            FuzzedParameter("x_len", 4, 1024, distribution="uniform"),

            # `y` will either be size one, or match the size of `x`.
            FuzzedParameter("y_len", distribution={
                0.5: 1,
                0.5: ParameterAlias("x_len")
            }),
        ],
        tensors=[
            FuzzedTensor("x", size=("x_len",)),
            FuzzedTensor("y", size=("y_len",)),
        ],
    )

    Chains of alias' are allowed, but may not contain cycles.
    """
    def __init__(self, alias_to):
        self.alias_to = alias_to

    def __repr__(self):
        return f"ParameterAlias[alias_to: {self.alias_to}]"


def dtype_size(dtype):
    if dtype == torch.bool:
        return 1
    if dtype.is_floating_point or dtype.is_complex:
        return int(torch.finfo(dtype).bits / 8)
    return int(torch.iinfo(dtype).bits / 8)


def prod(values, base=1):
    """np.prod can overflow, so for sizes the product should be done in Python.

    Even though np.prod type promotes to int64, it can still overflow in which
    case the negative value will pass the size check and OOM when attempting to
    actually allocate the Tensor.
    """
    return functools.reduce(lambda x, y: int(x) * int(y), values, base)


class FuzzedTensor:
    def __init__(
        self,
        name: str,
        size: Tuple[Union[str, int], ...],
        steps: Optional[Tuple[Union[str, int], ...]] = None,
        probability_contiguous: float = 0.5,
        min_elements: Optional[int] = None,
        max_elements: Optional[int] = None,
        max_allocation_bytes: Optional[int] = None,
        dim_parameter: Optional[str] = None,
        roll_parameter: Optional[str] = None,
        dtype=torch.float32,
        cuda=False,
        tensor_constructor: Optional[Callable] = None
    ):
        """
        Args:
            name:
                A string identifier for the generated Tensor.
            size:
                A tuple of integers or strings specifying the size of the generated
                Tensor. String values will replaced with a concrete int during the
                generation process, while ints are simply passed as literals.
            steps:
                An optional tuple with the same length as `size`. This indicates
                that a larger Tensor should be allocated, and then sliced to
                produce the generated Tensor. For instance, if size is (4, 8)
                and steps is (1, 4), then a tensor `t` of size (4, 32) will be
                created and then `t[:, ::4]` will be used. (Allowing one to test
                Tensors with strided memory.)
            probability_contiguous:
                A number between zero and one representing the chance that the
                generated Tensor has a contiguous memory layout. This is achieved by
                randomly permuting the shape of a Tensor, calling `.contiguous()`,
                and then permuting back. This is applied before `steps`, which can
                also cause a Tensor to be non-contiguous.
            min_elements:
                The minimum number of parameters that this Tensor must have for a
                set of parameters to be valid. (Otherwise they are resampled.)
            max_elements:
                Like `min_elements`, but setting an upper bound.
            max_allocation_bytes:
                Like `max_elements`, but for the size of Tensor that must be
                allocated prior to slicing for `steps` (if applicable). For
                example, a FloatTensor with size (1024, 1024) and steps (4, 4)
                would have 1M elements, but would require a 64 MB allocation.
            dim_parameter:
                The length of `size` and `steps` will be truncated to this value.
                This allows Tensors of varying dimensions to be generated by the
                Fuzzer.
            dtype:
                The PyTorch dtype of the generated Tensor.
            cuda:
                Whether to place the Tensor on a GPU.
            tensor_constructor:
                Callable which will be used instead of the default Tensor
                construction method. This allows the author to enforce properties
                of the Tensor (e.g. it can only have certain values). The dtype and
                concrete shape of the Tensor to be created will be passed, and
                concrete values of all parameters will be passed as kwargs. Note
                that transformations to the result (permuting, slicing) will be
                performed by the Fuzzer; the tensor_constructor is only responsible
                for creating an appropriately sized Tensor.
        """
        self._name = name
        self._size = size
        self._steps = steps
        self._probability_contiguous = probability_contiguous
        self._min_elements = min_elements
        self._max_elements = max_elements
        self._max_allocation_bytes = max_allocation_bytes
        self._dim_parameter = dim_parameter
        self._dtype = dtype
        self._cuda = cuda
        self._tensor_constructor = tensor_constructor

    @property
    def name(self):
        return self._name

    @staticmethod
    def default_tensor_constructor(size, dtype, **kwargs):
        if dtype.is_floating_point or dtype.is_complex:
            return torch.rand(size=size, dtype=dtype, device="cpu")
        else:
            return torch.randint(1, 127, size=size, dtype=dtype, device="cpu")

    def _make_tensor(self, params, state):
        import numpy as np
        size, steps, allocation_size = self._get_size_and_steps(params)
        constructor = (
            self._tensor_constructor or
            self.default_tensor_constructor
        )

        raw_tensor = constructor(size=allocation_size, dtype=self._dtype, **params)
        if self._cuda:
            raw_tensor = raw_tensor.cuda()

        # Randomly permute the Tensor and call `.contiguous()` to force re-ordering
        # of the memory, and then permute it back to the original shape.
        dim = len(size)
        order = np.arange(dim)
        if state.rand() > self._probability_contiguous:
            while dim > 1 and np.all(order == np.arange(dim)):
                order = state.permutation(raw_tensor.dim())

            raw_tensor = raw_tensor.permute(tuple(order)).contiguous()
            raw_tensor = raw_tensor.permute(tuple(np.argsort(order)))

        slices = [slice(0, size * step, step) for size, step in zip(size, steps)]
        tensor = raw_tensor[slices]

        properties = {
            "numel": int(tensor.numel()),
            "order": order,
            "steps": steps,
            "is_contiguous": tensor.is_contiguous(),
            "dtype": str(self._dtype),
        }

        return tensor, properties

    def _get_size_and_steps(self, params):
        dim = (
            params[self._dim_parameter]
            if self._dim_parameter is not None
            else len(self._size)
        )

        def resolve(values, dim):
            """Resolve values into concrete integers."""
            values = tuple(params.get(i, i) for i in values)
            if len(values) > dim:
                values = values[:dim]
            if len(values) < dim:
                values = values + tuple(1 for _ in range(dim - len(values)))
            return values

        size = resolve(self._size, dim)
        steps = resolve(self._steps or (), dim)
        allocation_size = tuple(size_i * step_i for size_i, step_i in zip(size, steps))
        return size, steps, allocation_size

    def satisfies_constraints(self, params):
        size, _, allocation_size = self._get_size_and_steps(params)
        # Product is computed in Python to avoid integer overflow.
        num_elements = prod(size)
        assert num_elements >= 0

        allocation_bytes = prod(allocation_size, base=dtype_size(self._dtype))

        def nullable_greater(left, right):
            if left is None or right is None:
                return False
            return left > right

        return not any((
            nullable_greater(num_elements, self._max_elements),
            nullable_greater(self._min_elements, num_elements),
            nullable_greater(allocation_bytes, self._max_allocation_bytes),
        ))


class Fuzzer:
    def __init__(
        self,
        parameters: List[Union[FuzzedParameter, List[FuzzedParameter]]],
        tensors: List[Union[FuzzedTensor, List[FuzzedTensor]]],
        constraints: Optional[List[Callable]] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            parameters:
                List of FuzzedParameters which provide specifications
                for generated parameters. Iterable elements will be
                unpacked, though arbitrary nested structures will not.
            tensors:
                List of FuzzedTensors which define the Tensors which
                will be created each step based on the parameters for
                that step. Iterable elements will be unpacked, though
                arbitrary nested structures will not.
            constraints:
                List of callables. They will be called with params
                as kwargs, and if any of them return False the current
                set of parameters will be rejected.
            seed:
                Seed for the RandomState used by the Fuzzer. This will
                also be used to set the PyTorch random seed so that random
                ops will create reproducible Tensors.
        """
        import numpy as np
        if seed is None:
            seed = int(np.random.RandomState().randint(0, 2 ** 32 - 1, dtype=np.int64))
        self._seed = seed
        self._parameters = Fuzzer._unpack(parameters, FuzzedParameter)
        self._tensors = Fuzzer._unpack(tensors, FuzzedTensor)
        self._constraints = constraints or ()

        p_names = {p.name for p in self._parameters}
        t_names = {t.name for t in self._tensors}
        name_overlap = p_names.intersection(t_names)
        if name_overlap:
            raise ValueError(f"Duplicate names in parameters and tensors: {name_overlap}")

        self._rejections = 0
        self._total_generated = 0

    @staticmethod
    def _unpack(values, cls):
        return tuple(it.chain(
            *[[i] if isinstance(i, cls) else i for i in values]
        ))

    def take(self, n):
        import numpy as np
        state = np.random.RandomState(self._seed)
        torch.manual_seed(state.randint(low=0, high=2 ** 63, dtype=np.int64))
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
        strict_params: Dict[str, Union[float, int, ParameterAlias]] = {}
        for _ in range(1000):
            candidate_params: Dict[str, Union[float, int, ParameterAlias]] = {}
            for p in self._parameters:
                if p.strict:
                    if p.name in strict_params:
                        candidate_params[p.name] = strict_params[p.name]
                    else:
                        candidate_params[p.name] = p.sample(state)
                        strict_params[p.name] = candidate_params[p.name]
                else:
                    candidate_params[p.name] = p.sample(state)

            candidate_params = self._resolve_aliases(candidate_params)

            self._total_generated += 1
            if not all(f(candidate_params) for f in self._constraints):
                self._rejections += 1
                continue

            if not all(t.satisfies_constraints(candidate_params) for t in self._tensors):
                self._rejections += 1
                continue

            return candidate_params
        raise ValueError("Failed to generate a set of valid parameters.")

    @staticmethod
    def _resolve_aliases(params):
        params = dict(params)
        alias_count = sum(isinstance(v, ParameterAlias) for v in params.values())

        keys = list(params.keys())
        while alias_count:
            for k in keys:
                v = params[k]
                if isinstance(v, ParameterAlias):
                    params[k] = params[v.alias_to]
            alias_count_new = sum(isinstance(v, ParameterAlias) for v in params.values())
            if alias_count == alias_count_new:
                raise ValueError(f"ParameterAlias cycle detected\n{params}")

            alias_count = alias_count_new

        return params
