import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Union, Optional, Iterable, Dict, Tuple
from contextlib import contextmanager


_cache_enabled = 0
_cache: Dict[Tuple[int, str], Optional[Tensor]] = {}


@contextmanager
def cached():
    r"""Context manager that enables the caching system within parametrizations
    registered with :func:`register_parametrization`.

    The value of the parametrized objects is computed and cached the first time
    they are required when this context manager is active. The cached values are
    discarded when leaving the context manager.

    This is useful when using a parametrized parameter more than once in the forward pass.
    An example of this is when parametrizing the recurrent kernel of an RNN or when
    sharing weights.

    The simplest way to activate the cache is by wrapping the forward pass of the neural network

    .. code-block:: python

        import torch.nn.utils.parametrize as P
        ...
        with P.cached():
            output = model(inputs)

    in training and evaluation. One may also wrap the parts of the modules that use
    several times the parametrized tensors. For example, the loop of an RNN with a
    parametrized recurrent kernel:

    .. code-block:: python

        with P.cached():
            for x in xs:
                out_rnn = self.rnn_cell(x, out_rnn)
    """
    global _cache
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache = {}


class ParametrizationList(ModuleList):
    r"""A sequential container that holds and manages the ``original`` parameter or buffer of
    a parametrized :class:`torch.nn.Module`. It is the type of
    ``module.parametrizations[tensor_name]`` when ``module[tensor_name]`` has been parametrized
    with :func:`register_parametrization`.

    .. note ::
        This class is used internally by :func:`register_parametrization`. It is documented
        here for completeness. It should not be instantiated by the user.

    Args:
        modules (iterable): an iterable of modules representing the parametrizations
        original (Parameter or Tensor): parameter or buffer that is parametrized
    """
    original: Tensor

    def __init__(
        self, modules: Iterable[Module], original: Union[Tensor, Parameter]
    ) -> None:
        super().__init__(modules)
        if isinstance(original, Parameter):
            self.register_parameter("original", original)
        else:
            self.register_buffer("original", original)

    def set_original_(self, value: Tensor) -> None:
        r"""This method is called when assigning to a parametrized tensor.

        It calls the methods ``right_inverse`` (see :func:`register_parametrization`)
        of the parametrizations in the inverse order that they have been registered.
        Then, it assigns the result to ``self.original``.

        Args:
            value (Tensor): Value to which initialize the module

        Raises:
            RuntimeError: if any of the parametrizations do not implement a ``right_inverse`` method
        """
        with torch.no_grad():
            # See https://github.com/pytorch/pytorch/issues/53103
            for module in reversed(self):  # type: ignore[call-overload]
                if hasattr(module, "right_inverse"):
                    value = module.right_inverse(value)
                else:
                    raise RuntimeError(
                        "The parametrization '{}' does not implement a 'right_inverse' method. "
                        "Assigning to a parametrized tensor is only possible when all the parametrizations "
                        "implement a 'right_inverse' method.".format(module.__class__.__name__)
                    )
            self.original.copy_(value)

    def forward(self) -> Tensor:
        x = self.original
        for module in self:
            x = module(x)
        if x.size() != self.original.size():
            raise RuntimeError(
                "The parametrization may not change the size of the parametrized tensor. "
                "Size of original tensor: {} "
                "Size of parametrized tensor: {}".format(self.original.size(), x.size())
            )
        return x


def _inject_new_class(module: Module) -> None:
    r"""Sets up the parametrization mechanism used by parametrizations.

    This works by substituting the class of the module by a class
    that extends it to be able to inject a property

    Args:
        module (nn.Module): module into which to inject the property
    """
    cls = module.__class__

    def getstate(self):
        raise RuntimeError(
            "Serialization of parametrized modules is only "
            "supported through state_dict(). See:\n"
            "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
            "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
        )

    param_cls = type(
        "Parametrized{}".format(cls.__name__),
        (cls,),
        {
            "__getstate__": getstate,
        },
    )

    module.__class__ = param_cls


def _inject_property(module: Module, tensor_name: str) -> None:
    r"""Injects a property into module[tensor_name].

    It assumes that the class in the module has already been modified from its
    original one using _inject_new_class and that the tensor under :attr:`tensor_name`
    has already been moved out

    Args:
        module (nn.Module): module into which to inject the property
        tensor_name (str): name of the name of the property to create
    """
    # We check the precondition.
    # This should never fire if register_parametrization is correctly implemented
    assert not hasattr(module, tensor_name)

    def get_parametrized(self) -> Tensor:
        global _cache

        parametrization = self.parametrizations[tensor_name]
        if _cache_enabled:
            key = (id(module), tensor_name)
            tensor = _cache.get(key)
            if tensor is None:
                tensor = parametrization()
                _cache[key] = tensor
            return tensor
        else:
            # If caching is not active, this function just evaluates the parametrization
            return parametrization()

    def set_original(self, value: Tensor) -> None:
        self.parametrizations[tensor_name].set_original_(value)

    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))


def register_parametrization(
    module: Module, tensor_name: str, parametrization: Module
) -> Module:
    r"""Adds a parametrization to a tensor in a module.

    Assume that ``tensor_name="weight"`` for simplicity. When accessing ``module.weight``,
    the module will return the parametrized version ``parametrization(module.weight)``.
    If the original tensor requires a gradient, the backward pass will differentiate
    through the :attr:`parametrization`, and the optimizer will update the tensor accordingly.

    The first time that a module registers a parametrization, this function will add an attribute
    ``parametrizations`` to the module of type :class:`~ParametrizationList`.

    The list of parametrizations on a tensor will be accessible under
    ``module.parametrizations.weight``.

    The original tensor will be accessible under
    ``module.parametrizations.weight.original``.

    Parametrizations may be concatenated by registering several parametrizations
    on the same attribute.

    The training mode of the registered parametrizations are updated on registration
    if necessary to match the training mode of the host module

    Parametrized parameters and buffers have an inbuilt caching system that can be activated
    using the context manager :func:`cached`.

    A :attr:`parametrization` may optionally implement a method with signature

    .. code-block:: python

        def right_inverse(self, X: Tensor) -> Tensor

    If :attr:`parametrization` implements this method, it will be possible to assign
    to the parametrized tensor. This may be used to initialize the tensor, as shown in the example.

    In most situations, ``right_inverse`` will be a function such that
    ``forward(right_inverse(X)) == X`` (see
    `right inverse <https://en.wikipedia.org/wiki/Inverse_function#Right_inverses>`_).
    Sometimes, when the parametrization is not surjective, it may be reasonable
    to relax this, as shown in the example below.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (str): name of the parameter or buffer on which to register
            the parametrization
        parametrization (nn.Module): the parametrization to register

    Returns:
        Module: module

    Raises:
        ValueError: if the module does not have a parameter or a buffer named :attr:`tensor_name`

    Examples:
        >>> import torch
        >>> import torch.nn.utils.parametrize as P
        >>>
        >>> class Symmetric(torch.nn.Module):
        >>>     def forward(self, X):
        >>>         return X.triu() + X.triu(1).T  # Return a symmetric matrix
        >>>
        >>>     def right_inverse(self, A):
        >>>         return A.triu()
        >>>
        >>> m = torch.nn.Linear(5, 5)
        >>> P.register_parametrization(m, "weight", Symmetric())
        >>> print(torch.allclose(m.weight, m.weight.T))  # m.weight is now symmetric
        True
        >>> A = torch.rand(5, 5)
        >>> A = A + A.T   # A is now symmetric
        >>> m.weight = A  # Initialize the weight to be the symmetric matrix A
        >>> print(torch.allclose(m.weight, A))
        True
    """
    parametrization.train(module.training)
    if is_parametrized(module, tensor_name):
        # Just add the new parametrization to the parametrization list
        module.parametrizations[tensor_name].append(parametrization)  # type: ignore[index, union-attr]
    elif tensor_name in module._buffers or tensor_name in module._parameters:
        # Set the parametrization mechanism
        # Fetch the original buffer or parameter
        original = getattr(module, tensor_name)
        # Delete the previous parameter or buffer
        delattr(module, tensor_name)
        # If this is the first parametrization registered on the module,
        # we prepare the module to inject the property
        if not is_parametrized(module):
            # Change the class
            _inject_new_class(module)
            # Inject the a ``ModuleDict`` into the instance under module.parametrizations
            module.parametrizations = ModuleDict()
        # Add a property into the class
        _inject_property(module, tensor_name)
        # Add a ParametrizationList
        module.parametrizations[tensor_name] = ParametrizationList(  # type: ignore[assignment, index, operator]
            [parametrization], original
        )
    else:
        raise ValueError(
            "Module '{}' does not have a parameter, a buffer, or a "
            "parametrized element with name '{}'".format(module, tensor_name)
        )
    return module


def is_parametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    r"""Returns ``True`` if module has an active parametrization.

    If the argument :attr:`tensor_name` is specified, returns ``True`` if
    ``module[tensor_name]`` is parametrized.

    Args:
        module (nn.Module): module to query
        name (str, optional): attribute in the module to query
            Default: ``None``
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, ModuleDict):
        return False
    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
        return len(parametrizations) > 0
    else:
        return tensor_name in parametrizations


def remove_parametrizations(
    module: Module, tensor_name: str, leave_parametrized: bool = True
) -> Module:
    r"""Removes the parametrizations on a tensor in a module.

    - If ``leave_parametrized=True``, ``module[tensor_name]`` will be set to
      its current output. In this case, the parametrization shall not change the ``dtype``
      of the tensor.
    - If ``leave_parametrized=False``, ``module[tensor_name]`` will be set to
      the unparametrised tensor in ``module.parametrizations[tensor_name].original``.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute :attr:`tensor_name` parametrized.
            Default: ``True``

    Returns:
        Module: module

    Raises:
        ValueError: if ``module[tensor_name]`` is not parametrized
        ValueError: if ``leave_parametrized=True`` and the parametrization changes the size or dtype
            of the tensor
    """

    if not is_parametrized(module, tensor_name):
        raise ValueError(
            "Module {} does not have a parametrization on {}".format(
                module, tensor_name
            )
        )

    # Fetch the original tensor
    original = module.parametrizations[tensor_name].original  # type: ignore[index, union-attr]
    if leave_parametrized:
        with torch.no_grad():
            t = getattr(module, tensor_name)
        # If they have the same dtype, we reuse the original tensor.
        # We do this so that the parameter does not to change the id()
        # This way the user does not need to update the optimizer
        if t.dtype == original.dtype:
            with torch.no_grad():
                original.set_(t)
        else:
            raise ValueError(
                "The parametrization changes the dtype of the tensor from {} to {}. "
                "It is not supported to leave the tensor parametrized (`leave_parametrized=True`) "
                "in this case.".format(original.dtype, t.dtype)
            )
    # Delete the property that manages the parametrization
    delattr(module.__class__, tensor_name)
    # Delete the ParametrizationList
    del module.parametrizations[tensor_name]  # type: ignore[operator, union-attr]

    # Restore the parameter / buffer into the main class
    if isinstance(original, Parameter):
        module.register_parameter(tensor_name, original)
    else:
        module.register_buffer(tensor_name, original)

    # Roll back the parametrized class if no other buffer or parameter
    # is currently parametrized in this class
    if not is_parametrized(module):
        delattr(module, "parametrizations")
        # Restore class
        orig_cls = module.__class__.__bases__[0]
        module.__class__ = orig_cls
    return module
