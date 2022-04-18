import functools

# Implementation note: I had a choice about how much of mode stacks
# to implement in Python versus in C++.  At time of writing, I did not care
# too much about implementation efficiency; however, I do care about making it
# hard for users to implement modes in the wrong way.  In the end, it turned
# out to be possible to implement mode stacks entirely from userland, with the
# C++ API providing only _get_torch_function_mode() and
# _set_torch_function_mode(), so I opted to provide some unsafe C++ bindings and
# have the bulk of the logic for managing the stack in Python, which helped
# simplify the C++ API surface.  It would also have been valid to build in the
# notion of mode stack directly into C++ but in this design it's substantially
# more difficult to interact with TorchFunctionModeMeta.

def _wrap_init(f):
    undef = object()

    @functools.wraps(f)
    def wrapped(self, *args, inner=undef, **kwargs):
        if inner is undef:
            raise TypeError(
                "missing inner keyword argument; instead of constructing a TorchModeFunction directly, "
                "pass the constructor to push_python_mode"
            )
        self.inner = inner
        return f(self, *args, **kwargs)
    return wrapped

def _wrap_torch_dispatch(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        with enable_python_mode(self.inner):
            return f(self, *args, **kwargs)
    return wrapped

class ModeMeta(type):
    def __new__(metacls, name, bases, dct):
        if '__init__' in dct:
            dct['__init__'] = _wrap_init(dct['__init__'])
        return super().__new__(metacls, name, bases, dct)

class PythonModeMeta(ModeMeta):
    """
    Metaclass for :class:`PythonMode`; it does two things:

        * Adds an implicit ``inner`` kwarg to ``__init__``, to
          allow the modes to be chained together to form a stack.

        * Reenables the inner mode, so that by default PyTorch API calls
          will compositionally proceed to the next mode on the stack.

    The default behavior for the second bullet is important, as it is easy to
    accidentally write ``_wrap_torch_dispatch`` implementations that are not
    compositional, and the wrapping here makes the obvious code do the
    right thing (aka, this is why there is a metaclass).
    """
    def __new__(metacls, name, bases, dct):
        if '__torch_dispatch__' in dct:
            dct['__torch_dispatch__'] = _wrap_torch_dispatch(dct['__torch_dispatch__'])
        return super().__new__(metacls, name, bases, dct)


class PythonMode(metaclass=PythonModeMeta):
    """
    A ``PythonMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`PythonMode` are compositional:
    modes can be pushed onto a stack with :func:`push_python_mode`.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self, replace=self.inner)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    # Force metaclass to generate constructor at the base of the hierarchy
    def __init__(self):
        pass

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()


class BasePythonMode(PythonMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

class TorchFunctionModeMeta(ModeMeta):
    """
    Metaclass for :class:`TorchFunctionMode`; it does two things:

        * Adds an implicit ``inner`` kwarg to ``__init__``, to
          allow the modes to be chained together to form a stack.

        * Reenables the inner mode, so that by default PyTorch API calls
          will compositionally proceed to the next mode on the stack.

    The default behavior for the second bullet is important, as it is easy to
    accidentally write ``__torch_function__`` implementations that are not
    compositional, and the wrapping here makes the obvious code do the
    right thing (aka, this is why there is a metaclass).
    """
    def __new__(metacls, name, bases, dct):
        if '__torch_function__' in dct:
            dct['__torch_function__'] = _wrap_torch_function(dct['__torch_function__'])
        return super().__new__(metacls, name, bases, dct)


class TorchFunctionMode(metaclass=TorchFunctionModeMeta):
    """
    A ``TorchFunctionMode`` allows you to override the meaning of all
    ``__torch_function__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchFunctionMode` are compositional:
    modes can be pushed onto a stack with :func:`push_torch_function_mode`.
    When you call functions in the PyTorch API inside your
    ``__torch_function__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_function__`` implementation, either explicitly
    invoke ``self.__torch_function__(...)``, or use the context manager
    ``enable_torch_function_mode(self, replace=self.inner)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    # Force metaclass to generate constructor at the base of the hierarchy
    def __init__(self):
        pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()


class BaseTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)