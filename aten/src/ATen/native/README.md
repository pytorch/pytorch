ATen native functions are a mechanism to write ATen methods which only
make use of other ATen operations (e.g., it is not necessary to bind into
TH/THC code).  These functions are declared in this file and then folded
into the ATen code generation process, so that they become available
either as methods on `Tensor` (`t.mymeth()`) and functions in the ATen
namespace (`at::myfunc()`).

In PyTorch, every native function gets Python bindings generated,
making them available as methods on `torch._C._VariableBase`
(inherited by `Variable`, making such methods available to all
use sites of `Variable`) or `torch._C._FunctionBase` (and subsequently
rebound to the Python module that will make the function publically
visible).

## Registering it in `native_functions.yaml`

The first step is to write an entry for your function in
`native_functions.yaml`.  The format is as follows:

```
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> ReturnType
```

ArgType(s) are allowed to be simple types understood by ATen
(e.g. `Tensor`, `TensorList`, `IntList`, `int64_t`, `double`, ...).

By default, `Tensor` arguments do not accept undefined tensors
(`None` in Python). This restriction can be relaxed in two ways:

1. If a `Tensor` is at the end of an argument list, it can be defaulted
   with `Tensor={}`, which means it defaults to undefined when not passed.
2. Any `Tensor` can be made to accept undefined tensors with `Tensor?`.
   The argument must be specified, but a user can pass `{}` (C++) or `None`
   (Python).
   
`IntList` accepts an optional length specifier, e.g., `IntList[2]`.  When
possible, you should explicitly specify a length, as this allows our
Python bindings to accept a bare number (which will be expanded into an appropriate
sized list by repeating the number) along with lists/tuples.

ReturnType is allowed to be any ArgType or tuple combination of ArgTypes(s),
e.g. `(Tensor, Tensor)`

Defaults are optional.  Here are the supported default values:
* Numbers (e.g., `0` or `5.0` for `int64_t`, `double` and `IntList`
  with an explicit length (e.g., `IntList[2]`)--in the case of IntList,
  a number is replicated to fill the length (e.g., `IntList[2] x=2`
  is equivalent to `IntList[2] x={2,2}`.
* Lists of numbers (e.g., `{0, 0}`) for `IntList`.
* Booleans (e.g., `true`) for `bool`.
* Empty initializer lists (e.g., `{}`) for `Tensor`.
* `nullptr` for pointer types (e.g., `Generator*`)

The ATen code generation process will generate a header corresponding
to the C++ implementation you will have to write.
The C++ function declarations won't match the declaration here because
they will undergo the standard ATen C++ transformations, e.g. use of const-ref
for non-inplace Tensor arguments (instead of `Tensor` you will
take `const Tensor&`).  So after you finish writing your header, we recommend
running a build and getting the declaration from `NativeFunctions.h`
(find it with `find -name NativeFunctions.h`) to copy paste into your
C++ file.

The declarations also support the following attributes:

```
variants: function, method
```

Controls whether Tensor method (`t.foo()`) or namespace Function (`at::foo()`) is
generated as a result of this declaration.  If the declaration is a method,
you must have an argument `Tensoœr self` as the first argument.  In general, you
should default to defining a new ATen function as `variants: function`, unless
you know you want something to be usable as a method.

```
dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

This specifies the actual name of the function you want to dispatch to, so you
can dispatch to different functions depending on whether or not you have CPU or
CUDA tensors.  Technically, it is also possible to write `dispatch: func_name`
to unconditionally dispatch to a native function whose name is different than
the name in the public ATen API, but this is generally frowned upon (just name
them the same thing!)

```
python_default_init:
  argument_name: initializing_expression
```

A map from argument names to default initialize expressions in C++. Such default
expressions will only be used in Python API. This allows us to write argument
with a default value that can either cause ambiguity in C++ (e.g., `Scalar p`
argument in `norm`) or have a type that doesn't allow default value
None/NULL/nullptr (e.g., `int64_t fft_size` argument in stft, which we want to
default to value of another argument if not provided).

## Writing the implementation

Implementations of native functions go in an appropriate C++ file in the
`native/` directory (they are organized roughly by topic, but there is no
semantic meaning to their organization, except for CUDA files, which must
go in `cuda`.)  To write a native function, you only need to write a C++
implementation (no header necessary) with a matching signature to
the generated header from the ATen metadata.  There are many
simple native functions; take a look at some of them to see what to do.

There are some important gotchas which are important to keep in mind when
writing a native function:

* NEVER EVER EVER use `at::CPU` or `at::CUDA` directly; instead, create tensors
  from the `type()` of one of the input tensors, e.g., `input.type().tensor()`
  or `input.type().toScalarType(kByte)` if you need a tensor type directly.
  See https://github.com/pytorch/pytorch/issues/4477 for more details.

* Keep in mind whether or not your function must have an explicit derivative
  implemented for it, or if it is automatically differentiable.  Things that
  involve `data_ptr()` are unlikely to be automatically differentiable, but
  if your function just involves applying already existing ATen functions,
  it may be automatically differentiable (and you can skip an entry for
  it in `tools/autograd/derivatives.yaml` in PyTorch).

* If you are writing a function that is not intended to be automatically
  differentiable, make sure you explicitly qualify calls to other ATen
  functions with `at::foo`, and don't call them unqualified (in the
  `at::native` namespace.  This is important because ATen generates wrappers
  in the `at::` namespace which dispatch in a way that they can be overridden
  by differentiable Variables; this is not true if you short circuit this
  wrapper.

* Don't return undefined tensors from native functions, it doesn't fully work
  yet.

* If you get the signature wrong, you'll get a linker error.  Read it
  carefully; it will tell you what to do.
