# Python Object Interop

Include `<torch/csrc/stable/pyobject.h>` to convert between Python objects and
stable C++ values without including `Python.h` in the extension's public headers.
You may be interested in these conversions if you'd like to bypass the torch
dispatcher when binding ops to Python.
Similar to all stable header usage, configure `TORCH_TARGET_VERSION` for the
oldest runtime you intend to support.

Your extension can link only `libtorch`, but in order to use these APIs, please
ensure `libtorch_python` is loaded at runtime. When calling any of the Python
conversion functions below, the caller must hold the GIL.

Without an explicit target, the build version is used. Each group below requires
both a target and runtime at least as new as its stated minimum.

## Tensor Conversions

Minimum target and runtime: PyTorch 2.14.

```{doxygenfunction} torch::stable::tensor_from_pyobject
```

```{doxygenfunction} torch::stable::tensor_to_pyobject
```

## Tensor Probe, Dtypes and Devices

Minimum target and runtime: PyTorch 2.15.

```{doxygenfunction} torch::stable::is_tensor_pyobject
```

```{doxygenfunction} torch::stable::dtype_from_pyobject
```

```{doxygenfunction} torch::stable::dtype_to_pyobject
```

```{doxygenfunction} torch::stable::device_from_pyobject
```

```{doxygenfunction} torch::stable::device_to_pyobject
```

## Layout and Memory Format

Minimum target and runtime: PyTorch 2.16.

```{doxygenfunction} torch::stable::layout_from_pyobject
```

```{doxygenfunction} torch::stable::layout_to_pyobject
```

```{doxygenfunction} torch::stable::memory_format_from_pyobject
```

```{doxygenfunction} torch::stable::memory_format_to_pyobject
```

## Scalars

Minimum target and runtime: PyTorch 2.16.

```{doxygenfunction} torch::stable::bool_from_pyobject
```

```{doxygenfunction} torch::stable::bool_to_pyobject
```

```{doxygenfunction} torch::stable::int64_from_pyobject
```

```{doxygenfunction} torch::stable::int64_to_pyobject
```

```{doxygenfunction} torch::stable::double_from_pyobject
```

```{doxygenfunction} torch::stable::double_to_pyobject
```

## Strings

Minimum target and runtime: PyTorch 2.16.

```{doxygenfunction} torch::stable::string_from_pyobject
```

```{doxygenfunction} torch::stable::string_to_pyobject
```
