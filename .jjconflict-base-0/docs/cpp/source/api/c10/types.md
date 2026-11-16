---
myst:
  html_meta:
    description: Core types in PyTorch C++ — ArrayRef, optional, Dict, List, IListRef, Half, and IValue.
    keywords: PyTorch, C++, c10, ArrayRef, optional, Dict, List, IListRef, Half, IValue
---

# Core Types

C10 provides fundamental types used throughout PyTorch.

## ArrayRef

```{doxygenclass} c10::ArrayRef
:members:
:undoc-members:
```

**Example:**

```cpp
std::vector<int64_t> sizes = {3, 4, 5};
c10::ArrayRef<int64_t> sizes_ref(sizes);

// Can also use initializer list
auto tensor = at::zeros({3, 4, 5});  // implicitly converts
```

## OptionalArrayRef

```{doxygenclass} c10::OptionalArrayRef
:members:
:no-link:
```

**Example:**

```cpp
void my_function(c10::OptionalArrayRef<int64_t> sizes = c10::nullopt) {
    if (sizes.has_value()) {
        for (auto s : sizes.value()) {
            // process sizes
        }
    }
}
```

## Optional

```{cpp:class} c10::optional

A wrapper type that may or may not contain a value.
Similar to `std::optional`.
```

```{cpp:function} bool has_value() const

Returns true if a value is present.
```

```{cpp:function} T& value()

Returns the contained value. Throws if empty.
```

```{cpp:function} T value_or(T default_value) const

Returns the value if present, otherwise returns the default.
```

**Example:**

```cpp
c10::optional<int64_t> maybe_dim = c10::nullopt;

if (maybe_dim.has_value()) {
    std::cout << "Dim: " << maybe_dim.value() << std::endl;
}

int64_t dim = maybe_dim.value_or(-1);  // Returns -1 if empty
```

## Half

```{cpp:class} c10::Half

16-bit floating point type (IEEE 754 half-precision).
```

```{cpp:function} Half(float value)

Construct from a float.
```

```{cpp:function} operator float() const

Convert to float.
```

**Example:**

```cpp
c10::Half h = 3.14f;
float f = static_cast<float>(h);
```

## Containers

C10 provides container types that store `IValue` elements internally. These
are pointer types: copies share the same underlying storage.

### Dict

An ordered hash map from `Key` to `Value`. Valid key types are `int64_t`,
`double`, `bool`, `std::string`, and `at::Tensor`.

```{doxygenclass} c10::Dict
:members:
:undoc-members:
```

**Example:**

```cpp
#include <ATen/core/Dict.h>

c10::Dict<std::string, at::Tensor> named_tensors;
named_tensors.insert("weight", torch::randn({3, 3}));
named_tensors.insert("bias", torch::zeros({3}));

if (named_tensors.contains("weight")) {
    at::Tensor w = named_tensors.at("weight");
}

for (const auto& entry : named_tensors) {
    std::cout << entry.key() << ": " << entry.value().sizes() << std::endl;
}
```

### List

A type-safe list container backed by `IValue` elements.

```{doxygenclass} c10::List
:members:
:undoc-members:
```

**Example:**

```cpp
#include <ATen/core/List.h>

c10::List<at::Tensor> tensor_list;
tensor_list.push_back(torch::randn({2, 3}));
tensor_list.push_back(torch::zeros({2, 3}));

at::Tensor first = tensor_list.get(0);
std::cout << "List size: " << tensor_list.size() << std::endl;

c10::List<int64_t> int_list;
int_list.push_back(1);
int_list.push_back(2);
int_list.push_back(3);
```

### IListRef

`c10::IListRef<T>` is a lightweight reference type that provides a unified
interface over different list-like types (`List<T>`, `ArrayRef<T>`,
`std::vector<T>`). It avoids copying when passing list arguments to operators.

```{doxygenclass} c10::IListRef
:members:
:undoc-members:
```

**Example:**

```cpp
#include <ATen/core/IListRef.h>

// IListRef can wrap different underlying types
std::vector<at::Tensor> vec = {torch::randn({2}), torch::randn({3})};
c10::IListRef<at::Tensor> ref(vec);

for (const auto& t : ref) {
    std::cout << t.sizes() << std::endl;
}
```

## IValue

`c10::IValue` (Interpreter Value) is a type-erased container used extensively
for storing values of different types. It can hold tensors,
scalars, lists, dictionaries, and other types.

```{note}

The full API documentation for IValue is complex due to its many type
conversion methods. See the header file `ATen/core/ivalue.h` for complete
details.
```

**Common methods:**

- `isTensor()` / `toTensor()` - Check if tensor / convert to tensor
- `isInt()` / `toInt()` - Check if int / convert to int
- `isDouble()` / `toDouble()` - Check if double / convert to double
- `isBool()` / `toBool()` - Check if bool / convert to bool
- `isString()` / `toString()` - Check if string / convert to string
- `isList()` / `toList()` - Check if list / convert to list
- `isGenericDict()` / `toGenericDict()` - Check if dict / convert to dict
- `isTuple()` / `toTuple()` - Check if tuple / convert to tuple
- `isNone()` - Check if None/null

**Example:**

```cpp
c10::IValue val = at::ones({2, 2});

if (val.isTensor()) {
    at::Tensor t = val.toTensor();
}
```
