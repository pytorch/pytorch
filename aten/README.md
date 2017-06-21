# ATen: A TENsor library

ATen is a simple tensor library thats exposes the Tensor operations in Torch
and PyTorch directly in C++11. The wrapper respects the semantics of operators
in PyTorch, except minor details due to differences between C++ in Python in
the way default arguments are handled. See the [documentation for tensors](http://pytorch.org/docs/tensors.html) in PyTorch for what these operations do.
ATen's API is auto-generated from the same declarations PyTorch uses so the
two APIs will track each other over time.

Tensor types are resolved dynamically, such that the API is generic and
does not include templates. That is, there is one `Tensor` type. It can hold a
CPU or CUDA Tensor, and the tensor may have Doubles, Float, Ints, etc. This design
makes it easy to write generic code without templating everything.

See the _generated_ [`Tensor.h` file](doc/Tensor.h) and [`Functions.h` file](doc/Tensor.h) for the provided API. Excerpt:
```c++
Tensor atan2(const Tensor & other) const;
Tensor & atan2_(const Tensor & other);
Tensor pow(Scalar exponent) const;
Tensor pow(const Tensor & exponent) const;
Tensor & pow_(Scalar exponent);
Tensor & pow_(const Tensor & exponent);
Tensor lerp(const Tensor & end, Scalar weight) const;
Tensor & lerp_(const Tensor & end, Scalar weight);
Tensor histc() const;
Tensor histc(int64_t bins) const;
Tensor histc(int64_t bins, Scalar min) const;
Tensor histc(int64_t bins, Scalar min, Scalar max) const;
```

Inplace operations are also provided, and always suffixed by `_` to indicate they will modify the Tensor.

### Installation

TH/THC/THNN/THCUNN are provided (as git subtrees), so the repo is standalone. You will need a C++11 compiler, cmake, and the pyyaml python package.
```

# Install pyyaml used by python code generation to read API declarations

# OSX: if you don't have pip
sudo easy_install pip
# Ubuntu: if you don't have pip
apt-get -y install python-pip

# if you don't have pyyaml
sudo pip install pyyaml

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/where/you/want # specify your dest directory
make install
```

### Example usage

Here is a simple example; again, the syntax follows Torch semantics.

```c++
using namespace at; // assumed in the following

Tensor d = CPU(kFloat).ones({3, 4});
Tensor r = CPU(kFloat).zeros({3,4})
for(auto i = 0; i < 100000; i++) {
  r = r.add(d);
  // equivalently
  r = r + d;
  // or
  r += d;
}
```

Want this running on the GPU?
```c++
using namespace at; // assumed in the following

Tensor d = CUDA(kFloat).ones({3, 4});
Tensor r = CUDA(kFloat).zeros({3,4})
for(auto i = 0; i < 100000; i++) {
  r = r.add(d);
  // equivalently
  r = r + d;
  // or
  r += d;
}
```

Expressions like `CUDA(kFloat)` are first-class `at::Type` objects that represent
the type of a Tensor and are used to create Tensors when their type cannot be
inferred. See the _generated_ [Type header](doc/Type.h) for its API.

See more in [sample files](src/ATen/test).

### Creating your kernel

It is easy to create new kernels, thanks to the `dispatch<>()` templated function. Example:
```c++

// a simple sum kernel (for CPU only)
template<typename T>
struct sum_op {
  // dispatch handles variable arguments for you
  Tensor CPU(const Type & t, Tensor & x_)
  {
    Tensor x = x_.contiguous();
    auto x_p = x.data<T>();
    int64_t size = x.numel();
    T sum = 0;
    for(int64_t i = 0; i < size; i++) {
      sum += x_p[i];
    }
    return sum;
  };
  Tensor CUDA(Tensor& x) {
    throw std::invalid_argument("device not supported");
  };
};

Tensor a = CPU(kFloat).rand({3, 7});
std::cout << a << std::endl;
std::cout << dispatch<sum_op>(a.type(),a) << " == " << a.sum() << std::endl;
```

### Efficient access to template elements

When using Tensor-wide operations, the cost of dynamic dispatch is very small.
However, there are cases, especially in your own kernels, where efficient element-wise access is needed.
ATen provides _accessors_ that are created with a single dynamic check that a Tensor is the type and number of
dimensions. Accessors then expose an API for accessing the Tensor elements efficiently:

```c++

Tensor foo = CPU(kFloat).rand({12,12});

// assert foo is 2-dimensional and holds floats.
auto foo_a = foo.accessor<float,2>();
float trace = 0;

for(int i = 0; i < foo_a.size(0); i++) {
  // use the accessor foo_a to get tensor data.
  trace += foo_a[i][i];
}
```

Accessors are temporary views of a Tensor. They are only valid for the lifetime of the tensor that they
view and hence should only be used locally in a function, like iterators.
