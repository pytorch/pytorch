#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "TH/THRandom.h"
#include "ATen/CheckGenerator.h"
#include "ATen/CPUGenerator.h"
#include "ATen/Dispatch.h"
#include <algorithm>
#include <sstream>

namespace at {
namespace native {

Tensor arange(const Type& dtype, Scalar start, Scalar end, Scalar step) {
  return dtype._arange(start, end, step);
}

Tensor& arange_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  return at::_arange_out(result, start, end, step);
}

Tensor arange(const Type& dtype, Scalar end) {
  return dtype._arange(end);
}

Tensor& arange_out(Tensor& result, Scalar end) {
  return at::_arange_out(result, end);
}

Tensor empty_like(const Tensor& self) {
  return self.type().tensor(self.sizes());
}

Tensor empty_like(const Tensor& self, const Type& dtype) {
  return dtype.tensor(self.sizes());
}

Tensor eye(const Type& dtype, int64_t n, int64_t m) {
  auto result = dtype.tensor();
  return at::eye_out(result, n, m);
}

Tensor& eye_out_cpu(Tensor& result, int64_t n, int64_t m) {
  if (n <= 0) {
    std::ostringstream oss;
    oss << "n must be greater than 0, got: " << n;
    throw std::runtime_error(oss.str());
  }
  if(m <= 0) {
    m = n;
  }

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  AT_DISPATCH_ALL_TYPES(result.type(), "eye", [&]() -> void {
    scalar_t* result_data = result.data<scalar_t>();
    for(int64_t i = 0; i < sz; i++) {
      result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
    }
  });

  return result;
}

Tensor linspace(const Type& dtype, Scalar start, Scalar end, int64_t steps) {
  return dtype._linspace(start, end, steps);
}

Tensor& linspace_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  return at::_linspace_out(result, start, end, steps);
}

Tensor logspace(const Type& dtype, Scalar start, Scalar end, int64_t steps) {
  return dtype._logspace(start, end, steps);
}

Tensor& logspace_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  return at::_logspace_out(result, start, end, steps);
}

Tensor ones(const Type& dtype, IntList size) {
  auto result = dtype.tensor(size);
  return result.fill_(1);
}

Tensor& ones_out(Tensor& result, IntList size) {
  result.resize_(size);
  return result.fill_(1);
}

Tensor ones_like(const Tensor& self) {
  return at::native::ones(self.type(), self.sizes());
}

Tensor ones_like(const Tensor& self, const Type& dtype) {
  return at::native::ones(dtype, self.sizes());
}

Tensor rand(const Type& dtype, IntList size, Generator* generator) {
  Tensor result = dtype.tensor(size);
  return result.uniform_(0, 1, generator);
}

Tensor& rand_out(Tensor& result, IntList size, Generator* generator) {
  result.resize_(size);
  return result.uniform_(0, 1, generator);
}

Tensor rand_like(const Tensor& self) {
  return at::native::rand_like(self, self.type());
}

Tensor rand_like(const Tensor& self, const Type& dtype) {
  return at::native::rand(dtype, self.sizes());
}

Tensor randn(const Type& dtype, IntList size, Generator* generator) {
  Tensor result = dtype.tensor(size);
  return result.normal_(0, 1, generator);
}

Tensor& randn_out(Tensor& result, IntList size, Generator* generator) {
  result.resize_(size);
  return result.normal_(0, 1, generator);
}

Tensor randn_like(const Tensor& self) {
  return at::native::randn_like(self, self.type());
}

Tensor randn_like(const Tensor& self, const Type& dtype) {
  return at::native::randn(dtype, self.sizes(), nullptr);
}

namespace {
template <typename scalar_t>
void randperm_cpu(Tensor& result, int64_t n, THGenerator* generator) {
  scalar_t *r__data = result.data<scalar_t>();

  result.resize_({n});
  int64_t r__stride_0 = result.stride(0);

  for(int64_t i = 0; i < n; i++) {
    r__data[i*r__stride_0] = static_cast<scalar_t>(i);
  }

  for(int64_t i = 0; i < n - 1; i++)
  {
    int64_t z = THRandom_random(generator) % (n-i);
    scalar_t sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}
} // namespace


THGenerator* get_generator(at::Generator* gen) {
  auto default_gen = &at::globalContext().defaultGenerator(at::Backend::CPU);
  auto gen_ = at::check_generator<at::CPUGenerator>(gen, default_gen);
  return gen_->generator;
}

Tensor randperm(const Type& dtype, int64_t n, Generator* generator) {
  Tensor result = dtype.tensor(n);
  return at::native::randperm_out(result, n, generator);
}

Tensor& randperm_out(Tensor& result, int64_t n, Generator* generator) {
  if (n <= 0) {
    std::ostringstream oss;
    oss << "n must be strictly positive, got " << n;
    throw std::runtime_error(oss.str());
  }
  if (result.type().backend() != at::kCPU) {
    throw std::runtime_error("randperm is only implemented for CPU");
  }

  result.resize_({n});
  auto gen = get_generator(generator);
  AT_DISPATCH_ALL_TYPES(result.type(), "randperm", [&]() -> void {
    randperm_cpu<scalar_t>(result, n, gen);
  });

  return result;
}

Tensor range(const Type& dtype, Scalar start, Scalar end, Scalar step) {
  return dtype._range(start, end, step);
}

Tensor& range_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  return at::_range_out(result, start, end, step);
}

Tensor zeros(const Type& dtype, IntList size) {
  auto result = dtype.tensor(size);
  return at::native::zeros_out(result, size);
}

Tensor& zeros_out(Tensor& result, IntList size) {
  if (result.is_sparse()) {
    result.sparse_raw_resize_(size, size.size(), 0);
  } else {
    result.resize_(size);
  }
  return result.zero_();
}

Tensor zeros_like(const Tensor& self) {
  return at::native::zeros_like(self, self.type());
}

Tensor zeros_like(const Tensor& self, const Type& dtype) {
  if (dtype.is_sparse() && self.type().is_sparse()) {
    auto res = dtype.tensor();
    res.resize_as_(self);
    res.zero_();
    return res;
  }
  return at::native::zeros(dtype, self.sizes());
}

}
}
