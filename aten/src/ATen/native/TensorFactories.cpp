#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor empty_like(const Tensor& self) {
  return self.type().tensor(self.sizes());
}

Tensor empty_like(const Tensor& self, const Type& dtype) {
  return dtype.tensor(self.sizes());
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

Tensor zeros(const Type& dtype, IntList size) {
  auto result = dtype.tensor(size);
  return result.fill_(0);
}

Tensor& zeros_out(Tensor& result, IntList size) {
  result.resize_(size);
  return result.fill_(0);
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
