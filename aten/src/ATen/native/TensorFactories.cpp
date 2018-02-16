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

Tensor ones_like(const Tensor& self) {
  return self.type().ones(self.sizes());
}

Tensor& ones_like_out(Tensor& result, const Tensor& self) {
  return self.type().ones_out(result, self.sizes());
}

Tensor ones_like(const Tensor& self, const Type& dtype) {
  return dtype.ones(self.sizes());
}

Tensor rand_like(const Tensor& self) {
  return self.type().rand(self.sizes());
}

Tensor& rand_like_out(Tensor& result, const Tensor& self) {
  return self.type().rand_out(result, self.sizes());
}

Tensor rand_like(const Tensor& self, const Type& dtype) {
  return dtype.rand(self.sizes());
}

Tensor randn_like(const Tensor& self) {
  return self.type().randn(self.sizes());
}

Tensor& randn_like_out(Tensor& result, const Tensor& self) {
  return self.type().randn_out(result, self.sizes());
}

Tensor randn_like(const Tensor& self, const Type& dtype) {
  return dtype.randn(self.sizes());
}

Tensor zeros_like(const Tensor& self) {
  return self.type().zeros(self.sizes());
}

Tensor& zeros_like_out(Tensor& result, const Tensor& self) {
  return self.type().zeros_out(result, self.sizes());
}

Tensor zeros_like(const Tensor& self, const Type& dtype) {
  return dtype.zeros(self.sizes());
}

}
}
