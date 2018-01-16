#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include <type_traits>

namespace at {
namespace native {

bool is_cuda(const Tensor& self) {
  return self.type().is_cuda();
}

bool is_distributed(const Tensor& self) {
  return self.type().is_distributed();
}

template <typename scalar>
struct IsSigned {
  static bool apply() { return std::is_signed<scalar>(); }
};

template<>
struct IsSigned<Half> {
  static bool apply() { return true; }
};

bool is_signed(const Tensor &self) {
  return dispatch_all<bool, IsSigned>(self.type(), "is_signed");
}

bool is_sparse(const Tensor& self) {
  return self.type().is_sparse();
}

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.toType(other.type());
}

}
}
