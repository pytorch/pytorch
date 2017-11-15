#include <type_traits>

namespace at {
namespace native {

template <typename scalartype> bool is_signed(const Tensor& self) {
  return std::is_signed<scalartype>::value;
}

}
}
