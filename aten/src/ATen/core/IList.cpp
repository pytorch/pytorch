#include <ATen/core/IList.h>
#include <ATen/core/Tensor.h>

namespace c10 {
namespace detail {

IListConstRef<at::OptionalTensorRef>
IListTagImpl<IListTag::Boxed, at::OptionalTensorRef>::iterator_get(
    const typename list_type::const_iterator& it
) {
  const auto& ivalue = (*it).get();
  return (ivalue.isNone()) ? at::OptionalTensorRef{} : ivalue.toTensor();
}

} // namespace detail
} // namespace c10
