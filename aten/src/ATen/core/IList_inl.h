#pragma once

#include <ATen/core/Tensor.h>

namespace at {
class Tensor;
}

namespace c10 {

/*
 * 'IList' implementation for 'at::Tensor'.
 */
namespace detail {
template <>
class IListTagImpl<at::Tensor, IListTag::Unboxed> {
 public:
  using list_type = at::ArrayRef<at::Tensor>;

  // Unwraps an `IList` into a const-ref of type `list_type`.
  static const list_type& unwrap(const IList<at::Tensor>& ilist);

  // Unwraps an `IListIterator` into a (const) ref of type
  // `list_type::const_iterator`. Has overload for const.
  static list_type::const_iterator& unwrap(IListIterator<at::Tensor>& it);
  static const list_type::const_iterator& unwrap(
      const IListIterator<at::Tensor>& it);

  // Accesses the element referenced by the unwrapped iterator `it`.
  static IListConstRef<at::Tensor> iterator_get(
      const list_type::const_iterator& it);
};

template <>
class IListTagImpl<at::Tensor, IListTag::Boxed> {
 public:
  using list_type = List<at::Tensor>;

  static const list_type& unwrap(const IList<at::Tensor>& ilist);
  static list_type::const_iterator& unwrap(IListIterator<at::Tensor>& it);
  static const list_type::const_iterator& unwrap(
      const IListIterator<at::Tensor>& it);
  static IListConstRef<at::Tensor> iterator_get(
      const list_type::const_iterator& it);
};

/*
 * 'IList' implementation for optional tensors.
 */
template <>
class IListTagImpl<at::OptionalTensorRef, IListTag::Unboxed> {
 public:
  using list_type = at::ArrayRef<at::OptionalTensorRef>;

  static const list_type& unwrap(const IList<at::OptionalTensorRef>& ilist);
  static list_type::const_iterator& unwrap(IListIterator<at::OptionalTensorRef>& it);
  static const list_type::const_iterator& unwrap(
      const IListIterator<at::OptionalTensorRef>& it);
  static IListConstRef<at::OptionalTensorRef> iterator_get(
      const list_type::const_iterator& it);
};

template <>
class IListTagImpl<at::OptionalTensorRef, IListTag::Boxed> {
 public:
  using list_type = List<optional<at::Tensor>>;

  static const list_type& unwrap(const IList<at::OptionalTensorRef>& ilist);
  static list_type::const_iterator& unwrap(IListIterator<at::OptionalTensorRef>& it);
  static const list_type::const_iterator& unwrap(
      const IListIterator<at::OptionalTensorRef>& it);
  static IListConstRef<at::OptionalTensorRef> iterator_get(
      const list_type::const_iterator& it);
};


}} // namespace c10::detail

inline
const TORCH_ILIST_IMPL(at::Tensor, Unboxed)::list_type&
TORCH_ILIST_IMPL(at::Tensor, Unboxed)::unwrap(
    const c10::IList<at::Tensor>& ilist
) {
  return ilist.payload_.unboxed;
}

inline
TORCH_ILIST_IMPL(at::Tensor, Unboxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::Tensor, Unboxed)::unwrap(
    c10::IListIterator<at::Tensor>& it
) {
  return it.payload_.unboxed_iterator;
}

inline
const TORCH_ILIST_IMPL(at::Tensor, Unboxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::Tensor, Unboxed)::unwrap(
    const c10::IListIterator<at::Tensor>& it
) {
  return it.payload_.unboxed_iterator;
}

inline
c10::detail::IListConstRef<at::Tensor>
TORCH_ILIST_IMPL(at::Tensor, Unboxed)::iterator_get(
    const list_type::const_iterator& it
) {
  return *it;
}

inline
const TORCH_ILIST_IMPL(at::Tensor, Boxed)::list_type&
TORCH_ILIST_IMPL(at::Tensor, Boxed)::unwrap(
    const c10::IList<at::Tensor>& ilist
) {
  return *ilist.payload_.boxed;
}

inline
TORCH_ILIST_IMPL(at::Tensor, Boxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::Tensor, Boxed)::unwrap(
    c10::IListIterator<at::Tensor>& it
) {
  return it.payload_.boxed_iterator;
}

inline
const TORCH_ILIST_IMPL(at::Tensor, Boxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::Tensor, Boxed)::unwrap(
    const c10::IListIterator<at::Tensor>& it
) {
  return it.payload_.boxed_iterator;
}

inline
c10::detail::IListConstRef<at::Tensor>
TORCH_ILIST_IMPL(at::Tensor, Boxed)::iterator_get(
    const list_type::const_iterator& it
) {
  return (*it).get().toTensor();
}

inline
const TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::list_type&
TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::unwrap(
    const c10::IList<at::OptionalTensorRef>& ilist
) {
  return ilist.payload_.unboxed;
}

inline
TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::unwrap(
    c10::IListIterator<at::OptionalTensorRef>& it
) {
  return it.payload_.unboxed_iterator;
}

inline
const TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::unwrap(
    const c10::IListIterator<at::OptionalTensorRef>& it
) {
  return it.payload_.unboxed_iterator;
}

inline
c10::detail::IListConstRef<at::OptionalTensorRef>
TORCH_ILIST_IMPL(at::OptionalTensorRef, Unboxed)::iterator_get(
    const list_type::const_iterator& it
) {
    return *it;
}

inline
const TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::list_type&
TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::unwrap(
    const c10::IList<at::OptionalTensorRef>& ilist
) {
  return *ilist.payload_.boxed;
}

inline
TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::unwrap(
    c10::IListIterator<at::OptionalTensorRef>& it
) {
  return it.payload_.boxed_iterator;
}

inline
const TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::list_type::const_iterator&
TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::unwrap(
    const c10::IListIterator<at::OptionalTensorRef>& it
) {
  return it.payload_.boxed_iterator;
}

inline
c10::detail::IListConstRef<at::OptionalTensorRef>
TORCH_ILIST_IMPL(at::OptionalTensorRef, Boxed)::iterator_get(
    const list_type::const_iterator& it
) {
  const auto& ivalue = (*it).get();
  return (ivalue.isNone()) ? at::OptionalTensorRef{} : ivalue.toTensor();
}

namespace at {
// [Note: ITensorList]
using ITensorList = c10::IList<at::Tensor>;
using ITensorListIterator = c10::IListIterator<at::Tensor>;
// [Note: at::OptionalTensorRefRefList]
using IOptTensorRefList = c10::IList<at::OptionalTensorRef>;
using IOptTensorRefListIterator = c10::IListIterator<at::OptionalTensorRef>;
}
