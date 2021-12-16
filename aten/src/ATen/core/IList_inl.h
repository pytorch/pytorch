#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/IList.h>

namespace at {
class Tensor;
}

namespace c10 {
namespace detail {

/*
 * Specializations of `IListTagImplBase` that implement the default
 * implementation for `IListTag::Unboxed`.
 */
template <typename ListT, typename T>
class IListTagImplBase<ListT, T, IListTag::Unboxed> {
 public:
  using list_type = ListT;

  static const list_type& unwrap(const IList<T>& ilist) {
    return ilist.payload_.unboxed;
  }

  static typename list_type::const_iterator& unwrap(IListIterator<T>& it) {
    return it.payload_.unboxed_iterator;
  }

  static const typename list_type::const_iterator& unwrap(
      const IListIterator<T>& it) {
    return it.payload_.unboxed_iterator;
  }

  /*
   * We have this function (besides the `unwrap`s above) because the
   * implementation for `IListIterator::operator*` wasn't syntatically
   * equal for the existing tags at the time (`Unboxed` and `Boxed`).
   */
  static IListConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return *it;
  }
};

/*
 * Specializations of `IListTagImplBase` that implement the default
 * implementation for `IListTag::Boxed`.
 */
template <typename ListT, typename T>
class IListTagImplBase<ListT, T, IListTag::Boxed> {
 public:
  using list_type = ListT;

  static const list_type& unwrap(const IList<T>& ilist) {
    return *ilist.payload_.boxed;
  }

  static typename list_type::const_iterator& unwrap(IListIterator<T>& it) {
    return it.payload_.boxed_iterator;
  }

  static const typename list_type::const_iterator& unwrap(
      const IListIterator<T>& it) {
    return it.payload_.boxed_iterator;
  }

  static IListConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return (*it).get().toTensor();
  }
};

/*
 * [Note: ITensorList]
 * Specializations necessary for `IList<at::Tensor>` type.
 *
 * Since the default implementations are usually done with supporting
 * `Tensor` in mind, we only have to inherit from the base implementations.
 */
template <>
class IListTagImpl<at::Tensor, IListTag::Unboxed>
    : public IListTagImplBase<
    ArrayRef<at::Tensor>, at::Tensor, IListTag::Unboxed> {};

template <>
class IListTagImpl<at::Tensor, IListTag::Boxed>
    : public IListTagImplBase<
    List<at::Tensor>, at::Tensor, IListTag::Boxed> {};

/*
 * [Note: IOptTensorRefList]
 * Specializations necessary for `IList<at::OptionalTensorRef>` type.
 *
 * We can't get an `at::OptionalTensorRef` directly from an instance of
 * `List<optional<Tensor>>` (the type that corresponds to the boxed world).
 *
 * So, the default implementation won't help us. Thus, we have to implement
 * this method ourselves.
 */
template <>
class IListTagImpl<at::OptionalTensorRef, IListTag::Unboxed>
    : public IListTagImplBase<
    ArrayRef<at::OptionalTensorRef>, at::OptionalTensorRef, IListTag::Unboxed> {};

template <>
class IListTagImpl<at::OptionalTensorRef, IListTag::Boxed>
    : public IListTagImplBase<
    List<optional<at::Tensor>>, at::OptionalTensorRef, IListTag::Boxed> {

 public:
  /*
   * Given an iterator type corresponding to the `Boxed` tag, we override
   * the default implementation, so that we can return a `at::OptionalTensorRef`.
   */
  static IListConstRef<at::OptionalTensorRef> iterator_get(
      const typename list_type::const_iterator& it) {
    const auto& ivalue = (*it).get();
    return (ivalue.isNone()) ? at::OptionalTensorRef{} : ivalue.toTensor();
  }

};

} // namespace detail
} // namespace c10

namespace at {

// [Note: ITensorList]
using ITensorList = c10::IList<at::Tensor>;
using ITensorListIterator = c10::IListIterator<at::Tensor>;
// [Note: IOptTensorRefList]
using IOptTensorRefList = c10::IList<at::OptionalTensorRef>;
using IOptTensorRefListIterator = c10::IListIterator<at::OptionalTensorRef>;

} // namespace at
