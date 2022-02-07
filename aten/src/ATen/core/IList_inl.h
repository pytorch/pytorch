#pragma once

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <type_traits>

namespace at {
class Tensor;
class OptionalTensorRef;

/*
 * Temporary ArrayRef<T> class.
 *
 * What is this for?
 * =================
 * Conveniently provides an 'ArrayRef<T>' out of a 'IList<T>'. It tries not to
 * copy, but will do so if 'IList' is not unboxed.
 *
 * This is a workaround for, mainly 'MatrixRef'. It needs 'ArrayRef' methods,
 * such as 'slice', which is not possible to implement for 'List' (boxed_type).
 */
template <typename T>
class TempArrayRef {
 public:
  explicit TempArrayRef(IList<T> il) {
    constexpr bool is_unboxed_arrayref = std::is_same<ArrayRef<T>, typename c10::IList<T>::unboxed_type>::value;
    bool should_own_list = !is_unboxed_arrayref || !il.isUnboxed();
    data_ = should_own_list ? c10::optional<std::vector<T>>({il.begin(), il.end()}) : c10::nullopt;
    arr_ = should_own_list ? data_.value() : il.toUnboxed();
  }

  ArrayRef<T> operator*() const {
    return arr_;
  }

 private:
  ArrayRef<T> arr_;
  c10::optional<std::vector<T>> data_;
};
}

namespace c10 {
namespace detail {

/*
 * Specializations of `IListTagImplBase` that implement the default
 * implementation for `IListTag::Unboxed`.
 */
template <typename T, typename ListElemT>
class IListTagImplBase<IListTag::Unboxed, T, ListElemT> {
 public:
  using elem_type = ListElemT;
  using list_type = ArrayRef<elem_type>;

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
   * We have these function (besides the `unwrap`s above) because the
   * implementation for both `IList::operator[]` and `IListIterator::operator*`
   * weren't syntatically equal for the existing tags at the time
   * (`Unboxed` and `Boxed`).
   */
  static IListConstRef<T> get(const list_type& it, size_t i) {
    return it[i];
  }

  static IListConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return *it;
  }
};

/*
 * Specializations of `IListTagImplBase` that implement the default
 * implementation for `IListTag::Boxed`.
 */
template <typename T, typename ListElemT>
class IListTagImplBase<IListTag::Boxed, T, ListElemT> {
 public:
  using elem_type = ListElemT;
  using list_type = List<elem_type>;

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

  static IListConstRef<T> get(const list_type& it, size_t i) {
    return it[i];
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
class IListTagImpl<IListTag::Unboxed, at::Tensor>
    : public IListTagImplBase<IListTag::Unboxed, at::Tensor> {};

template <>
class IListTagImpl<IListTag::Boxed, at::Tensor>
    : public IListTagImplBase<IListTag::Boxed, at::Tensor> {};

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
class IListTagImpl<IListTag::Unboxed, at::OptionalTensorRef>
    : public IListTagImplBase<IListTag::Unboxed, at::OptionalTensorRef> {};

template <>
class IListTagImpl<IListTag::Boxed, at::OptionalTensorRef>
    : public IListTagImplBase<IListTag::Boxed, at::OptionalTensorRef, optional<at::Tensor>> {

 public:
  /*
   * Given an instance of the types corresponding to the `Boxed` tag, we override
   * the default implementation, so that we can return a `at::OptionalTensorRef`.
   */
  static IListConstRef<at::OptionalTensorRef> get(const list_type& list, size_t i) {
    const auto& opt = list[i];
    return (opt.has_value()) ? *opt : at::OptionalTensorRef{};
  }

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
