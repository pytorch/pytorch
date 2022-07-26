#pragma once

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

namespace at {
class Tensor;
class OptionalTensorRef;
} // namespace at

namespace c10 {
namespace detail {

/*
 * Specializations of `IListRefTagImplBase` that implement the default
 * implementation for `IListRefTag::Unboxed`.
 */
template <typename T, typename ListElemT>
class IListRefTagImplBase<IListRefTag::Unboxed, T, ListElemT> {
 public:
  using elem_type = ListElemT;
  using list_type = ArrayRef<elem_type>;

  /*
   * These `unwrap` static methods unwraps the inner containers out
   * of `IListRef<T>` (and `IListRefIterator<T>`). They are required when
   * the macro `TORCH_ILISTREF_UNWRAP` is called.
   */
  static const list_type& unwrap(const IListRef<T>& ilist) {
    return ilist.payload_.unboxed;
  }

  static typename list_type::const_iterator& unwrap(IListRefIterator<T>& it) {
    return it.payload_.unboxed_iterator;
  }

  static const typename list_type::const_iterator& unwrap(
      const IListRefIterator<T>& it) {
    return it.payload_.unboxed_iterator;
  }

  /*
   * We have these function (besides the `unwrap`s above) because the
   * implementation for both `IListRef::operator[]` and `IListRefIterator::operator*`
   * weren't syntatically equal for the existing tags at the time
   * (`Unboxed` and `Boxed`).
   */
  static IListRefConstRef<T> front(const list_type& lst) {
    return lst.front();
  }

  static IListRefConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return *it;
  }
};

/*
 * Specializations of `IListRefTagImplBase` that implement the default
 * implementation for `IListRefTag::Boxed`.
 */
template <typename T, typename ListElemT>
class IListRefTagImplBase<IListRefTag::Boxed, T, ListElemT> {
 public:
  using elem_type = ListElemT;
  using list_type = List<elem_type>;

  static const list_type& unwrap(const IListRef<T>& ilist) {
    return *ilist.payload_.boxed;
  }

  static typename list_type::const_iterator& unwrap(IListRefIterator<T>& it) {
    return it.payload_.boxed_iterator;
  }

  static const typename list_type::const_iterator& unwrap(
      const IListRefIterator<T>& it) {
    return it.payload_.boxed_iterator;
  }

  static IListRefConstRef<T> front(const list_type& lst) {
    return lst[0];
  }

  static IListRefConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return (*it).get().toTensor();
  }
};

/*
 * Specializations of `IListRefTagImplBase` that implement the default
 * implementation for `IListRefTag::Materialized`.
 */
template <typename T>
class IListRefTagImplBase<IListRefTag::Materialized, T, MaterializedIListRefElem<T>> {
 public:
  using elem_type = MaterializedIListRefElem<T>;
  using list_type = MaterializedIListRef<T>;

  static const list_type& unwrap(const IListRef<T>& ilist) {
    return *ilist.payload_.materialized;
  }

  static typename list_type::const_iterator& unwrap(IListRefIterator<T>& it) {
    return it.payload_.materialized_iterator;
  }

  static const typename list_type::const_iterator& unwrap(
      const IListRefIterator<T>& it) {
    return it.payload_.materialized_iterator;
  }

  static IListRefConstRef<T> front(const list_type& lst) {
    return lst[0];
  }

  static IListRefConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return *it;
  }
};

/*
 * [Note: ITensorListRef]
 * Specializations necessary for `IListRef<at::Tensor>` type.
 *
 * Since the default implementations are usually done with supporting
 * `Tensor` in mind, we only have to inherit from the base implementations.
 */
template <>
class IListRefTagImpl<IListRefTag::Unboxed, at::Tensor>
    : public IListRefTagImplBase<IListRefTag::Unboxed, at::Tensor> {};

template <>
class IListRefTagImpl<IListRefTag::Boxed, at::Tensor>
    : public IListRefTagImplBase<IListRefTag::Boxed, at::Tensor> {};

template <>
class IListRefTagImpl<IListRefTag::Materialized, at::Tensor>
    : public IListRefTagImplBase<
          IListRefTag::Materialized,
          at::Tensor,
          MaterializedIListRefElem<at::Tensor>> {};

/*
 * [Note: IOptTensorListRef]
 * Specializations necessary for `IListRef<at::OptionalTensorRef>` type.
 *
 * We can't get an `at::OptionalTensorRef` directly from an instance of
 * `List<optional<Tensor>>` (the type that corresponds to the boxed world).
 *
 * So, the default implementation won't help us. Thus, we have to implement
 * this method ourselves.
 */
template <>
class IListRefTagImpl<IListRefTag::Unboxed, at::OptionalTensorRef>
    : public IListRefTagImplBase<IListRefTag::Unboxed, at::OptionalTensorRef> {};

template <>
class IListRefTagImpl<IListRefTag::Boxed, at::OptionalTensorRef>
    : public IListRefTagImplBase<IListRefTag::Boxed, at::OptionalTensorRef, optional<at::Tensor>> {

 public:
  /*
   * Given an instance of the types corresponding to the `Boxed` tag, we override
   * the default implementation, so that we can return a `at::OptionalTensorRef`.
   */
  static IListRefConstRef<at::OptionalTensorRef> iterator_get(
      const typename list_type::const_iterator& it) {
    const auto& ivalue = (*it).get();
    if (!ivalue.isNone()) {
        const auto& tensor = ivalue.toTensor();
        return (tensor.defined()) ? tensor : at::OptionalTensorRef{};
    }
    return {};
  }
};

template <>
class IListRefTagImpl<IListRefTag::Materialized, at::OptionalTensorRef>
    : public IListRefTagImplBase<
          IListRefTag::Materialized,
          at::OptionalTensorRef,
          MaterializedIListRefElem<at::OptionalTensorRef>> {};

} // namespace detail
} // namespace c10

namespace at {

// [Note: ITensorListRef]
using ITensorListRef = c10::IListRef<at::Tensor>;
using ITensorListRefIterator = c10::IListRefIterator<at::Tensor>;
using MaterializedITensorListRef = c10::detail::MaterializedIListRef<at::Tensor>;
// [Note: IOptTensorListRef]
using IOptTensorListRef = c10::IListRef<at::OptionalTensorRef>;
using IOptTensorListRefIterator = c10::IListRefIterator<at::OptionalTensorRef>;
using MaterializedIOptTensorListRef = c10::detail::MaterializedIListRef<at::OptionalTensorRef>;

/*
 * Helper class for converting an `IOptTensorListRef` into its boxed container.
 *
 * What is this for?
 * =================
 * There are some situations where we need the boxed container of
 * `IOptTensorListRef`. If it already is in its boxed form, we can just return
 * that by calling `IOptTensorListRef::toBoxed()`. Otherwise, we have to
 * create a new boxed container, and copy the elements to it.
 *
 * What does it do?
 * ================
 * It optionally creates and owns a new boxed container. A reference
 * to it can be accessed by calling `IListMaybeIntoBoxed::get`.
 */
class IOptTensorListRefMaybeOwnBoxed {
 private:
  using IntoT = typename IOptTensorListRef::boxed_type;

 public:
  IOptTensorListRefMaybeOwnBoxed(IOptTensorListRef ref) : ref_(ref) {
    if (!ref.isBoxed()) {
      own_ = IntoT();
      own_->reserve(ref.size());
      for (const auto& t : ref) {
          own_->push_back(t.has_value() ? optional<at::Tensor>(*t) : nullopt);
      }
    } else {
      own_ = nullopt;
    }
  }

  const IntoT& get() const {
    return ref_.isBoxed() ? ref_.toBoxed() : own_.value();
  }

 private:
  IOptTensorListRef ref_;
  optional<IntoT> own_;
};

} // namespace at
