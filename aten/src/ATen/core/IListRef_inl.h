#pragma once

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <type_traits>

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
class IListRefTagImplBase<IListRefTag::Materialized, T, _MaterializedIListRefElem<T>> {
 public:
  using elem_type = _MaterializedIListRefElem<T>;
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
          _MaterializedIListRefElem<at::Tensor>> {};

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
          _MaterializedIListRefElem<at::OptionalTensorRef>> {};

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
 * Helper class for converting an `IListRef<T>` into another tag, while owning
 * extra data, if necessary.
 *
 * What is this for?
 * =================
 * There are some situations where we need a specific container type`.
 * If the `IListRef<T>` is referencing that container type, we can just return
 * that by calling `IListRef<T>::to<Tag>()`. Otherwise, we have to create a new
 * owning container of that type, and copy the elements to it.
 *
 * What does it do?
 * ================
 * It optionally creates and owns a new container of type `OwnedT`, if
 * necessary. A reference to it can be accessed by calling
 * `IListRefMaybeOwn::get`. Notice that the returned type will be the one
 * corresponding to that tag.
 *
 * For example:
 * `IListRefMaybeOwn<Unboxed, Tensor, std::vector<Tensor>, Tensor>::get()`
 * will return `const ArrayRef<Tensor>&`, even though its owning container
 * type is `std::vector<Tensor>`.
 */
template <
    typename ImplT,
    IListRefTag TAG,
    typename T,
    typename OwnedT = typename c10::detail::IListRefTagImpl<TAG, T>::list_type,
    typename ReturnT = const OwnedT&>
class IListRefMaybeOwn {
 protected:
  using owned_list_type = OwnedT;
  using return_type = ReturnT;

  static typename owned_list_type::value_type map_elem(const T& elem) {
    return elem;
  }

 public:
  IListRefMaybeOwn(IListRef<T> ref) : ref_(ref), own_(c10::nullopt) {
    if (ImplT::needs_owning_data(ref)) {
      own_ = owned_list_type();
      own_->reserve(ref.size());
      for (const auto& elem : ref) {
        own_->emplace_back(ImplT::map_elem(elem));
      }
    } else {
      own_ = nullopt;
    }
  }

  return_type get() const {
    return ImplT::needs_owning_data(ref_) ? *own_ : ImplT::from_ref(ref_);
  }
  return_type operator*() const {
    return get();
  }

 protected:
  IListRef<T> ref_;
  optional<owned_list_type> own_;
};

/*
 * Helper for converting things into their unboxed representation.
 * Since the unboxed `list_type` does not own the data, we create a
 * `std::vector` instead.
 */
template <typename T>
class MaybeOwnUnboxed : public IListRefMaybeOwn<
                            MaybeOwnUnboxed<T>,
                            IListRefTag::Unboxed,
                            T,
                            std::vector<T>,
                            ArrayRef<T>> {
 public:
  using Super = IListRefMaybeOwn<
      MaybeOwnUnboxed<T>,
      IListRefTag::Unboxed,
      T,
      std::vector<T>,
      ArrayRef<T>>;
  using Super::IListRefMaybeOwn;
  using typename Super::return_type;

  static return_type from_ref(IListRef<T> ilist) {
    return ilist.toUnboxed();
  }

  static bool needs_owning_data(IListRef<T> ilist) {
    return !ilist.isUnboxed();
  }
};

/*
 * Helper for converting things into their boxed representation.
 */
template <typename T>
class MaybeOwnBoxed
    : public IListRefMaybeOwn<MaybeOwnBoxed<T>, IListRefTag::Boxed, T> {
 public:
  using Super = IListRefMaybeOwn<MaybeOwnBoxed<T>, IListRefTag::Boxed, T>;
  using Super::IListRefMaybeOwn;
  using typename Super::return_type;

  static return_type from_ref(IListRef<T> ilist) {
    return ilist.toBoxed();
  }

  static bool needs_owning_data(IListRef<T> ilist) {
    return !ilist.isBoxed();
  }
};

/*
 * Helper for converting `IOptTensorRefList` into its boxed representation.
 * We need this specialization, since its boxed and unboxed representations
 * store different element types.
 * For naming consitency, we specialize instead of inherit.
 */
template <>
class MaybeOwnBoxed<OptionalTensorRef> : public IListRefMaybeOwn<
                                             MaybeOwnBoxed<OptionalTensorRef>,
                                             IListRefTag::Boxed,
                                             OptionalTensorRef> {
 public:
  using Super = IListRefMaybeOwn<
      MaybeOwnBoxed<OptionalTensorRef>,
      IListRefTag::Boxed,
      OptionalTensorRef>;
  using Super::IListRefMaybeOwn;
  using typename Super::owned_list_type;
  using typename Super::return_type;

  static return_type from_ref(IListRef<OptionalTensorRef> ilist) {
    return ilist.toBoxed();
  }

  static bool needs_owning_data(IListRef<OptionalTensorRef> ilist) {
    return !ilist.isBoxed();
  }

  static typename owned_list_type::value_type map_elem(
      const OptionalTensorRef& elem) {
    return elem.has_value() ? c10::optional<Tensor>(*elem) : c10::nullopt;
  }
};

} // namespace at
