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
 * Specializations of `IListTagImplBase` that implement the default
 * implementation for `IListTag::Unboxed`.
 */
template <typename T, typename ListElemT>
class IListTagImplBase<IListTag::Unboxed, T, ListElemT> {
 public:
  using elem_type = ListElemT;
  using list_type = ArrayRef<elem_type>;

  /*
   * These `unwrap` static methods unwraps the inner containers out
   * of `IList<T>` (and `IListIterator<T>`). They are required when
   * the macro `TORCH_ILIST_UNWRAP` is called.
   */
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
  static IListConstRef<T> front(const list_type& lst) {
    return lst.front();
  }

  static IListConstRef<T> get(const list_type& lst, size_t i) {
    return lst[i];
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

  static IListConstRef<T> front(const list_type& lst) {
    return lst[0];
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

/*
 * Helper class for converting an `IList<T>` into another tag, while owning
 * extra data, if necessary.
 *
 * What is this for?
 * =================
 * There are some situations where we need a specific container type`.
 * If the `IList<T>` is referencing that container type, we can just return
 * that by calling `IList<T>::to<Tag>()`. Otherwise, we have to create a new
 * owning container of that type, and copy the elements to it.
 *
 * What does it do?
 * ================
 * It optionally creates and owns a new container of type `OwnedT`, if
 * necessary. A reference to it can be accessed by calling
 * `IListMaybeOwn::get`. Notice that the returned type will be the one
 * corresponding to that tag.
 *
 * For example:
 * `IListMaybeOwn<Unboxed, Tensor, std::vector<Tensor>, Tensor>::get()`
 * will return `const ArrayRef<Tensor>&`, even though its owning container
 * type is `std::vector<Tensor>`.
 */
template <
    typename ImplT,
    IListTag TAG,
    typename T,
    typename OwnedT = typename c10::detail::IListTagImpl<TAG, T>::list_type,
    typename ReturnT = const OwnedT&>
class IListMaybeOwn {
 protected:
  using owned_list_type = OwnedT;
  using return_type = ReturnT;

  static typename owned_list_type::value_type map_elem(const T& elem) {
    return elem;
  }

 public:
  IListMaybeOwn(IList<T> ref) : ref_(ref), own_(c10::nullopt) {
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
  IList<T> ref_;
  optional<owned_list_type> own_;
};

/*
 * Helper for converting things into their unboxed representation.
 * Since the unboxed `list_type` does not own the data, we create a
 * `std::vector` instead.
 */
template <typename T>
class MaybeOwnUnboxed : public IListMaybeOwn<
                            MaybeOwnUnboxed<T>,
                            IListTag::Unboxed,
                            T,
                            std::vector<T>,
                            ArrayRef<T>> {
 public:
  using Super = IListMaybeOwn<
      MaybeOwnUnboxed<T>,
      IListTag::Unboxed,
      T,
      std::vector<T>,
      ArrayRef<T>>;
  using Super::IListMaybeOwn;
  using typename Super::return_type;

  static return_type from_ref(IList<T> ilist) {
    return ilist.toUnboxed();
  }

  static bool needs_owning_data(IList<T> ilist) {
    return !ilist.isUnboxed();
  }
};

/*
 * Helper for converting things into their boxed representation.
 */
template <typename T>
class MaybeOwnBoxed
    : public IListMaybeOwn<MaybeOwnBoxed<T>, IListTag::Boxed, T> {
 public:
  using Super = IListMaybeOwn<MaybeOwnBoxed<T>, IListTag::Boxed, T>;
  using Super::IListMaybeOwn;
  using typename Super::return_type;

  static return_type from_ref(IList<OptionalTensorRef> ilist) {
    return ilist.toBoxed();
  }

  static bool needs_owning_data(IList<T> ilist) {
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
class MaybeOwnBoxed<OptionalTensorRef> : public IListMaybeOwn<
                                             MaybeOwnBoxed<OptionalTensorRef>,
                                             IListTag::Boxed,
                                             OptionalTensorRef> {
 public:
  using Super = IListMaybeOwn<
      MaybeOwnBoxed<OptionalTensorRef>,
      IListTag::Boxed,
      OptionalTensorRef>;
  using Super::IListMaybeOwn;
  using typename Super::owned_list_type;
  using typename Super::return_type;

  static return_type from_ref(IList<OptionalTensorRef> ilist) {
    return ilist.toBoxed();
  }

  static bool needs_owning_data(IList<OptionalTensorRef> ilist) {
    return !ilist.isBoxed();
  }

  static typename owned_list_type::value_type map_elem(
      const OptionalTensorRef& elem) {
    return elem.has_value() ? c10::optional<Tensor>(*elem) : c10::nullopt;
  }
};

} // namespace at
