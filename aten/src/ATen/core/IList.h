#pragma once

#include <ATen/core/ivalue_to.h>
#include <c10/util/Exception.h>

#include <initializer_list>
#include <iterator>
#include <type_traits>

/*
 * Adding support for a new Tag
 * ============================
 * Suppose we want to add a new tag: `Chest`. Here are the steps
 * we would have to go through:
 *
 * 1. Add a line for it in the macro `TORCH_ILIST_FORALL_TAGS`.
 *
 *   #define TORCH_ILIST_FORALL_TAGS(_, ...) \
 *     ...
 *     _(Chest, ##__VA_ARGS__)
 *
 * 2. Add type aliases, union members, and constructors.
 *
 *   template <typename T>
 *   class IList {
 *     ...
 *     using chest_type =
 *       typename detail::IListTagImpl<T, IListTag::Chest>::list_type;
 *     ...
 *     IList(...) : tag_(IListTag::Chest) {
 *       ...
 *     }
 *     ...
 *     union Payload {
 *       ...
 *       chest_type chest;
 *       ...
 *     };
 *     ...
 *   };
 *
 * 3. Add a default implementation for it (in 'IList_inl.h'). It's
 *    preferable to make the default implementation work for `T = Tensor`
 *    (both `Unboxed` and `Boxed` do it).
 *
 *   template <typename T, typename ListElemT>
 *   class IListTagImplBase<IListTag::Chest, T, ListElemT> {
 *    public:
 *     using elem_type = ListElemT;
 *     using list_type = ChestContainer<elem_type>;
 *
 *     static const list_type& unwrap(const IList<T>& ilist) { ... }
 *
 *     static typename list_type::const_iterator& unwrap(
 *         IListIterator<T>& it) { ... }
 *
 *     static const typename list_type::const_iterator& unwrap(
 *         const IListIterator<T>& it) { ... }
 *
 *     static IListConstRef<T> iterator_get(
 *         const typename list_type::const_iterator& it) { ... }
 *   }
 *
 * 4. Add an specialization for each of the already supported types.
 *    Finally, for consistency, add them to the tracking list.
 *    (see [Note: IListTagImpl Specializations])
 *
 *   template <>
 *   class IListTagImpl<IListTag::Chest, at::Tensor>
 *       : public IListTagImplBase<IListTag::Chest, at::Tensor> {};
 *
 * Adding support for a new Type
 * =============================
 * Suppose we want to add support for a new type: `Matrix`.
 * Here are the steps we would have to go through:
 *
 * 1. Add an specialization for each of the existing tags.
 *    For consistency, add them to the tracking list.
 *    (see [Note: IListTagImpl Specializations])
 *
 *   template <>
 *   class IListTagImpl<IListTag::Unboxed, Matrix>
 *       : public IListTagImplBase<IListTag::Unboxed, Matrix> {};
 *
 *   template <>
 *   class IListTagImpl<Matrix, IListTag::Boxed>
 *       : public IListTagImplBase<IListTag::Boxed, Matrix> {};
 *
 * Common Problems
 * ===============
 * 1. One of `IList(Iterator)` methods are failing to compile.
 *
 *     That may be happening because the container type you added
 *     is not compatible with the code written for that method. If
 *     that's true, then you might have to transform that code into
 *     a static method call (see `List::operator[]` method).
 *
 * 2. Can't make `IList<T>::operator[]` return a const-reference.
 *
 *    First, keep in mind that we assume that boxed containers will
 *    have to deal with `IValue` (e.g. `c10::List`). In this context,
 *    what may be happening is that `IValue` doesn't store internally
 *    your type `T`. Instead, it constructs a type new `T` everytime
 *    you try to get `T` for it (see `IList<at::OptinalTensorRef>`).
 */

namespace c10 {
/*
 * Applies arbitrary macros to each `IListTag`.
 */
#define TORCH_ILIST_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)             \
  _(Boxed, ##__VA_ARGS__)

/*
 * Defines a "switch-case" for `TAG`. Inside, it executes `BODY`,
 * while bringing to scope:
 *
 * - `ImplT`: the implementation class for `TAG`
 * - `this_`: the result of unwrapping `this`
 */
#define TORCH_ILIST_UNWRAP_CASE(TAG, BODY)                     \
  case c10::IListTag::TAG: {                                   \
    using ImplT = c10::detail::IListTagImpl<IListTag::TAG, T>; \
    auto& this_ = ImplT::unwrap(*this);                        \
    BODY                                                       \
  } break;

/*
 * Dispatches the unwrap call, depending on `TAG`, followed by
 * the execution of `BODY`. It aborts if `TAG` is not a `IListTag`.
 *
 * This macro is useful because it allows us to handle different
 * types (that correspond to different tags) to be implemented
 * only once. We can do it even when the implementation of the
 * different tags aren't syntatically the same, by dispatching
 * it to a function (e.g. `ImplT::<dispatch-function>(this_)`).
 */
#define TORCH_ILIST_UNWRAP(TAG, BODY)                      \
  switch (TAG) {                                           \
    TORCH_ILIST_FORALL_TAGS(TORCH_ILIST_UNWRAP_CASE, BODY) \
    break;                                                 \
    default:                                               \
      TORCH_INTERNAL_ASSERT(false, "invalid IList tag.");  \
  }

enum class IListTag {
#define DEFINE_TAG(tag, ...) tag,
  TORCH_ILIST_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
      None
};

namespace detail {
/*
 * Type alias that specifies whether we return a reference or a copy of `T`.
 *
 * What is this for?
 * =================
 * Since values in the boxed world are represented by an `IValue`, we also
 * depend on whether it can be converted to a const-reference (`Tensor`) or
 * has to create a new copy of `T` (`OptionalTensorRef`).
 */
template <typename T>
using IListConstRef = typename ivalue_to_const_ref_overload_return<T>::type;

/*
 * Interface that implements key functions for each `IListTag` type.
 *
 * What is this for?
 * =================
 * Given an `IList(Iterator)<T>`, some methods have to be implemented
 * differently for each `TAG`. Therefore, the methods inside this class
 * are used as dispatch targets for the different `IListTag` values.
 *
 * You should create an specialization of this class for each possible
 * combination of `IListTag` type (except `None`) and element types
 * (e.g. `Tensor`).
 *
 * What does it do?
 * ================
 * 1. defines static methods to be used as dispatch targets by both
 *    `IList<T>` and `IListIterator<T>` (see the implementation of
 *    `IListTagImplBase`).
 *
 * 2. defines the `elem_type` and `list_type` aliases that will be
 *    used in the definition of `IList<T>`. In general, we should do
 *    so by inheriting from `IListTagImplBase<TAG, T, ListElemT>`.
 *
 * Existing Specializations
 * ========================
 * [Note: IListTagImpl Specialization]
 *
 * For `IList(Iterator)<at::Tensor>`:
 * - <IListTag::Unboxed, at::Tensor>
 * - <IListTag::Boxed, at::Tensor>
 *
 * For `IList(Iterator)<at::OptionalTensorRef>`:
 * - <IListTag::Unboxed, at::OptionalTensorRef>
 * - <IListTag::Boxed, at::OptionalTensorRef>
 */
template <IListTag TAG, typename T>
class IListTagImpl {};

/*
 * Base implementation of `IListTagImpl<TAG, T>` methods.
 *
 * What is this for?
 * =================
 * This should make adding specializations for new types easier. For
 * example, one should be able to add a new type just by making its
 * `IListTagImpl` specialization inherit from `IListTagImplBase`.
 *
 * You should create a partial specialization for this class only if
 * you introduce a new `IListTag`. The idea being that there is one
 * default implementation for each possible value of `IListTag`.
 *
 * What does it do?
 * ================
 * 1. defines `elem_type` as an alias to `ListElemT`.
 *
 * 1. defines `list_type` as an alias to the default container type
 *    that will hold a collection of `elem_type`. The idea being that
 *    all types tagged as `TAG` will have `list_type` as its container,
 *    with different `elem_type`.
 *
 * 3. defines the default implementation for each of the methods that
 *    are supposed to be defined on `IListTagImpl` specializations.
 *
 * 4. inheriting from `IListTagImplBase<TAG, T, ListElemT>` also means
 *    that the payload of the type `IList<T>` will be of type `list_type`
 *    when it is tagged as `TAG`.
 */
template <IListTag TAG, typename T, typename ListElemT = T>
class IListTagImplBase {};

} // namespace detail

/*
 * Iterator for `IList<T>`.
 *
 * What is it?
 * ===========
 * Currently, a `std::bidirectional_iterator` that wraps the iterator
 * types defined for each of the `IListTag`.
 *
 * One should be able to use it, as if it were the unwrapped
 * iterators themselves.

 * What does it do?
 * ================
 * Similarly to `IList<T>`, this is a wrapper class. Specifically, it
 * wraps each container's `const_iterator` type alias. So, for example,
 * given that the container for `IListTag::Boxed` is `c10::List`, this
 * iterator will wrap a `c10::List::const_iterator`.
 */
template <typename T>
class IListIterator : public std::iterator<std::bidirectional_iterator_tag, T> {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...)                  \
  friend class detail::IListTagImpl<IListTag::TAG, T>; \
  friend class detail::IListTagImplBase<               \
      IListTag::TAG,                                   \
      T,                                               \
      typename detail::IListTagImpl<IListTag::TAG, T>::elem_type>;
  TORCH_ILIST_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

 public:
  using unboxed_iterator_type = typename detail::
      IListTagImpl<IListTag::Unboxed, T>::list_type::const_iterator;
  using boxed_iterator_type = typename detail::
      IListTagImpl<IListTag::Boxed, T>::list_type::const_iterator;

  IListIterator() : tag_(IListTag::None) {}

  IListIterator(boxed_iterator_type boxed) : tag_(IListTag::Boxed) {
    payload_.boxed_iterator = boxed;
  }

  IListIterator(unboxed_iterator_type unboxed) : tag_(IListTag::Unboxed) {
    payload_.unboxed_iterator = unboxed;
  }

  detail::IListConstRef<T> operator*() const {
    TORCH_ILIST_UNWRAP(tag_, { return ImplT::iterator_get(this_); });
  }

  IListIterator& operator++() {
    TORCH_ILIST_UNWRAP(tag_, { ++this_; });
    return *this;
  }

  IListIterator operator++(int) {
    auto old = *this;
    TORCH_ILIST_UNWRAP(tag_, { ++this_; });
    return old;
  }

  IListIterator& operator--() {
    TORCH_ILIST_UNWRAP(tag_, { --this_; });
    return *this;
  }

  IListIterator operator--(int) {
    auto old = *this;
    TORCH_ILIST_UNWRAP(tag_, { --this_; });
    return old;
  }

  bool operator==(const IListIterator& rhs) const {
    if (tag_ != rhs.tag_) {
      return false;
    }
    TORCH_ILIST_UNWRAP(tag_, {
      auto& rhs_it = ImplT::unwrap(rhs);
      return this_ == rhs_it;
    });
  }

  bool operator!=(const IListIterator& rhs) const {
    return !(*this == rhs);
  }

 private:
  union Payload {
    boxed_iterator_type boxed_iterator;
    unboxed_iterator_type unboxed_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}
    ~Payload() = default;
  };

  Payload payload_;
  IListTag tag_;
};

/*
 * Wrapper around different API containers (e.g. boxed and unboxed).
 *
 * What is it?
 * ===========
 * It is a tagged union of both boxed and unboxed API containers.
 * Working implementations:
 *
 * - `IList<at::Tensor>`
 * - `IList<at::OptionalTensorRef>`
 *
 * Note that `ITensorList` is a view type. Meaning that it won't own the
 * tensors it holds. If you need it to last longer, make sure that there is
 * actually a non-temporary list of tensors (e.g. `vector<Tensor>`) that owns
 * them and outlives the `ITensorList` instance.
 *
 * What is this for?
 * =================
 * Historically, PyTorch has maintained 2 different APIs: the unboxed
 * (called from C++ API and Python eager mode) and boxed APIs (called
 * from the TorchScript JIT, mobile interpreter, and boxed fallbacks).
 *
 * Calling unboxed kernels from the boxed "world" and vice-versa may
 * result in non-negligible overhead. Lists are one of those types:
 *
 * - Boxed world: `c10::List`
 * - Unboxed world: `c10::ArrayRef`
 *
 * In this context, `c10::IList` solves this problem by wrapping those
 * 2 container types, so that we don't need to convert from one to
 * the other.
 *
 * (see https://github.com/pytorch/pytorch/issues/66328)
 *
 * What does it do?
 * ================
 * This container wraps around the different tagged containers
 * (currently, only boxed and unboxed), without incurring in extra
 * overhead for converting from one to another. It does so while
 * exposing usual container methods, which dispatch to corresponding
 * implementations.
 */
template <typename T>
class IList {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...)                  \
  friend class detail::IListTagImpl<IListTag::TAG, T>; \
  friend class detail::IListTagImplBase<               \
      IListTag::TAG,                                   \
      T,                                               \
      typename detail::IListTagImpl<IListTag::TAG, T>::elem_type>;
  TORCH_ILIST_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

 public:
  using unboxed_type =
      typename detail::IListTagImpl<IListTag::Unboxed, T>::list_type;
  using boxed_type =
      typename detail::IListTagImpl<IListTag::Boxed, T>::list_type;

  using iterator = IListIterator<T>;
  using const_iterator = IListIterator<T>;
  using value_type = typename iterator::value_type;

  IList() : tag_(IListTag::None) {}

  IList(const boxed_type& boxed) : tag_(IListTag::Boxed) {
    payload_.boxed = &boxed;
  }

  IList(const unboxed_type& unboxed) : tag_(IListTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  /*
   * Enable only 2 implicit constructors:
   * 1. IList(const std::vector<T>&)
   * 2. IList(std::initializer_list<T>)
   *
   * These are only enabled if the `unboxed_type` also has a constructor
   * for those types.
   */
  IList(
      const std::vector<T>& vec,
      std::enable_if_t<
          std::is_constructible<unboxed_type, std::vector<T>>::value,
          std::nullptr_t> = nullptr)
      : tag_(IListTag::Unboxed) {
    payload_.unboxed = unboxed_type(vec);
  }

  constexpr IList(
      std::initializer_list<T> il,
      std::enable_if_t<
          std::is_constructible<unboxed_type, std::initializer_list<T>>::value,
          std::nullptr_t> = nullptr)
      : tag_(IListTag::Unboxed) {
    payload_.unboxed = unboxed_type(il);
  }

  /*
   * For convenience, enable other supported `unboxed_type` constructors as
   * explicit constructors.
   */
  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible<unboxed_type, UnboxedConstructorArgs...>::value>>
  explicit IList(UnboxedConstructorArgs&&... args) : tag_(IListTag::Unboxed) {
    payload_.unboxed = unboxed_type(std::forward<UnboxedConstructorArgs>(args)...);
  }

  size_t size() const {
    TORCH_ILIST_UNWRAP(tag_, { return this_.size(); });
  }

  bool empty() const {
    return size() == 0;
  }

  iterator begin() const {
    TORCH_ILIST_UNWRAP(tag_, { return this_.begin(); });
  }

  iterator end() const {
    TORCH_ILIST_UNWRAP(tag_, { return this_.end(); });
  }

  detail::IListConstRef<T> front() const {
    TORCH_ILIST_UNWRAP(tag_, { return ImplT::front(this_); });
  }

  detail::IListConstRef<T> operator[](size_t i) const {
    TORCH_ILIST_UNWRAP(tag_, { return ImplT::get(this_, i); });
  }

  std::vector<T> vec() const {
    return {begin(), end()};
  }

#define DEFINE_CHECK(TAG, ...)    \
  bool is##TAG() const {          \
    return tag_ == IListTag::TAG; \
  }
  TORCH_ILIST_FORALL_TAGS(DEFINE_CHECK);
#undef DEFINE_CHECK

  bool isNone() const {
    return tag_ == IListTag::None;
  }

#define DEFINE_CASTING(TAG, ...)                                              \
  const typename detail::IListTagImpl<IListTag::TAG, T>::list_type& to##TAG() \
      const {                                                                 \
    TORCH_INTERNAL_ASSERT(is##TAG());                                         \
    return detail::IListTagImpl<IListTag::TAG, T>::unwrap(*this);             \
  }
  TORCH_ILIST_FORALL_TAGS(DEFINE_CASTING);
#undef DEFINE_CASTING

 private:
  union Payload {
    const boxed_type* boxed;
    unboxed_type unboxed;
    Payload() : boxed(nullptr) {}
    ~Payload() = default;
  };

  Payload payload_;
  IListTag tag_;
};

} // namespace c10

#include <ATen/core/IList_inl.h>
