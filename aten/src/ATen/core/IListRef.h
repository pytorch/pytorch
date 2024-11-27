#pragma once

#include <ATen/core/ivalue_to.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>

/*
 * [Note: IListRef]
 * Wrapper around different API containers (e.g. boxed and unboxed).
 *
 * What is it?
 * ===========
 * It is a tagged union of both boxed and unboxed API containers.
 * Working implementations:
 *
 * - `IListRef<at::Tensor>`
 * - `IListRef<at::OptionalTensorRef>`
 *
 * Note that `IListRef` is a view type. Meaning that it won't own the
 * tensors it holds. It's intended to be used only as argument parameters.
 * Specifically, where these 2 worlds overlap.
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
 * In this context, `c10::IListRef` solves this problem by wrapping those
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
 *
 * While it works with different container types, it introduces
 * overhead for repeatedly calling member functions (since those will
 * get dispatched, again). Therefore, you should only use it to iterate
 * through the list up to one time. If you need to do more complex things,
 * call `materialize()` first.
 *
 * Adding support for a new Tag
 * ============================
 * Suppose we want to add a new tag: `Chest`. Here are the steps
 * we would have to go through:
 *
 * 1. Add a line for it in the macro `TORCH_ILISTREF_FORALL_TAGS`.
 *
 *   #define TORCH_ILISTREF_FORALL_TAGS(_, ...) \
 *     ...
 *     _(Chest, ##__VA_ARGS__)
 *
 * 2. Add type aliases, union members, and constructors.
 *
 *   template <typename T>
 *   class IListRef {
 *     ...
 *     using chest_type =
 *       typename detail::IListRefTagImpl<T, IListRefTag::Chest>::list_type;
 *     ...
 *     IListRef(...) : tag_(IListRefTag::Chest) {
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
 * 3. Add a default implementation for it (in 'IListRef_inl.h'). It's
 *    preferable to make the default implementation work for `T = Tensor`
 *    (both `Unboxed` and `Boxed` do it).
 *
 *   template <typename T, typename ListElemT>
 *   class IListRefTagImplBase<IListRefTag::Chest, T, ListElemT> {
 *    public:
 *     using elem_type = ListElemT;
 *     using list_type = ChestContainer<elem_type>;
 *
 *     static const list_type& unwrap(const IListRef<T>& ilist) { ... }
 *
 *     static typename list_type::const_iterator& unwrap(
 *         IListRefIterator<T>& it) { ... }
 *
 *     static const typename list_type::const_iterator& unwrap(
 *         const IListRefIterator<T>& it) { ... }
 *
 *     static IListRefConstRef<T> iterator_get(
 *         const typename list_type::const_iterator& it) { ... }
 *   }
 *
 * 4. Add an specialization for each of the already supported types.
 *    Finally, for consistency, add them to the tracking list.
 *    (see [Note: IListRefTagImpl Specializations])
 *
 *   template <>
 *   class IListRefTagImpl<IListRefTag::Chest, at::Tensor>
 *       : public IListRefTagImplBase<IListRefTag::Chest, at::Tensor> {};
 *
 * Adding support for a new Type
 * =============================
 * Suppose we want to add support for a new type: `Matrix`.
 * Here are the steps we would have to go through:
 *
 * 1. Add an specialization for each of the existing tags.
 *    For consistency, add them to the tracking list.
 *    (see [Note: IListRefTagImpl Specializations])
 *
 *   template <>
 *   class IListRefTagImpl<IListRefTag::Unboxed, Matrix>
 *       : public IListRefTagImplBase<IListRefTag::Unboxed, Matrix> {};
 *
 *   template <>
 *   class IListRefTagImpl<Matrix, IListRefTag::Boxed>
 *       : public IListRefTagImplBase<IListRefTag::Boxed, Matrix> {};
 *
 * Common Problems
 * ===============
 * 1. One of `IListRef(Iterator)` methods are failing to compile.
 *
 *     That may be happening because the container type you added
 *     is not compatible with the code written for that method. If
 *     that's true, then you might have to transform that code into
 *     a static method call (see `List::operator[]` method).
 *
 * 2. Can't make `IListRefIterator<T>::operator*` return a const-reference.
 *
 *    First, keep in mind that we assume that boxed containers will
 *    have to deal with `IValue` (e.g. `c10::List`). In this context,
 *    what may be happening is that `IValue` doesn't store internally
 *    your type `T`. Instead, it constructs a type new `T` everytime
 *    you try to get `T` for it (see `IListRef<at::OptinalTensorRef>`).
 */

namespace c10 {
template <typename T>
class IListRef;

/*
 * Applies arbitrary macros to each `IListRefTag`.
 */
#define TORCH_ILISTREF_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)                \
  _(Boxed, ##__VA_ARGS__)                  \
  _(Materialized, ##__VA_ARGS__)

/*
 * Defines a "switch-case" for `TAG`. Inside, it executes `BODY`,
 * while bringing to scope:
 *
 * - `ImplT`: the implementation class for `TAG`
 * - `this_`: the result of unwrapping `this`
 */
#define TORCH_ILISTREF_UNWRAP_CASE(TAG, BODY)                        \
  case c10::IListRefTag::TAG: {                                      \
    using ImplT = c10::detail::IListRefTagImpl<IListRefTag::TAG, T>; \
    auto& this_ = ImplT::unwrap(*this);                              \
    BODY                                                             \
  } break;

/*
 * Dispatches the unwrap call, depending on `TAG`, followed by
 * the execution of `BODY`. It aborts if `TAG` is not a `IListRefTag`.
 *
 * This macro is useful because it allows us to handle different
 * types (that correspond to different tags) to be implemented
 * only once. We can do it even when the implementation of the
 * different tags aren't syntatically the same, by dispatching
 * it to a function (e.g. `ImplT::<dispatch-function>(this_)`).
 */
#define TORCH_ILISTREF_UNWRAP(TAG, BODY)                         \
  switch (TAG) {                                                 \
    TORCH_ILISTREF_FORALL_TAGS(TORCH_ILISTREF_UNWRAP_CASE, BODY) \
    break;                                                       \
    default:                                                     \
      TORCH_INTERNAL_ASSERT(false, "invalid IListRef tag.");     \
  }

enum class IListRefTag {
#define DEFINE_TAG(tag, ...) tag,
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_TAG)
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
using IListRefConstRef = typename ivalue_to_const_ref_overload_return<T>::type;

/*
 * Interface that implements key functions for each `IListRefTag` type.
 *
 * What is this for?
 * =================
 * Given an `IListRef(Iterator)<T>`, some methods have to be implemented
 * differently for each `TAG`. Therefore, the methods inside this class
 * are used as dispatch targets for the different `IListRefTag` values.
 *
 * You should create an specialization of this class for each possible
 * combination of `IListRefTag` type (except `None`) and element types
 * (e.g. `Tensor`).
 *
 * What does it do?
 * ================
 * 1. defines static methods to be used as dispatch targets by both
 *    `IListRef<T>` and `IListRefIterator<T>` (see the implementation of
 *    `IListRefTagImplBase`).
 *
 * 2. defines the `elem_type` and `list_type` aliases that will be
 *    used in the definition of `IListRef<T>`. In general, we should do
 *    so by inheriting from `IListRefTagImplBase<TAG, T, ListElemT>`.
 *
 * [Note: IListRefTagImpl Specialization]
 * ======================================
 * For `IListRef(Iterator)<at::Tensor>`:
 * - <IListRefTag::Unboxed, at::Tensor>
 * - <IListRefTag::Boxed, at::Tensor>
 * - <IListRefTag::Materialized, at::Tensor>
 *
 * For `IListRef(Iterator)<at::OptionalTensorRef>`:
 * - <IListRefTag::Unboxed, at::OptionalTensorRef>
 * - <IListRefTag::Boxed, at::OptionalTensorRef>
 * - <IListRefTag::Materialized, at::OptionalTensorRef>
 */
template <IListRefTag TAG, typename T>
class IListRefTagImpl {};

/*
 * Base implementation of `IListRefTagImpl<TAG, T>` methods.
 *
 * What is this for?
 * =================
 * This should make adding specializations for new types easier. For
 * example, one should be able to add a new type just by making its
 * `IListRefTagImpl` specialization inherit from `IListRefTagImplBase`.
 *
 * You should create a partial specialization for this class only if
 * you introduce a new `IListRefTag`. The idea being that there is one
 * default implementation for each possible value of `IListRefTag`.
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
 *    are supposed to be defined on `IListRefTagImpl` specializations.
 *
 * 4. inheriting from `IListRefTagImplBase<TAG, T, ListElemT>` also means
 *    that the payload of the type `IListRef<T>` will be of type `list_type`
 *    when it is tagged as `TAG`.
 */
template <IListRefTag TAG, typename T, typename ListElemT = T>
class IListRefTagImplBase {};

/*
 * Materialized container for `IListRef<T>`.
 *
 * What is this for?
 * =================
 * Container that groups `T` references together. This exchanges the
 * overhead of every method call from `IListRef<T>` for a dynamic allocation.
 *
 * You should use this container instead of `IListRef<T>` if:
 *
 *   - You are going to iterate the list more than once
 *   - You need to repeatedly access arbitrary elements (using `operator[]`)
 * What does it do?

 * ================
 * Removes the reference (&) from the type, and wraps it into a
 * `std::reference_wrapper`. If `IListRefConstRef<T>` is not a
 * reference type, then it's left unchanged.
 */
template <typename T>
using _MaterializedIListRefElem = std::conditional_t<
    std::is_reference_v<T>,
    typename std::reference_wrapper<std::remove_reference_t<T>>,
    T>;

template <typename T>
using MaterializedIListRefElem = _MaterializedIListRefElem<IListRefConstRef<T>>;

template <typename T>
using MaterializedIListRef = std::vector<MaterializedIListRefElem<T>>;

} // namespace detail

/*
 * Iterator for `IListRef<T>`.
 *
 * What is it?
 * ===========
 * Currently, a `std::bidirectional_iterator` that wraps the iterator
 * types defined for each of the `IListRefTag`.
 *
 * One should be able to use it, as if it were the unwrapped
 * iterators themselves.

 * What does it do?
 * ================
 * Similarly to `IListRef<T>`, this is a wrapper class. Specifically, it
 * wraps each container's `const_iterator` type alias. So, for example,
 * given that the container for `IListRefTag::Boxed` is `c10::List`, this
 * iterator will wrap a `c10::List::const_iterator`.
 *
 * [Note: MSVC Iterator Debug]
 * ===========================
 * MSVC `vector<T>::iterator` implementation (used in the boxed variant)
 * makes it so this union's destructor, copy-constructor (assignment), and
 * move-constructor (assignment) are implicitly deleted.
 *
 * Therefore, we need to explicitly define them as needed. Follows a list
 * of places where these are needed and their reason:
 *
 *   - `Payload` destructor:
 *     it is deleted only if the macro `_ITERATOR_DEBUG_LEVEL` is set to 2.
 *
 *   - `IListRefIterator` destructor:
 *     same as above. However, we need to explicitly call the variant
 *     destructor explicitly.
 *
 *   - `IListRefIterator` copy-constructor:
 *     it is deleted only if the macro `_ITERATOR_DEBUG_LEVEL` is different
 *     than 0.
 */
template <typename T>
class IListRefIterator {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...)                        \
  friend class detail::IListRefTagImpl<IListRefTag::TAG, T>; \
  friend class detail::IListRefTagImplBase<                  \
      IListRefTag::TAG,                                      \
      T,                                                     \
      typename detail::IListRefTagImpl<IListRefTag::TAG, T>::elem_type>;
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

 public:
  // C++17 friendly std::iterator implementation
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  using unboxed_iterator_type = typename detail::
      IListRefTagImpl<IListRefTag::Unboxed, T>::list_type::const_iterator;
  using boxed_iterator_type = typename detail::
      IListRefTagImpl<IListRefTag::Boxed, T>::list_type::const_iterator;
  using materialized_iterator_type =
      typename detail::MaterializedIListRef<T>::const_iterator;

  IListRefIterator() : tag_(IListRefTag::None) {}

#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL != 0
  // See [Note: MSVC Iterator Debug]
  IListRefIterator(const IListRefIterator& iterator)
      : tag_(iterator.tag_) {
    switch (tag_) {
      case IListRefTag::Boxed:
        payload_.boxed_iterator = iterator.payload_.boxed_iterator;
        break;
      case IListRefTag::Unboxed:
        payload_.unboxed_iterator = iterator.payload_.unboxed_iterator;
        break;
      case IListRefTag::Materialized:
        payload_.materialized_iterator = iterator.payload_.materialized_iterator;
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "invalid IListRef tag.");
    }
  }
#endif

#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL == 2
  // See [Note: MSVC Iterator Debug]
  ~IListRefIterator() noexcept(false) {
    switch (tag_) {
      case IListRefTag::Boxed:
        payload_.boxed_iterator.~boxed_iterator_type();
        break;
      case IListRefTag::Unboxed:
        payload_.unboxed_iterator.~unboxed_iterator_type();
        break;
      case IListRefTag::Materialized:
        payload_.materialized_iterator.~materialized_iterator_type();
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "invalid IListRef tag.");
    }
  }
#endif

  IListRefIterator(boxed_iterator_type boxed) : tag_(IListRefTag::Boxed) {
    payload_.boxed_iterator = boxed;
  }

  IListRefIterator(unboxed_iterator_type unboxed) : tag_(IListRefTag::Unboxed) {
    payload_.unboxed_iterator = unboxed;
  }

  IListRefIterator(materialized_iterator_type materialized) : tag_(IListRefTag::Materialized) {
    payload_.materialized_iterator = materialized;
  }

  detail::IListRefConstRef<T> operator*() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return ImplT::iterator_get(this_); });
  }

  IListRefIterator& operator++() {
    TORCH_ILISTREF_UNWRAP(tag_, { ++this_; });
    return *this;
  }

  IListRefIterator operator++(int) {
    auto old = *this;
    TORCH_ILISTREF_UNWRAP(tag_, { ++this_; });
    return old;
  }

  IListRefIterator& operator--() {
    TORCH_ILISTREF_UNWRAP(tag_, { --this_; });
    return *this;
  }

  IListRefIterator operator--(int) {
    auto old = *this;
    TORCH_ILISTREF_UNWRAP(tag_, { --this_; });
    return old;
  }

  bool operator==(const IListRefIterator& rhs) const {
    if (tag_ != rhs.tag_) {
      return false;
    }
    TORCH_ILISTREF_UNWRAP(tag_, {
      auto& rhs_it = ImplT::unwrap(rhs);
      return this_ == rhs_it;
    });
  }

  bool operator!=(const IListRefIterator& rhs) const {
    return !(*this == rhs);
  }

 private:
  union Payload {
    boxed_iterator_type boxed_iterator;
    unboxed_iterator_type unboxed_iterator;
    materialized_iterator_type materialized_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}
#if defined(_MSC_VER)
    // See [Note: MSVC Iterator Debug]
    ~Payload() {}
#endif
  };

  Payload payload_;
  IListRefTag tag_;
};

/*
 * See [Note: IListRef]
 */
template <typename T>
class IListRef {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...)                        \
  friend class detail::IListRefTagImpl<IListRefTag::TAG, T>; \
  friend class detail::IListRefTagImplBase<                  \
      IListRefTag::TAG,                                      \
      T,                                                     \
      typename detail::IListRefTagImpl<IListRefTag::TAG, T>::elem_type>;
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

 public:
  using unboxed_type =
      typename detail::IListRefTagImpl<IListRefTag::Unboxed, T>::list_type;
  using boxed_type =
      typename detail::IListRefTagImpl<IListRefTag::Boxed, T>::list_type;
  using materialized_type =
      typename detail::MaterializedIListRef<T>;

  using iterator = IListRefIterator<T>;
  using const_iterator = IListRefIterator<T>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using value_type = typename iterator::value_type;

  IListRef() : tag_(IListRefTag::None) {}

  IListRef(const boxed_type& boxed) : tag_(IListRefTag::Boxed) {
    payload_.boxed = &boxed;
  }

  IListRef(const unboxed_type& unboxed) : tag_(IListRefTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  IListRef(const std::initializer_list<T>& list) : tag_(IListRefTag::Unboxed) {
    payload_.unboxed = at::ArrayRef<T>(list);
  }

  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible_v<unboxed_type, UnboxedConstructorArgs...>>>
  IListRef(UnboxedConstructorArgs&&... args) : tag_(IListRefTag::Unboxed) {
    payload_.unboxed = unboxed_type(std::forward<UnboxedConstructorArgs>(args)...);
  }

  IListRef(const materialized_type& materialized) : tag_(IListRefTag::Materialized) {
    payload_.materialized = &materialized;
  }

  size_t size() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return this_.size(); });
  }

  bool empty() const {
    return size() == 0;
  }

  iterator begin() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return this_.begin(); });
  }

  iterator end() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return this_.end(); });
  }

  detail::IListRefConstRef<T> front() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return ImplT::front(this_); });
  }

  /*
   * Materializes the `IListRef` into a `std::vector`.
   *
   * This should be used when one wishes to either:
   *
   *   - iterate over the list more than once: each `IListRefIterator`
   *     member function call has to go through a switch, introducing
   *     non-negligible overhead
   *
   *   - randomly access an arbitrary element using `operator[]`:
   *     same reason as above
   */
  detail::MaterializedIListRef<T> materialize() const {
    if (isMaterialized()) {
      return toMaterialized();
    }

    detail::MaterializedIListRef<T> materialized;
    materialized.reserve(size());
    for (const auto& t : *this) {
      materialized.emplace_back(t);
    }
    return materialized;
  }

#define DEFINE_CHECK(TAG, ...)    \
  bool is##TAG() const {          \
    return tag_ == IListRefTag::TAG; \
  }
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_CHECK)
#undef DEFINE_CHECK

  bool isNone() const {
    return tag_ == IListRefTag::None;
  }

#define DEFINE_CASTING(TAG, ...)                                          \
  const typename detail::IListRefTagImpl<IListRefTag::TAG, T>::list_type& \
      to##TAG() const {                                                   \
    TORCH_INTERNAL_ASSERT(is##TAG());                                     \
    return detail::IListRefTagImpl<IListRefTag::TAG, T>::unwrap(*this);   \
  }
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_CASTING)
#undef DEFINE_CASTING

 private:
  union Payload {
    const boxed_type* boxed;
    unboxed_type unboxed;
    const materialized_type* materialized;
    Payload() : boxed(nullptr) {}
  };

  Payload payload_;
  IListRefTag tag_;
};

} // namespace c10

#include <ATen/core/IListRef_inl.h>
