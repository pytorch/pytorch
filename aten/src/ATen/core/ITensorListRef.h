#pragma once

#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>

namespace at {
class Tensor;
}

namespace c10 {
class ITensorListRef;
class ITensorListRefIterator;

// Applies arbitrary macros to each `ITensorListRefTag`.
#define TORCH_ITENSORLISTREF_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)                      \
  _(Boxed, ##__VA_ARGS__)

// Builds the name of the implementation class for `TAG`.
#define TORCH_ITENSORLISTREF_IMPL(TAG) \
  c10::detail::ITensorListRefTagImpl<c10::ITensorListRefTag::TAG>

// Defines a "switch-case" for `TAG`. Inside, it executes `BODY`,
// while bringing to scope:
//     - `ImplT`: the implementation class for `TAG`
//     - `this_`: the result of unwrapping `this`
#define TORCH_ITENSORLISTREF_UNWRAP_CASE(TAG, BODY) \
  case c10::ITensorListRefTag::TAG: {               \
    using ImplT = TORCH_ITENSORLISTREF_IMPL(TAG);   \
    auto& this_ = ImplT::unwrap(*this);             \
    BODY                                            \
  } break;

// Dispatches the unwrap call, depending on `TAG`, followed by
// the execution of `BODY`. It aborts if `TAG` is not a `ITensorListRefTag`.
#define TORCH_ITENSORLISTREF_UNWRAP(TAG, BODY)                               \
  switch (TAG) {                                                             \
    TORCH_ITENSORLISTREF_FORALL_TAGS(TORCH_ITENSORLISTREF_UNWRAP_CASE, BODY) \
    default:                                                                 \
      TORCH_INTERNAL_ASSERT(false, "invalid ITensorListRef tag.");           \
  }

enum class ITensorListRefTag {
#define DEFINE_TAG(tag, ...) tag,
  TORCH_ITENSORLISTREF_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
      None
};

namespace detail {
using ITensorListRefConstRef =
    typename detail::ivalue_to_const_ref_overload_return<at::Tensor>::type;

/*
 * Interface that implements key functions for each `ITensorListRefTag` type.
 *
 * You should create an specialization of this class for each
 * possible `ITensorListRefTag` type (except `None`).
 *
 * Specializations of this class should, at least, define:
 *     - a type `list_type`
 *     - 1 function `unwrap` for getting the actual `list_type`
 *     - 2 functions `unwrap` (const and non-const overloads) for getting
 *       iterators of `list_type`
 *     - a function `iterator_get`
 *
 * See the examples below.
 */
template <ITensorListRefTag TAG>
class ITensorListRefTagImpl {};

template <>
class ITensorListRefTagImpl<ITensorListRefTag::Unboxed> {
 public:
  using list_type = at::ArrayRef<at::Tensor>;

  // Unwraps an `ITensorListRef` into a const-ref of type `list_type`.
  static const list_type& unwrap(const ITensorListRef& ilist);

  // Unwraps an `ITensorListRefIterator` into a (const) ref of type
  // `list_type::const_iterator`. Has overload for const.
  static list_type::const_iterator& unwrap(ITensorListRefIterator& it);
  static const list_type::const_iterator& unwrap(const ITensorListRefIterator& it);

  // Accesses the element referenced by the unwrapped iterator `it`.
  static ITensorListRefConstRef iterator_get(const list_type::const_iterator& it);
};

template <>
class ITensorListRefTagImpl<ITensorListRefTag::Boxed> {
 public:
  using list_type = List<at::Tensor>;
  static const list_type& unwrap(const ITensorListRef& ilist);
  static list_type::const_iterator& unwrap(ITensorListRefIterator& it);
  static const list_type::const_iterator& unwrap(const ITensorListRefIterator& it);
  static ITensorListRefConstRef iterator_get(const list_type::const_iterator& it);
};
} // namespace detail

/*
 * Materialized list for `ITensorListRef`.
 *
 * Container that groups `Tensor` references together. This exchanges the
 * overhead of every method call from `ITensorListRef` for a dynamic allocation.
 *
 * You should use this container instead of `ITensorListRef` if:
 *
 *   - You are going to iterate the list of tensors more than once
 *   - You need to repeatedly access arbitrary elements (using `operator[]`)
 */
using MaterializedITensorListRef =
    std::vector<std::reference_wrapper<const at::Tensor>>;

/*
 * Wrapper around both boxed and unboxed iterators.
 *
 * Currently, a `std::bidirectional_iterator` that wraps those defined for
 * each of the `ITensorListRefTag`.
 *
 * One should be able to use it, as if it were the unwrapped iterators
 * themselves.
 *
 * [Note: MSVC Iterator Debug]
 * ===========================
 * MSVC `vector<T>::iterator` implementation (used in the boxed variant)
 * makes it so this union's destructor, copy-constructor (assignment), and
 * move-constructor (assignment) are implcitly deleted.
 *
 * Therefore, we need to explicitly define them as needed. Follows a list
 * of places where these are needed and their reason:
 *
 *   - `Payload` destructor:
 *     it is deleted only if the macro `_ITERATOR_DEBUG_LEVEL` is set to 2.
 *
 *   - `ITensorListRefIterator` destructor:
 *     same as above. However, we need to explicitly call the variant
 *     destructor explicitly.
 *
 *   - `ITensorListRefIterator` copy-constructor:
 *     it is deleted only if the macro `_ITERATOR_DEBUG_LEVEL` is different
 *     than 0.
 */
class ITensorListRefIterator
    : public std::iterator<
          std::bidirectional_iterator_tag,
          detail::ITensorListRefConstRef,
          ptrdiff_t,
          std::add_pointer<detail::ITensorListRefConstRef>,
          std::add_rvalue_reference<detail::ITensorListRefConstRef>> {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ITENSORLISTREF_IMPL(TAG);
  TORCH_ITENSORLISTREF_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

  using unboxed_iterator_type =
      TORCH_ITENSORLISTREF_IMPL(Unboxed)::list_type::const_iterator;
  using boxed_iterator_type =
      TORCH_ITENSORLISTREF_IMPL(Boxed)::list_type::const_iterator;

  union Payload {
    boxed_iterator_type boxed_iterator;
    unboxed_iterator_type unboxed_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}
#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL == 2
    // See [Note: MSVC Iterator Debug]
    ~Payload() {}
#endif
  };

 public:
  ITensorListRefIterator() : tag_(ITensorListRefTag::None) {}

#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL != 0
  // See [Note: MSVC Iterator Debug]
  ITensorListRefIterator(const ITensorListRefIterator& iterator)
      : tag_(iterator.tag_) {
    switch (tag_) {
      case ITensorListRefTag::Boxed:
        payload_.boxed_iterator = iterator.payload_.boxed_iterator;
      case ITensorListRefTag::Unboxed:
        payload_.unboxed_iterator = iterator.payload_.unboxed_iterator;
      default:
        TORCH_INTERNAL_ASSERT(false, "invalid ITensorListRef tag.");
    }
  }
#endif

#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL == 2
  // See [Note: MSVC Iterator Debug]
  ~ITensorListRefIterator() {
    switch (tag_) {
      case ITensorListRefTag::Boxed:
        payload_.boxed_iterator.~boxed_iterator_type();
      case ITensorListRefTag::Unboxed:
        payload_.unboxed_iterator.~unboxed_iterator_type();
      default:
        TORCH_INTERNAL_ASSERT(false, "invalid ITensorListRef tag.");
    }
  }
#endif

  ITensorListRefIterator(boxed_iterator_type boxed) : tag_(ITensorListRefTag::Boxed) {
    payload_.boxed_iterator = boxed;
  }

  ITensorListRefIterator(unboxed_iterator_type unboxed)
      : tag_(ITensorListRefTag::Unboxed) {
    payload_.unboxed_iterator = unboxed;
  }

  detail::ITensorListRefConstRef operator*() const {
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { return ImplT::iterator_get(this_); });
  }

  ITensorListRefIterator& operator++() {
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { ++this_; });
    return *this;
  }

  ITensorListRefIterator operator++(int) {
    auto old = *this;
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { ++this_; });
    return old;
  }

  ITensorListRefIterator& operator--() {
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { --this_; });
    return *this;
  }

  ITensorListRefIterator operator--(int) {
    auto old = *this;
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { --this_; });
    return old;
  }

  bool operator==(const ITensorListRefIterator& rhs) const {
    if (tag_ != rhs.tag_) {
      return false;
    }
    TORCH_ITENSORLISTREF_UNWRAP(tag_, {
      auto& rhs_it = ImplT::unwrap(rhs);
      return this_ == rhs_it;
    });
  }

  bool operator!=(const ITensorListRefIterator& rhs) const {
    return !(*this == rhs);
  }

 private:
  Payload payload_;
  ITensorListRefTag tag_;
};

/*
 * [Note: ITensorListRef]
 * Wrapper around boxed and unboxed API containers.
 *
 * Tagged union of both API containers:
 *     - `TensorList`, a.k.a. `ArrayRef<Tensor>` (the unboxed API container)
 *     - `List<Tensor>` (the boxed API container)
 *
 * This container wraps around these two, without incurring in extra overhead
 * for converting from one to another.
 *
 * Note that `ITensorListRef` is a view type. Meaning that it won't own the
 * tensors it holds. If you need it to last longer, make sure that there is
 * actually a non-temporary list of tensors (e.g. `vector<Tensor>`) that owns
 * them and outlives the `ITensorListRef` instance.
 *
 * (see https://github.com/pytorch/pytorch/issues/66328)
 */
class ITensorListRef {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ITENSORLISTREF_IMPL(TAG);
  TORCH_ITENSORLISTREF_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

  using unboxed_type = TORCH_ITENSORLISTREF_IMPL(Unboxed)::list_type;
  using boxed_type = TORCH_ITENSORLISTREF_IMPL(Boxed)::list_type;

  union Payload {
    const boxed_type* boxed;
    unboxed_type unboxed;
    Payload() : boxed(nullptr) {}
    ~Payload() {};
  };

 public:
  using iterator = ITensorListRefIterator;
  using const_iterator = ITensorListRefIterator;
  using value_type = typename iterator::value_type;

  ITensorListRef() : tag_(ITensorListRefTag::None) {}

  ITensorListRef(const std::initializer_list<at::Tensor>& list)
      : tag_(ITensorListRefTag::Unboxed) {
    payload_.unboxed = at::ArrayRef<at::Tensor>(list);
  }

  ITensorListRef(const boxed_type& boxed) : tag_(ITensorListRefTag::Boxed) {
    payload_.boxed = &boxed;
  }

  ITensorListRef(const unboxed_type& unboxed) : tag_(ITensorListRefTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible<unboxed_type, UnboxedConstructorArgs...>::value>>
  ITensorListRef(UnboxedConstructorArgs&&... args)
      : tag_(ITensorListRefTag::Unboxed) {
    payload_.unboxed = unboxed_type(std::forward<UnboxedConstructorArgs>(args)...);
  }

  size_t size() const {
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { return this_.size(); });
  }

  bool empty() const {
    return size() == 0;
  }

  iterator begin() const {
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { return this_.begin(); });
  }

  iterator end() const {
    TORCH_ITENSORLISTREF_UNWRAP(tag_, { return this_.end(); });
  }

  MaterializedITensorListRef materialize() const {
    MaterializedITensorListRef materialized;
    materialized.reserve(size());
    for (const auto& t : *this) {
      materialized.emplace_back(t);
    }
    return materialized;
  }

#define DEFINE_CHECK(TAG, ...)             \
  bool is##TAG() const {                   \
    return tag_ == ITensorListRefTag::TAG; \
  }
  TORCH_ITENSORLISTREF_FORALL_TAGS(DEFINE_CHECK);
#undef DEFINE_CHECK

  bool isNone() const {
    return tag_ == ITensorListRefTag::None;
  }

#define DEFINE_CASTING(TAG, ...)                                              \
  const typename TORCH_ITENSORLISTREF_IMPL(TAG)::list_type& to##TAG() const { \
    TORCH_INTERNAL_ASSERT(is##TAG());                                         \
    return TORCH_ITENSORLISTREF_IMPL(TAG)::unwrap(*this);                     \
  }
  TORCH_ITENSORLISTREF_FORALL_TAGS(DEFINE_CASTING);
#undef DEFINE_CASTING

 private:
  Payload payload_;
  ITensorListRefTag tag_;
};

} // namespace c10

inline
const TORCH_ITENSORLISTREF_IMPL(Unboxed)::list_type&
TORCH_ITENSORLISTREF_IMPL(Unboxed)::unwrap(
    const c10::ITensorListRef& ilist
) {
  return ilist.payload_.unboxed;
}

inline
TORCH_ITENSORLISTREF_IMPL(Unboxed)::list_type::const_iterator&
TORCH_ITENSORLISTREF_IMPL(Unboxed)::unwrap(
    c10::ITensorListRefIterator& it
) {
  return it.payload_.unboxed_iterator;
}

inline
const TORCH_ITENSORLISTREF_IMPL(Unboxed)::list_type::const_iterator&
TORCH_ITENSORLISTREF_IMPL(Unboxed)::unwrap(
    const c10::ITensorListRefIterator& it
) {
  return it.payload_.unboxed_iterator;
}

inline
c10::detail::ITensorListRefConstRef
TORCH_ITENSORLISTREF_IMPL(Unboxed)::iterator_get(
    const list_type::const_iterator& it
) {
  return *it;
}

inline
const TORCH_ITENSORLISTREF_IMPL(Boxed)::list_type&
TORCH_ITENSORLISTREF_IMPL(Boxed)::unwrap(
    const c10::ITensorListRef& ilist
) {
  return *ilist.payload_.boxed;
}

inline
TORCH_ITENSORLISTREF_IMPL(Boxed)::list_type::const_iterator&
TORCH_ITENSORLISTREF_IMPL(Boxed)::unwrap(
    c10::ITensorListRefIterator& it
) {
  return it.payload_.boxed_iterator;
}

inline
const TORCH_ITENSORLISTREF_IMPL(Boxed)::list_type::const_iterator&
TORCH_ITENSORLISTREF_IMPL(Boxed)::unwrap(
    const c10::ITensorListRefIterator& it
) {
  return it.payload_.boxed_iterator;
}

inline
c10::detail::ITensorListRefConstRef
TORCH_ITENSORLISTREF_IMPL(Boxed)::iterator_get(
    const list_type::const_iterator& it
) {
  return (*it).get().toTensor();
}

namespace at {
using ITensorListRef = c10::ITensorListRef;
using ITensorListRefIterator = c10::ITensorListRefIterator;
using MaterializedITensorListRef = c10::MaterializedITensorListRef;
} // namespace at
