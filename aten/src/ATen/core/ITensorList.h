#pragma once

#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <initializer_list>
#include <iterator>
#include <type_traits>

namespace at {
class Tensor;
}

namespace c10 {
class ITensorList;
class ITensorListIterator;

// Applies arbitrary macros to each `ITensorListTag`.
#define TORCH_ITENSORLIST_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)                   \
  _(Boxed, ##__VA_ARGS__)

// Builds the name of the implementation class for `TAG`.
#define TORCH_ITENSORLIST_IMPL(TAG) \
  c10::detail::ITensorListTagImpl<c10::ITensorListTag::TAG>

// Defines a "switch-case" for `TAG`. Inside, it executes `BODY`,
// while bringing to scope:
//     - `ImplT`: the implementation class for `TAG`
//     - `this_`: the result of unwrapping `this`
#define TORCH_ITENSORLIST_UNWRAP_CASE(TAG, BODY) \
  case c10::ITensorListTag::TAG: {               \
    using ImplT = TORCH_ITENSORLIST_IMPL(TAG);   \
    auto& this_ = ImplT::unwrap(*this);          \
    BODY                                         \
  } break;

// Dispatches the unwrap call, depending on `TAG`, followed by
// the execution of `BODY`. It aborts if `TAG` is not a `ITensorListTag`.
#define TORCH_ITENSORLIST_UNWRAP(TAG, BODY)                            \
  switch (TAG) {                                                       \
    TORCH_ITENSORLIST_FORALL_TAGS(TORCH_ITENSORLIST_UNWRAP_CASE, BODY) \
    default:                                                           \
      TORCH_INTERNAL_ASSERT(false, "invalid ITensorList tag.");        \
  }

enum class ITensorListTag {
#define DEFINE_TAG(tag, ...) tag,
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
      None
};

namespace detail {
using ITensorListConstRef =
    typename detail::ivalue_to_const_ref_overload_return<at::Tensor>::type;

/*
 * Interface that implements key functions for each `ITensorListTag` type.
 *
 * You should create an specialization of this class for each
 * possible `ITensorListTag` type (except `None`).
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
template <ITensorListTag TAG>
class ITensorListTagImpl {};

template <>
class ITensorListTagImpl<ITensorListTag::Unboxed> {
 public:
  using list_type = at::ArrayRef<at::Tensor>;

  // Unwraps an `ITensorList` into a const-ref of type `list_type`.
  static const list_type& unwrap(const ITensorList& ilist);

  // Unwraps an `ITensorListIterator` into a (const) ref of type
  // `list_type::const_iterator`. Has overload for const.
  static list_type::const_iterator& unwrap(ITensorListIterator& it);
  static const list_type::const_iterator& unwrap(const ITensorListIterator& it);

  // Accesses the element referenced by the unwrapped iterator `it`.
  static ITensorListConstRef iterator_get(const list_type::const_iterator& it);
};

template <>
class ITensorListTagImpl<ITensorListTag::Boxed> {
 public:
  using list_type = List<at::Tensor>;
  static const list_type& unwrap(const ITensorList& ilist);
  static list_type::const_iterator& unwrap(ITensorListIterator& it);
  static const list_type::const_iterator& unwrap(const ITensorListIterator& it);
  static ITensorListConstRef iterator_get(const list_type::const_iterator& it);
};
} // namespace detail

/*
 * Wrapper around both boxed and unboxed iterators.
 *
 * Currently, a `std::bidirectional_iterator` that wraps those
 * defined for each of the `ITensorListTag`.
 *
 * One should be able to use it, as if it were the unwrapped
 * iterators themselves.
 */
class ITensorListIterator
    : public std::iterator<std::bidirectional_iterator_tag, at::Tensor> {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ITENSORLIST_IMPL(TAG);
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

  using unboxed_iterator_type =
      TORCH_ITENSORLIST_IMPL(Unboxed)::list_type::const_iterator;
  using boxed_iterator_type =
      TORCH_ITENSORLIST_IMPL(Boxed)::list_type::const_iterator;

  union Payload {
    boxed_iterator_type boxed_iterator;
    unboxed_iterator_type unboxed_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}
    ~Payload() = default;
  };

 public:
  ITensorListIterator() : tag_(ITensorListTag::None) {}

  ITensorListIterator(boxed_iterator_type boxed) : tag_(ITensorListTag::Boxed) {
    payload_.boxed_iterator = boxed;
  }

  ITensorListIterator(unboxed_iterator_type unboxed)
      : tag_(ITensorListTag::Unboxed) {
    payload_.unboxed_iterator = unboxed;
  }

  detail::ITensorListConstRef operator*() const {
    TORCH_ITENSORLIST_UNWRAP(tag_, { return ImplT::iterator_get(this_); });
  }

  ITensorListIterator& operator++() {
    TORCH_ITENSORLIST_UNWRAP(tag_, { ++this_; });
    return *this;
  }

  ITensorListIterator operator++(int) {
    auto old = *this;
    TORCH_ITENSORLIST_UNWRAP(tag_, { ++this_; });
    return old;
  }

  ITensorListIterator& operator--() {
    TORCH_ITENSORLIST_UNWRAP(tag_, { --this_; });
    return *this;
  }

  ITensorListIterator operator--(int) {
    auto old = *this;
    TORCH_ITENSORLIST_UNWRAP(tag_, { --this_; });
    return old;
  }

  bool operator==(const ITensorListIterator& rhs) const {
    if (tag_ != rhs.tag_) {
      return false;
    }
    TORCH_ITENSORLIST_UNWRAP(tag_, {
      auto& rhs_it = ImplT::unwrap(rhs);
      return this_ == rhs_it;
    });
  }

  bool operator!=(const ITensorListIterator& rhs) const {
    return !(*this == rhs);
  }

 private:
  Payload payload_;
  ITensorListTag tag_;
};

/*
 * [Note: ITensorList]
 * Wrapper around boxed and unboxed API containers.
 *
 * Tagged union of both API containers:
 *     - `TensorList`, a.k.a. `ArrayRef<Tensor>` (the unboxed API container)
 *     - `List<Tensor>` (the boxed API container)
 *
 * This container wraps around these two, without incurring in extra overhead
 * for converting from one to another.
 *
 * Note that `ITensorList` is a view type. Meaning that it won't own the
 * tensors it holds. If you need it to last longer, make sure that there is
 * actually a non-temporary list of tensors (e.g. `vector<Tensor>`) that owns
 * them and outlives the `ITensorList` instance.
 *
 * (see https://github.com/pytorch/pytorch/issues/66328)
 */
class ITensorList {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ITENSORLIST_IMPL(TAG);
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

  using unboxed_type = TORCH_ITENSORLIST_IMPL(Unboxed)::list_type;
  using boxed_type = TORCH_ITENSORLIST_IMPL(Boxed)::list_type;

  union Payload {
    const boxed_type* boxed;
    unboxed_type unboxed;
    Payload() : boxed(nullptr) {}
    ~Payload() = default;
  };

 public:
  using iterator = ITensorListIterator;
  using const_iterator = ITensorListIterator;
  using value_type = typename iterator::value_type;

  ITensorList() : tag_(ITensorListTag::None) {}

  ITensorList(const std::initializer_list<at::Tensor>& list)
      : tag_(ITensorListTag::Unboxed) {
    payload_.unboxed = at::ArrayRef<at::Tensor>(list);
  }

  ITensorList(const boxed_type& boxed) : tag_(ITensorListTag::Boxed) {
    payload_.boxed = &boxed;
  }

  ITensorList(const unboxed_type& unboxed) : tag_(ITensorListTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible<unboxed_type, UnboxedConstructorArgs...>::value>>
  ITensorList(UnboxedConstructorArgs&&... args)
      : tag_(ITensorListTag::Unboxed) {
    payload_.unboxed = unboxed_type(std::forward<UnboxedConstructorArgs>(args)...);
  }

  size_t size() const {
    TORCH_ITENSORLIST_UNWRAP(tag_, { return this_.size(); });
  }

  bool empty() const {
    return size() == 0;
  }

  iterator begin() const {
    TORCH_ITENSORLIST_UNWRAP(tag_, { return this_.begin(); });
  }

  iterator end() const {
    TORCH_ITENSORLIST_UNWRAP(tag_, { return this_.end(); });
  }

  detail::ITensorListConstRef operator[](size_t i) const {
    TORCH_ITENSORLIST_UNWRAP(tag_, { return this_[i]; });
  }

#define DEFINE_CHECK(TAG, ...)          \
  bool is##TAG() const {                \
    return tag_ == ITensorListTag::TAG; \
  }
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_CHECK);
#undef DEFINE_CHECK

  bool isNone() const {
    return tag_ == ITensorListTag::None;
  }

#define DEFINE_CASTING(TAG, ...)                                        \
  const typename TORCH_ITENSORLIST_IMPL(TAG)::list_type& to##TAG() const { \
    TORCH_INTERNAL_ASSERT(is##TAG());                                   \
    return TORCH_ITENSORLIST_IMPL(TAG)::unwrap(*this);                  \
  }
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_CASTING);
#undef DEFINE_CASTING

 private:
  Payload payload_;
  ITensorListTag tag_;
};
} // namespace c10

inline
const TORCH_ITENSORLIST_IMPL(Unboxed)::list_type&
TORCH_ITENSORLIST_IMPL(Unboxed)::unwrap(
    const c10::ITensorList& ilist
) {
  return ilist.payload_.unboxed;
}

inline
TORCH_ITENSORLIST_IMPL(Unboxed)::list_type::const_iterator&
TORCH_ITENSORLIST_IMPL(Unboxed)::unwrap(
    c10::ITensorListIterator& it
) {
  return it.payload_.unboxed_iterator;
}

inline
const TORCH_ITENSORLIST_IMPL(Unboxed)::list_type::const_iterator&
TORCH_ITENSORLIST_IMPL(Unboxed)::unwrap(
    const c10::ITensorListIterator& it
) {
  return it.payload_.unboxed_iterator;
}

inline
c10::detail::ITensorListConstRef
TORCH_ITENSORLIST_IMPL(Unboxed)::iterator_get(
    const list_type::const_iterator& it
) {
  return *it;
}

inline
const TORCH_ITENSORLIST_IMPL(Boxed)::list_type&
TORCH_ITENSORLIST_IMPL(Boxed)::unwrap(
    const c10::ITensorList& ilist
) {
  return *ilist.payload_.boxed;
}

inline
TORCH_ITENSORLIST_IMPL(Boxed)::list_type::const_iterator&
TORCH_ITENSORLIST_IMPL(Boxed)::unwrap(
    c10::ITensorListIterator& it
) {
  return it.payload_.boxed_iterator;
}

inline
const TORCH_ITENSORLIST_IMPL(Boxed)::list_type::const_iterator&
TORCH_ITENSORLIST_IMPL(Boxed)::unwrap(
    const c10::ITensorListIterator& it
) {
  return it.payload_.boxed_iterator;
}

inline
c10::detail::ITensorListConstRef
TORCH_ITENSORLIST_IMPL(Boxed)::iterator_get(
    const list_type::const_iterator& it
) {
  return (*it).get().toTensor();
}
