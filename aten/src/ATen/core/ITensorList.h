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

/*
 * Adding a new tag
 * ================

 * -> Short Version
 *
 * If you are lucky, the only necessary things for introducing
 * a new tag are:
 *
 *     - Add a line to `TORCH_ITENSORLIST_FORALL_TAGS` such as:
 *                 `_(TagName, ##__VA_ARGS__)`
 *
 *     - Specialize `ITensorListTagImpl` to your tag, implementing
 *       the required methods
 *
 *     - Implement constructors for the type corresponding to your
 *       tag on both `ITensorList(Iterator)` classes
 *
 * -> Long Version
 *
 * That may not work if your container is "more different" than
 * the ones implemented. Specifically, maybe, some of the functions
 * implemented with `TORCH_ITENSORLIST_UNWRAP` may not work for it.
 * In those cases, you should do something like `get` and
 * `iterator_get` functions.
 *
 * Say the implementation for `begin` is different. Then, you should:
 *
 *     - Do everything done in the "Short Version" for introducing
 *       the new tag
 *
 *     - Implement a new function `begin` in all implementation
 *       classes (for all tag types)
 *
 *     - Call it directly inside a `TORCH_ITENSORLIST_UNWRAP`
 */

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
 *     - a type `self_t`
 *     - 2 functions `unwrap` (const and non-const overloads)
 *     - a function `iterator_get`
 *
 * See the examples below.
 */
template <ITensorListTag TAG>
class ITensorListTagImpl {};

template <>
class ITensorListTagImpl<ITensorListTag::Unboxed> {
 public:
  using self_t = at::ArrayRef<at::Tensor>;
  // Unwraps an `ITensorList` into a const-ref of type `self_t`.
  static const self_t& unwrap(const ITensorList& ilist);
  // Unwraps an `ITensorListIterator` into a (const) ref of type
  // `self_t::const_iterator`. Has overload for const.
  static self_t::const_iterator& unwrap(ITensorListIterator& it);
  static const self_t::const_iterator& unwrap(const ITensorListIterator& it);
  // Accesses the element referenced by the unwrapped iterator `it`.
  static ITensorListConstRef iterator_get(const self_t::const_iterator& it);
};

template <>
class ITensorListTagImpl<ITensorListTag::Boxed> {
 public:
  using self_t = List<at::Tensor>;
  static const self_t& unwrap(const ITensorList& ilist);
  static self_t::const_iterator& unwrap(ITensorListIterator& it);
  static const self_t::const_iterator& unwrap(const ITensorListIterator& it);
  static ITensorListConstRef iterator_get(const self_t::const_iterator& it);
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
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_FRIEND_CLASS);
#undef DEFINE_FRIEND_CLASS

  using unboxed_iterator_t =
      TORCH_ITENSORLIST_IMPL(Unboxed)::self_t::const_iterator;
  using boxed_iterator_t =
      TORCH_ITENSORLIST_IMPL(Boxed)::self_t::const_iterator;

  union Payload {
    boxed_iterator_t boxed_iterator;
    unboxed_iterator_t unboxed_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}
    ~Payload() = default;
  };

 public:
  ITensorListIterator() : tag_(ITensorListTag::None) {}

  ITensorListIterator(boxed_iterator_t boxed) : tag_(ITensorListTag::Boxed) {
    payload_.boxed_iterator = boxed;
  }

  ITensorListIterator(unboxed_iterator_t unboxed)
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
 * Wrapper around boxed and unboxed API containers.
 *
 * Tagged union of both API containers:
 *     - `TensorList`, a.k.a. `ArrayRef<Tensor>` (the unboxed API container)
 *     - `List<Tensor>` (the boxed API container)
 *
 * This container wraps around these two, without incurring in extra overhead
 * for converting from one to another.
 *
 * (see https://github.com/pytorch/pytorch/issues/66328)
 */
class ITensorList {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ITENSORLIST_IMPL(TAG);
  TORCH_ITENSORLIST_FORALL_TAGS(DEFINE_FRIEND_CLASS);
#undef DEFINE_FRIEND_CLASS

  using unboxed_t = TORCH_ITENSORLIST_IMPL(Unboxed)::self_t;
  using boxed_t = TORCH_ITENSORLIST_IMPL(Boxed)::self_t;

  union Payload {
    const boxed_t* boxed;
    unboxed_t unboxed;
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

  ITensorList(const boxed_t& boxed) : tag_(ITensorListTag::Boxed) {
    payload_.boxed = &boxed;
  }

  ITensorList(const unboxed_t& unboxed) : tag_(ITensorListTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible<unboxed_t, UnboxedConstructorArgs...>::value>>
  ITensorList(UnboxedConstructorArgs&&... args)
      : tag_(ITensorListTag::Unboxed) {
    payload_.unboxed = unboxed_t(std::forward<UnboxedConstructorArgs>(args)...);
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
  const typename TORCH_ITENSORLIST_IMPL(TAG)::self_t& to##TAG() const { \
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

inline const TORCH_ITENSORLIST_IMPL(Unboxed)::self_t& TORCH_ITENSORLIST_IMPL(
    Unboxed)::unwrap(const c10::ITensorList& ilist) {
  return ilist.payload_.unboxed;
}

inline TORCH_ITENSORLIST_IMPL(Unboxed)::self_t::
    const_iterator& TORCH_ITENSORLIST_IMPL(Unboxed)::unwrap(
        c10::ITensorListIterator& it) {
  return it.payload_.unboxed_iterator;
}

inline const TORCH_ITENSORLIST_IMPL(Unboxed)::self_t::
    const_iterator& TORCH_ITENSORLIST_IMPL(Unboxed)::unwrap(
        const c10::ITensorListIterator& it) {
  return it.payload_.unboxed_iterator;
}

inline c10::detail::ITensorListConstRef TORCH_ITENSORLIST_IMPL(
    Unboxed)::iterator_get(const self_t::const_iterator& it) {
  return *it;
}

inline const TORCH_ITENSORLIST_IMPL(Boxed)::self_t& TORCH_ITENSORLIST_IMPL(
    Boxed)::unwrap(const c10::ITensorList& ilist) {
  return *ilist.payload_.boxed;
}

inline TORCH_ITENSORLIST_IMPL(Boxed)::self_t::
    const_iterator& TORCH_ITENSORLIST_IMPL(Boxed)::unwrap(
        c10::ITensorListIterator& it) {
  return it.payload_.boxed_iterator;
}

inline const TORCH_ITENSORLIST_IMPL(Boxed)::self_t::
    const_iterator& TORCH_ITENSORLIST_IMPL(Boxed)::unwrap(
        const c10::ITensorListIterator& it) {
  return it.payload_.boxed_iterator;
}

inline c10::detail::ITensorListConstRef TORCH_ITENSORLIST_IMPL(
    Boxed)::iterator_get(const self_t::const_iterator& it) {
  return (*it).get();
}
