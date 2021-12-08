#pragma once

#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <initializer_list>
#include <iterator>
#include <type_traits>

namespace c10 {
// Applies arbitrary macros to each `IListTag`.
#define TORCH_ILIST_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)             \
  _(Boxed, ##__VA_ARGS__)

// Builds the name of the implementation class for `TAG`.
#define TORCH_ILIST_IMPL(T, TAG) \
  c10::detail::IListTagImpl<T, c10::IListTag::TAG>

// Defines a "switch-case" for `TAG`. Inside, it executes `BODY`,
// while bringing to scope:
//     - `ImplT`: the implementation class for `TAG`
//     - `this_`: the result of unwrapping `this`
#define TORCH_ILIST_UNWRAP_CASE(TAG, BODY)  \
  case c10::IListTag::TAG: {                \
    using ImplT = TORCH_ILIST_IMPL(T, TAG); \
    auto& this_ = ImplT::unwrap(*this);     \
    BODY                                    \
  } break;

// Dispatches the unwrap call, depending on `TAG`, followed by
// the execution of `BODY`. It aborts if `TAG` is not a `IListTag`.
#define TORCH_ILIST_UNWRAP(TAG, BODY)                      \
  switch (TAG) {                                           \
    TORCH_ILIST_FORALL_TAGS(TORCH_ILIST_UNWRAP_CASE, BODY) \
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
template <typename T>
using IListConstRef = typename ivalue_to_const_ref_overload_return<T>::type;

/*
 * Interface that implements key functions for each `IListTag` type.
 *
 * You should create an specialization of this class for each
 * possible combination of `IListTag` type (except `None`) and
 * element types (e.g. `Tensor`).
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
template <typename T, IListTag TAG>
class IListTagImpl {};

} // namespace detail

/*
 * Wrapper around both boxed and unboxed iterators.
 *
 * Currently, a `std::bidirectional_iterator` that wraps those
 * defined for each of the `IListTag`.
 *
 * One should be able to use it, as if it were the unwrapped
 * iterators themselves.
 */
template <typename T>
class IListIterator : public std::iterator<std::bidirectional_iterator_tag, T> {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ILIST_IMPL(T, TAG);
  TORCH_ILIST_FORALL_TAGS(DEFINE_FRIEND_CLASS);
#undef DEFINE_FRIEND_CLASS

  using unboxed_iterator_type =
      typename TORCH_ILIST_IMPL(T, Unboxed)::list_type::const_iterator;
  using boxed_iterator_type =
      typename TORCH_ILIST_IMPL(T, Boxed)::list_type::const_iterator;

  union Payload {
    boxed_iterator_type boxed_iterator;
    unboxed_iterator_type unboxed_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}
    ~Payload() = default;
  };

 public:
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
  Payload payload_;
  IListTag tag_;
};

/*
 * Wrapper around boxed and unboxed API containers.
 *
 * Tagged union of both boxed and unboxed API containers.
 * This container wraps around these two, without incurring in extra overhead
 * for converting from one to another.
 *
 * (see https://github.com/pytorch/pytorch/issues/66328)
 */
template <typename T>
class IList {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...) friend class TORCH_ILIST_IMPL(T, TAG);
  TORCH_ILIST_FORALL_TAGS(DEFINE_FRIEND_CLASS);
#undef DEFINE_FRIEND_CLASS

  using unboxed_type = typename TORCH_ILIST_IMPL(T, Unboxed)::list_type;
  using boxed_type = typename TORCH_ILIST_IMPL(T, Boxed)::list_type;

  union Payload {
    const boxed_type* boxed;
    unboxed_type unboxed;
    Payload() : boxed(nullptr) {}
    ~Payload() = default;
  };

 public:
  using iterator = IListIterator<T>;
  using const_iterator = IListIterator<T>;
  using value_type = typename iterator::value_type;

  IList() : tag_(IListTag::None) {}

  IList(const std::initializer_list<T>& list)
      : tag_(IListTag::Unboxed) {
    payload_.unboxed = at::ArrayRef<T>(list);
  }

  IList(const boxed_type& boxed) : tag_(IListTag::Boxed) {
    payload_.boxed = &boxed;
  }

  IList(const unboxed_type& unboxed) : tag_(IListTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible<unboxed_type, UnboxedConstructorArgs...>::value>>
  IList(UnboxedConstructorArgs&&... args) : tag_(IListTag::Unboxed) {
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

  detail::IListConstRef<T> operator[](size_t i) const {
    TORCH_ILIST_UNWRAP(tag_, { return this_[i]; });
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

#define DEFINE_CASTING(TAG, ...)                                     \
  const typename TORCH_ILIST_IMPL(T, TAG)::list_type& to##TAG() const { \
    TORCH_INTERNAL_ASSERT(is##TAG());                                \
    return TORCH_ILIST_IMPL(T, TAG)::unwrap(*this);                  \
  }
  TORCH_ILIST_FORALL_TAGS(DEFINE_CASTING);
#undef DEFINE_CASTING

 private:
  Payload payload_;
  IListTag tag_;
};

} // namespace c10

#include <ATen/core/IList_inl.h>
