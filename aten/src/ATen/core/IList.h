#pragma once

#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>

#include <initializer_list>
#include <iterator>
#include "c10/util/Exception.h"

namespace at {
class Tensor;
}

namespace c10 {
namespace details {
    template <typename T>
    class IList_Unboxed_Impl;
    template <typename T>
    class IList_Boxed_Impl;
}

#define TORCH_ILIST_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)             \
  _(Boxed, ##__VA_ARGS__)

#define TORCH_ILIST_IMPL(TAG, T) details::IList_##TAG##_Impl<T>

#define TORCH_ILIST_DISPATCH_CASE(TAG, FN, ...)                \
  case Tag::TAG:                                               \
    return TORCH_ILIST_IMPL(TAG, T)::FN(*this, ##__VA_ARGS__); \
    break;

#define TORCH_ILIST_DISPATCH_RETURN(FN, ...)                              \
  switch (tag_) {                                                         \
    TORCH_ILIST_FORALL_TAGS(TORCH_ILIST_DISPATCH_CASE, FN, ##__VA_ARGS__) \
    default:                                                              \
      TORCH_INTERNAL_ASSERT(false, "invalid IList tag.");                 \
  }

template <typename T>
class IList {
 public:
  using boxed_t = List<T>;
  using unboxed_t = ArrayRef<T>;

 private:
  union Payload {
    boxed_t* boxed;
    unboxed_t unboxed;
  };

  enum class Tag {
#define DEFINE_TAG(tag) tag,
    TORCH_ILIST_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
        None
  };

 public:
  class Iterator : public std::iterator<std::random_access_iterator_tag, T> {
   private:
    using boxed_iterator_t = typename boxed_t::iterator;
    using unboxed_iterator_t = typename unboxed_t::iterator;

    union {
      boxed_iterator_t boxed_iterator;
      unboxed_iterator_t unboxed_iterator;
    } impl_;

    Tag tag_;

   public:
    Iterator() : tag_(Tag::None) {}

    Iterator(boxed_iterator_t& boxed) : tag_(Tag::Boxed) {
      impl_.boxed = boxed;
    }

    Iterator(unboxed_iterator_t& unboxed) : tag_(Tag::Unboxed) {
      impl_.unboxed = unboxed;
    }

    T& operator*() const {
      TORCH_ILIST_DISPATCH_RETURN(iterator_get);
    }

    Iterator& operator++() {
      TORCH_ILIST_DISPATCH_RETURN(iterator_pre_inc);
    }

    Iterator operator++() const {
      TORCH_ILIST_DISPATCH_RETURN(iterator_post_inc);
    }

    Iterator& operator--() {
      TORCH_ILIST_DISPATCH_RETURN(iterator_pre_dec);
    }

    Iterator operator--() const {
      TORCH_ILIST_DISPATCH_RETURN(iterator_post_dec);
    }

    bool operator==(const Iterator& rhs) const {
      if (tag_ != rhs.tag_) {
        return false;
      }
      TORCH_ILIST_DISPATCH_RETURN(iterator_eq);
    }

    bool operator!=(const Iterator& rhs) const {
      return !(*this == rhs);
    }
  };

  using iterator = Iterator;
  using const_iterator = Iterator;
  using value_type = typename iterator::value_type;

  IList() : tag_(Tag::None) {}

  IList(const std::initializer_list<T>& list) : tag_(Tag::Unboxed) {
    payload_.unboxed = ArrayRef<T>(list);
  }

  IList(const unboxed_t& unboxed) : tag_(Tag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  IList(const boxed_t& boxed) : tag_(Tag::Boxed) {
    payload_.boxed = &boxed;
  }

  size_t size() const {
    TORCH_ILIST_DISPATCH_RETURN(size);
  }

  Iterator begin() const {
    TORCH_ILIST_DISPATCH_RETURN(begin);
  }

  Iterator end() const {
    TORCH_ILIST_DISPATCH_RETURN(end);
  }

  T& operator[](size_t i) const {
    TORCH_ILIST_DISPATCH_RETURN(get, i);
  }

#define DEFINE_CHECK(TAG)    \
  bool is##TAG() const {     \
    return tag_ == Tag::TAG; \
  }
  TORCH_ILIST_FORALL_TAGS(DEFINE_CHECK);
#undef DEFINE_CHECK

#define DEFINE_CASTING(TAG)                                   \
  typename TORCH_ILIST_IMPL(TAG, T)::self_t to##TAG() const { \
    TORCH_INTERNAL_ASSERT(is##TAG());                         \
    return TORCH_ILIST_IMPL(TAG, T)::unwrap(*this);           \
  }
  TORCH_ILIST_FORALL_TAGS(DEFINE_CASTING);
#undef DEFINE_CASTING

 private:
  Payload payload_;
  Tag tag_;

#define DEFINE_FRIEND_CLASS(TAG) friend class details::IList_##TAG##_Impl<T>;
  TORCH_ILIST_FORALL_TAGS(DEFINE_FRIEND_CLASS);
#undef DEFINE_FRIEND_CLASS
};

namespace details {
template <typename T>
class IList_Unboxed_Impl {
 public:
  using ilist_t = IList<T>;
  using self_t = typename ilist_t::unboxed_t;
  using iterator_t = typename ilist_t::Iterator;

  static self_t unwrap(const ilist_t& ilist) {
    return ilist.payload_.unboxed;
  }
  static size_t size(const ilist_t& ilist) {
    return unwrap(ilist).size();
  }
  static iterator_t get(const ilist_t& ilist, size_t i) {
    return unwrap(ilist)[i];
  }
  static iterator_t begin(const ilist_t& ilist) {
    return unwrap(ilist).begin();
  }
  static iterator_t end(const ilist_t& ilist) {
    return unwrap(ilist).end();
  }

  static T& iterator_get(const iterator_t& it) {
    return *it.impl_.unboxed_iterator;
  }
  static iterator_t iterator_pre_inc(const iterator_t& it) {
    ++it.impl_.unboxed_iterator;
    return it;
  }
  static iterator_t iterator_post_inc(const iterator_t& it) {
    auto old = it;
    ++it.impl_.unboxed_iterator;
    return old;
  }
  static iterator_t iterator_pre_dec(const iterator_t& it) {
    --it.impl_.unboxed_iterator;
    return it;
  }
  static iterator_t iterator_post_dec(const iterator_t& it) {
    auto old = it;
    --it.impl_.unboxed_iterator;
    return old;
  }
  static bool iterator_eq(const iterator_t& lhs, const iterator_t& rhs) {
    return lhs.impl_.unboxed_iterator == rhs.impl_.unboxed_iterator;
  }
};

template <typename T>
class IList_Boxed_Impl {
 public:
  using ilist_t = IList<T>;
  using self_t = typename ilist_t::boxed_t;
  using iterator_t = typename ilist_t::Iterator;

  static self_t unwrap(const ilist_t& ilist) {
    return ilist.payload_.boxed;
  }
  static size_t size(const ilist_t& ilist) {
    return unwrap(ilist)->size();
  }
  static iterator_t get(const ilist_t& ilist, size_t i) {
    return (*unwrap(ilist))[i];
  }
  static iterator_t begin(const ilist_t& ilist) {
    return unwrap(ilist)->begin();
  }
  static iterator_t end(const ilist_t& ilist) {
    return unwrap(ilist)->end();
  }

  static T& iterator_get(const iterator_t& it) {
    return T(*it.impl_.boxed_iterator);
  }
  static iterator_t iterator_pre_inc(const iterator_t& it) {
    ++it.impl_.boxed_iterator;
    return it;
  }
  static iterator_t iterator_post_inc(const iterator_t& it) {
    auto old = it;
    ++it.impl_.boxed_iterator;
    return old;
  }
  static iterator_t iterator_pre_dec(const iterator_t& it) {
    --it.impl_.boxed_iterator;
    return it;
  }
  static iterator_t iterator_post_dec(const iterator_t& it) {
    auto old = it;
    --it.impl_.boxed_iterator;
    return old;
  }
  static bool iterator_eq(const iterator_t& lhs, const iterator_t& rhs) {
    return lhs.impl_.boxed_iterator == rhs.impl_.boxed_iterator;
  }
};
} // namespace details

} // namespace c10
