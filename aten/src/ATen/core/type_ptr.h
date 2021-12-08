#pragma once

#include <memory>
#include <type_traits>

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>

namespace c10 {

// Compatibility wrapper around a raw pointer so that existing code
// written to deal with a shared_ptr can keep working.
template <typename T>
class SingletonTypePtr {
 public:
  /* implicit */ SingletonTypePtr(T* p) : repr_(p) {}

  template <typename U = T, std::enable_if_t<!std::is_same<std::remove_const_t<U>, void>::value, bool> = true>
  T& operator*() const {
    return *repr_;
  }

  T* get() const {
    return repr_;
  }

  T* operator->() const {
    return repr_;
  }

  operator bool() const {
    return repr_ != nullptr;
  }

 private:
  T* repr_;
};

template <typename T, typename U>
bool operator==(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return (void*)lhs.get() == (void*)rhs.get();
}

template <typename T, typename U>
bool operator!=(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return !(lhs == rhs);
}

template <typename T>
class SingletonOrSharedTypePtr {
 public:
  SingletonOrSharedTypePtr() = default;

  /* implicit */ SingletonOrSharedTypePtr(std::shared_ptr<T> x)
      : repr_(std::move(x)) {}

  template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
  /* implicit */ SingletonOrSharedTypePtr(std::shared_ptr<U> x)
      : repr_(std::move(x)) {}

  /* implicit */ SingletonOrSharedTypePtr(std::nullptr_t)
      : repr_(nullptr) {}

  /* implicit */ SingletonOrSharedTypePtr(SingletonTypePtr<T> p)
      : repr_(p) {}

  template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
  /* implicit */ SingletonOrSharedTypePtr(SingletonTypePtr<U> p)
      : repr_(SingletonTypePtr<T>(p.get())) {}


  // Ideally, we don't want to support this because it's not clear if
  // we are supposed to be taking shared ownership or not.
  /* implicit */ SingletonOrSharedTypePtr(T* p) = delete;


  T* get() const {
    return repr_.isSharedAndNonNull() ? repr_.shared_.get() : static_cast<T*>(repr_.rawRepr().first);
  }

  operator bool() const {
    return repr_.isNonNull();
  }

  bool operator==(std::nullptr_t) const {
    return !repr_.isNonNull();
  }

  bool operator!=(std::nullptr_t) const {
    return repr_.isNonNull();
  }

  template <typename U = T, std::enable_if_t<!std::is_same<std::remove_const_t<U>, void>::value, bool> = true>
  U& operator*() const {
    return *get();
  }

  T* operator->() const {
    return get();
  }

 private:
  union Repr {
    Repr() : Repr(nullptr) {}

    explicit Repr(std::shared_ptr<T> x)
        : shared_(std::move(x)) {}

    explicit Repr(std::nullptr_t)
        : singleton_(nullptr), unused_(nullptr) {}

    explicit Repr(SingletonTypePtr<T> p)
        : singleton_(p.get()), unused_(nullptr) {}

    ~Repr() {
      destroy();
    }

    // NOTE: the only non-UB way to access our null state is through
    // rawRepr(), because our copy operation doesn't preserve which
    // union member is active for null pointers.
    Repr(const Repr& rhs) {
      if (rhs.isSharedAndNonNull()) {
        new (&shared_) std::shared_ptr<T>(rhs.shared_);
      } else {
        singleton_ = static_cast<T*>(rhs.rawRepr().first);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.unused_ == nullptr);
        unused_ = nullptr;
      }
    }

    Repr(Repr&& rhs) noexcept(
#ifdef NDEBUG
        true
#else
        false
#endif
    ){
      if (rhs.isSharedAndNonNull()) {
        new (&shared_) std::shared_ptr<T>(std::move(rhs.shared_));
      } else {
        singleton_ = static_cast<T*>(rhs.rawRepr().first);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.unused_ == nullptr);
        unused_ = nullptr;
      }
    }

    Repr& operator=(const Repr& rhs) {
      if (&rhs == this) {
        return *this;
      }
      if (rhs.isSharedAndNonNull()) {
        if (isSharedAndNonNull()) {
          shared_ = rhs.shared_;
        } else {
          new (&shared_) std::shared_ptr<T>(rhs.shared_);
        }
      } else {
        if (isSharedAndNonNull()) {
          destroy();
        }
        singleton_ = static_cast<T*>(rhs.rawRepr().first);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.rawRepr().nullIfSingleton_ == nullptr);
        unused_ = nullptr;
      }
      return *this;
    }

  Repr& operator=(Repr&& rhs) noexcept(
#ifdef NDEBUG
      true
#else
      false
#endif
  ){
      if (&rhs == this) {
        return *this;
      }
      if (rhs.isSharedAndNonNull()) {
        if (isSharedAndNonNull()) {
          shared_ = std::move(rhs.shared_);
        } else {
          new (&shared_) std::shared_ptr<T>(std::move(rhs.shared_));
        }
      } else {
        if (isSharedAndNonNull()) {
          destroy();
        }
        singleton_ = static_cast<T*>(rhs.rawRepr().first);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.rawRepr().nullIfSingleton_ == nullptr);
        unused_ = nullptr;
      }
      return *this;
    }

    std::shared_ptr<T> shared_;

    struct {
      T* singleton_;
      void* unused_;
    };
    struct RawRepr {
      void* first;
      void* nullIfSingleton_;
    };

    // It is UB to read the singleton part of Repr if it was
    // constructed as a shared_ptr and vice versa, but memcpying out
    // the representation is always OK, so here's an accessor to obey
    // the letter of the law.
    RawRepr rawRepr() const {
      RawRepr repr;
      memcpy(&repr, reinterpret_cast<const char *>(this), sizeof(RawRepr));
      return repr;
    }

    bool isNonNull() const {
      auto repr = rawRepr();
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(repr.nullIfSingleton_ == nullptr || repr.first != nullptr);
      return repr.first != nullptr;
    }

    bool isSharedAndNonNull() const {
      return rawRepr().nullIfSingleton_ != nullptr;
    }

   private:
    void destroy() {
      if (isSharedAndNonNull()) {
        shared_.~shared_ptr();
      }
    }
  } repr_;
};

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const std::shared_ptr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const std::shared_ptr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const SingletonTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const SingletonTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const std::shared_ptr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const std::shared_ptr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const SingletonTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

struct Type;

using TypePtr = SingletonOrSharedTypePtr<Type>;
using ConstTypePtr = SingletonOrSharedTypePtr<const Type>;

// Explicitly enable MaybeOwned<shared_ptr<T>>, rather than allowing
// MaybeOwned to be used for any type right away.
template <typename T>
struct MaybeOwnedTraits<SingletonOrSharedTypePtr<T>>
    : public MaybeOwnedTraitsGenericImpl<SingletonOrSharedTypePtr<T>> {};

} // namespace c10

namespace std {
template <typename T>
struct hash<c10::SingletonOrSharedTypePtr<T>> {
  size_t operator()(const c10::SingletonOrSharedTypePtr<T>& x) const {
    return std::hash<T*>()(x.get());
  }
};
} // namespace std
