#pragma once

#include <ATen/core/Error.h>
#include <atomic>
#include <stdexcept>
#include <ATen/core/C++17.h>

namespace c10 {

/**
 * intrusive_ptr<T> is an alternative to shared_ptr<T> that has better
 * performance because it does the refcounting intrusively
 * (i.e. in a member of the object itself).
 * Your class T needs to inherit from intrusive_ptr_target to allow it to be
 * used in an intrusive_ptr<T>.
 */

class intrusive_ptr_target {
  // Note [Weak references for intrusive refcounting]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Here's the scheme:
  //
  //  - refcount == number of strong references to the object
  //    weakcount == number of weak references to the object,
  //      plus one more if refcount > 0
  //    An invariant: refcount > 0  =>  weakcount > 0
  //
  //  - THStorage stays live as long as there are any strong
  //    or weak pointers to it (weakcount > 0, since strong
  //    references count as a +1 to weakcount)
  //
  //  - finalizers are called and data_ptr is deallocated when refcount == 0
  //
  //  - Once refcount == 0, it can never again be > 0 (the transition
  //    from > 0 to == 0 is monotonic)
  //
  //  - When you access THStorage via a weak pointer, you must
  //    atomically increment the use count, if it is greater than 0.
  //    If it is not, you must report that the storage is dead.
  //
  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;

  template <typename T, typename NullType>
  friend class intrusive_ptr;
  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;

 protected:
  // protected destructor. We never want to destruct intrusive_ptr_target*
  // directly.
  virtual ~intrusive_ptr_target() {
// Disable -Wterminate and -Wexceptions so we're allowed to use assertions
// (i.e. throw exceptions) in a destructor.
// We also have to disable -Wunknown-warning-option and -Wpragmas, because
// some other compilers don't know about -Wterminate or -Wexceptions and
// will show a warning about unknown warning options otherwise.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wexceptions"
    AT_ASSERTM(
        refcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it");
    AT_ASSERTM(
        weakcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has weak_intrusive_ptr to it");
#pragma GCC diagnostic pop
  }

  constexpr intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}

 private:
  /**
   * This is called when refcount reaches zero.
   * You can override this to release expensive resources.
   * There might still be weak references, so your object might not get
   * destructed yet, but you can assume the object isn't used anymore,
   * i.e. no more calls to methods or accesses to members (we just can't
   * destruct it yet because we need the weakcount accessible).
   *
   * Even if there are no weak references (i.e. your class is about to be
   * destructed), this function is guaranteed to be called first.
   * However, if you use your class for an object on the stack that is
   * destructed by the scope (i.e. without intrusive_ptr), this function will
   * not be called.
   */
  virtual void release_resources() {}
};

namespace detail {
template <class TTarget>
struct intrusive_target_default_null_type final {
  static constexpr TTarget* singleton() noexcept {
    return nullptr;
  }
};
} // namespace detail

template <class TTarget, class NullType>
class weak_intrusive_ptr;

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
 private:
  static_assert(
      std::is_base_of<intrusive_ptr_target, TTarget>::value,
      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
  static_assert(
      std::is_same<TTarget*, decltype(NullType::singleton())>::value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
  friend class weak_intrusive_ptr<TTarget, NullType>;

  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_refcount = ++target_->refcount_;
      AT_ASSERTM(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() && --target_->refcount_ == 0) {
      // See comment above about weakcount. As long as refcount>0,
      // weakcount is one larger than the actual number of weak references.
      // So we need to decrement it here.
      auto weak_count = --target_->weakcount_;
      // justification for const_cast: release_resources is basically a destructor
      // and a destructor always mutates the object, even for const objects.
      const_cast<c10::guts::remove_const_t<TTarget>*>(target_)->release_resources();
      if (weak_count == 0) {
        delete target_;
      }
    }
    target_ = NullType::singleton();
  }

  // This constructor will not increase the ref counter for you.
  // This is not public because we shouldn't make intrusive_ptr out of raw
  // pointers except from inside the make_intrusive() and
  // weak_intrusive_ptr::lock() implementations
  explicit intrusive_ptr(TTarget* target) noexcept : target_(target) {}

 public:
  using element_type = TTarget;

  intrusive_ptr() noexcept : intrusive_ptr(NullType::singleton()) {}

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(rhs.target_) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. intrusive_ptr move constructor got pointer with differing null value.");
    rhs.target_ = FromNullType::singleton();
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(
      const intrusive_ptr<From, FromNullType>& rhs)
      : target_(rhs.target_) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. intrusive_ptr copy constructor got pointer with differing null value.");
    retain_();
  }

  ~intrusive_ptr() noexcept {
    reset_();
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
    return operator=<TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
      intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) &
      noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. intrusive_ptr move assignment got pointer with differing null value.");
    reset_();
    target_ = rhs.target_;
    rhs.target_ = FromNullType::singleton();
    return *this;
  }

  intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
    return operator=<TTarget, NullType>(rhs);
  }

  template <class From, class FromNullType>
      intrusive_ptr& operator=(const intrusive_ptr<From, NullType>& rhs) & {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy assignment got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. intrusive_ptr copy assignment got pointer with differing null value.");
    reset_();
    target_ = rhs.target_;
    retain_();
    return *this;
  }

  TTarget* get() const noexcept {
    return target_;
  }

  const TTarget& operator*() const noexcept {
    return *target_;
  }

  TTarget& operator*() noexcept {
    return *target_;
  }

  const TTarget* operator->() const noexcept {
    return target_;
  }

  TTarget* operator->() noexcept {
    return target_;
  }

  void reset() noexcept {
    reset_();
  }

  void swap(intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  bool defined() const noexcept {
    return target_ != NullType::singleton();
  }

  size_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount_.load();
  }

  bool unique() const noexcept {
    return use_count() == 1;
  }

  /**
   * Returns an owning (!) pointer to the underlying object and makes the
   * intrusive_ptr instance invalid. That means the refcount is not decreased.
   * You *must* put the returned pointer back into a intrusive_ptr using
   * intrusive_ptr::reclaim(ptr) to properly destruct it.
   * This is helpful for C APIs.
   */
  TTarget* release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * Takes an owning pointer to TTarget* and creates an intrusive_ptr that takes
   * over ownership. Thas means the refcount is not increased.
   * This is the counter-part to intrusive_ptr::release() and the pointer
   * passed in *must* have been created using intrusive_ptr::release().
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr) {
    AT_ASSERTM(
        owning_ptr->refcount_.load() > 0,
        "intrusive_ptr: Can only intrusive_ptr::reclaim() owning pointers that were created using intrusive_ptr::release().");
    return intrusive_ptr(owning_ptr);
  }

  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    auto result = intrusive_ptr(new TTarget(std::forward<Args>(args)...));
    // We can't use retain_(), because we also have to increase weakcount
    // and because we allow raising these values from 0, which retain_()
    // has an assertion against.
    ++result.target_->refcount_;
    ++result.target_->weakcount_;

    return result;
  }
};

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>,
    class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args&&... args) {
  return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget, class NullType>
inline void swap(
    intrusive_ptr<TTarget, NullType>& lhs,
    intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// To allow intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() < rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
 private:
  static_assert(
      std::is_base_of<intrusive_ptr_target, TTarget>::value,
      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
  static_assert(
      std::is_same<TTarget*, decltype(NullType::singleton())>::value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class weak_intrusive_ptr;

  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_weakcount = ++target_->weakcount_;
      AT_ASSERTM(
          new_weakcount != 1,
          "weak_intrusive_ptr: Cannot increase weakcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() && --target_->weakcount_ == 0) {
      delete target_;
    }
    target_ = NullType::singleton();
  }

  constexpr explicit weak_intrusive_ptr(TTarget* target) : target_(target) {}

 public:
  using element_type = TTarget;

  explicit weak_intrusive_ptr(const intrusive_ptr<TTarget, NullType>& ptr)
      : weak_intrusive_ptr(ptr.get()) {
    retain_();
  }

  weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      weak_intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(rhs.target_) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. weak_intrusive_ptr move constructor got pointer with differing null value.");
    rhs.target_ = FromNullType::singleton();
  }

  weak_intrusive_ptr(const weak_intrusive_ptr& rhs)
      : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      const weak_intrusive_ptr<From, FromNullType>& rhs)
      : target_(rhs.target_) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr copy constructor got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. weak_intrusive_ptr copy constructor got pointer with differing null value.");
    retain_();
  }

  ~weak_intrusive_ptr() noexcept {
    reset_();
  }

  weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) & noexcept {
    return operator=<TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
      weak_intrusive_ptr& operator=(
          weak_intrusive_ptr<From, FromNullType>&& rhs) &
      noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move assignment got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. weak_intrusive_ptr move assignment got pointer with differing null value.");
    reset_();
    target_ = rhs.target_;
    rhs.target_ = FromNullType::singleton();
    return *this;
  }

  weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
    return operator=<TTarget, NullType>(rhs);
  }

  template <class From, class FromNullType>
      weak_intrusive_ptr& operator=(
          const weak_intrusive_ptr<From, NullType>& rhs) & {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr copy assignment got pointer of wrong type.");
    static_assert(
        NullType::singleton() == FromNullType::singleton(),
        "NullType mismatch. weak_intrusive_ptr copy assignment got pointer with differing null value.");
    reset_();
    target_ = rhs.target_;
    retain_();
    return *this;
  }

  void reset() noexcept {
    reset_();
  }

  void swap(weak_intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  size_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount_.load(); // refcount, not weakcount!
  }

  bool expired() const noexcept {
    return use_count() == 0;
  }

  intrusive_ptr<TTarget, NullType> lock() const noexcept {
    auto refcount = target_->refcount_.load();
    do {
      if (refcount == 0) {
        // Object already destructed, no strong references left anymore.
        // Return nullptr.
        return intrusive_ptr<TTarget, NullType>(NullType::singleton());
      }
    } while (target_->refcount_.compare_exchange_weak(refcount, refcount + 1));
    return intrusive_ptr<TTarget, NullType>(target_);
  }

  /**
   * Returns an owning (but still only weakly referenced) pointer to the
   * underlying object and makes the weak_intrusive_ptr instance invalid.
   * That means the weakcount is not decreased.
   * You *must* put the returned pointer back into a weak_intrusive_ptr using
   * weak_intrusive_ptr::reclaim(ptr) to properly destruct it.
   * This is helpful for C APIs.
   */
  TTarget* release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * Takes an owning (but must be weakly referenced) pointer to TTarget* and
   * creates a weak_intrusive_ptr that takes over ownership.
   * Thas means the weakcount is not increased.
   * This is the counter-part to weak_intrusive_ptr::release() and the pointer
   * passed in *must* have been created using weak_intrusive_ptr::release().
   */
  static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr) {
    // if refcount > 0, weakcount must be >1 for weak references to exist.
    // see weak counting explanation at top of this file.
    // if refcount == 0, weakcount only must be >0.
    AT_ASSERTM(
        owning_weak_ptr->weakcount_.load() > 1 ||
            (owning_weak_ptr->refcount_.load() == 0 &&
             owning_weak_ptr->weakcount_.load() > 0),
        "weak_intrusive_ptr: Can only weak_intrusive_ptr::reclaim() owning pointers that were created using weak_intrusive_ptr::release().");
    return weak_intrusive_ptr(owning_weak_ptr);
  }

  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator<(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator==(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
  friend class std::hash<weak_intrusive_ptr>;
};

template <class TTarget, class NullType>
inline void swap(
    weak_intrusive_ptr<TTarget, NullType>& lhs,
    weak_intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// To allow weak_intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ < rhs.target_;
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ == rhs.target_;
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

} // namespace c10

namespace std {
// To allow intrusive_ptr and weak_intrusive_ptr inside std::unordered_map or
// std::unordered_set, we need std::hash
template <class TTarget, class NullType>
struct hash<c10::intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x.get());
  }
};
template <class TTarget, class NullType>
struct hash<c10::weak_intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::weak_intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x.target_);
  }
};
} // namespace std
