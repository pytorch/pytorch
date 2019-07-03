#pragma once

#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <stdexcept>

namespace c10 {
class intrusive_ptr_target;
/**
 * intrusive_ptr<T> is an alternative to shared_ptr<T> that has better
 * performance because it does the refcounting intrusively
 * (i.e. in a member of the object itself).
 * Your class T needs to inherit from intrusive_ptr_target to allow it to be
 * used in an intrusive_ptr<T>.
 */

// Note [Stack allocated intrusive_ptr_target safety]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A well known problem with std::enable_shared_from_this is that it
// allows you to create a std::shared_ptr from a stack allocated object,
// which is totally bogus because the object will die once you return
// from the stack.  In intrusive_ptr, we can detect that this has occurred,
// because we set the refcount/weakcount of objects which inherit from
// intrusive_ptr_target to zero, *unless* we can prove that the object
// was dynamically allocated (e.g., via make_intrusive).
//
// Thus, whenever you transmute a T* into a intrusive_ptr<T>, we check
// and make sure that the refcount isn't zero, because that
// tells us if the object was allocated by us.  If it wasn't, no
// intrusive_ptr for you!

class C10_API intrusive_ptr_target {
  mutable std::atomic<size_t> refcount_;
  static std::atomic<int64_t> unique_id_counter_;
  int64_t unique_id_;

  template <typename T, typename NullType>
  friend class intrusive_ptr;

 public:
  int64_t unique_id() {
    if (unique_id_ == 0) {
      unique_id_ = ++unique_id_counter_;
    }
    return unique_id_;
  }

 protected:
  // protected destructor. We never want to destruct intrusive_ptr_target*
  // directly.
  virtual ~intrusive_ptr_target() {
// Disable -Wterminate and -Wexceptions so we're allowed to use assertions
// (i.e. throw exceptions) in a destructor.
// We also have to disable -Wunknown-warning-option and -Wpragmas, because
// some other compilers don't know about -Wterminate or -Wexceptions and
// will show a warning about unknown warning options otherwise.
#if defined(_MSC_VER) && !defined(__clang__)
#  pragma warning(push)
#  pragma warning(disable: 4297) // function assumed not to throw an exception but does
#else
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wunknown-warning-option"
#  pragma GCC diagnostic ignored "-Wterminate"
#  pragma GCC diagnostic ignored "-Wexceptions"
#endif
    AT_ASSERTM(
        refcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it");
#if defined(_MSC_VER) && !defined(__clang__)
#  pragma warning(pop)
#else
#  pragma GCC diagnostic pop
#endif
  }

  constexpr intrusive_ptr_target() noexcept : refcount_(0), unique_id_(0) {}

  // intrusive_ptr_target supports copy and move: but refcount and weakcount don't
  // participate (since they are intrinsic properties of the memory location)
  intrusive_ptr_target(intrusive_ptr_target&& other) noexcept : intrusive_ptr_target() {}
  intrusive_ptr_target& operator=(intrusive_ptr_target&& other) noexcept { return *this; }
  intrusive_ptr_target(const intrusive_ptr_target& other) noexcept : intrusive_ptr_target() {}
  intrusive_ptr_target& operator=(const intrusive_ptr_target& other) noexcept { return *this; }

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

template<class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();
  } else {
    return rhs;
  }
}
} // namespace detail

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
 private:
//  the following static assert would be nice to have but it requires
//  the target class T to be fully defined when intrusive_ptr<T> is instantiated
//  this is a problem for classes that contain pointers to themselves
//  static_assert(
//      std::is_base_of<intrusive_ptr_target, TTarget>::value,
//      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
#ifndef _WIN32
  // This static_assert triggers on MSVC
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_same<TTarget*, decltype(NullType::singleton())>::value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;

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
      // justification for const_cast: release_resources is basically a destructor
      // and a destructor always mutates the object, even for const objects.
      const_cast<c10::guts::remove_const_t<TTarget>*>(target_)->release_resources();
      delete target_;
    }
    target_ = NullType::singleton();
  }

  // This constructor will not increase the ref counter for you.
  // This is not public because we shouldn't make intrusive_ptr out of raw
  // pointers except from inside the make_intrusive() implementation
  explicit intrusive_ptr(TTarget* target) noexcept : target_(target) {}

 public:
  using element_type = TTarget;

  intrusive_ptr() noexcept : intrusive_ptr(NullType::singleton()) {}

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(
      const intrusive_ptr<From, FromNullType>& rhs)
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
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
    intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
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
    intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  TTarget* get() const noexcept {
    return target_;
  }

  TTarget& operator*() const noexcept {
    return *target_;
  }

  TTarget* operator->() const noexcept {
    return target_;
  }

  operator bool() const noexcept {
    return target_ != NullType::singleton();
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
   * over ownership. That means the refcount is not increased.
   * This is the counter-part to intrusive_ptr::release() and the pointer
   * passed in *must* have been created using intrusive_ptr::release().
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    AT_ASSERTM(
        owning_ptr == NullType::singleton() || owning_ptr->refcount_.load() > 0,
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

    return result;
  }

  /**
   * Turn a **non-owning raw pointer** to an intrusive_ptr.
   *
   * This method is potentially dangerous (as it can mess up refcount).
   */
  static intrusive_ptr unsafe_reclaim_from_nonowning(TTarget* raw_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    AT_ASSERTM(
        raw_ptr == NullType::singleton() || raw_ptr->refcount_.load() > 0,
        "intrusive_ptr: Can only reclaim pointers that are owned by someone");
    auto ptr = reclaim(raw_ptr); // doesn't increase refcount
    ptr.retain_();
    return ptr;
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

// This namespace provides some methods for working with
// raw pointers that subclass intrusive_ptr_target.  They are not provided
// as methods on intrusive_ptr_target, because ideally you would not need these
// methods at all (use smart pointers), but if you are dealing with legacy code
// that still needs to pass around raw pointers, you may find these quite
// useful.
//
// An important usage note: some functions are only valid if you have a
// strong raw pointer to the object, while others are only valid if you
// have a weak raw pointer to the object.  ONLY call intrusive_ptr namespace
// functions on strong pointers. If you mix it up, you may get an assert failure.
namespace raw {

namespace intrusive_ptr {

  // WARNING: Unlike the reclaim() API, it is NOT valid to pass
  // NullType::singleton to this function
  inline void incref(intrusive_ptr_target* self) {
    auto ptr = c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    auto ptr_copy = ptr;
    ptr_copy.release();
    ptr.release();
  }

  // WARNING: Unlike the reclaim() API, it is NOT valid to pass
  // NullType::singleton to this function
  inline void decref(intrusive_ptr_target* self) {
    // Let it die
    c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    // NB: Caller still has 'self' pointer, but it's now invalid.
    // If you want more safety, used the actual c10::intrusive_ptr class
  }

  inline uint32_t use_count(intrusive_ptr_target* self) {
    auto ptr = c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    auto r = ptr.use_count();
    ptr.release();
    return r;
  }

} // namespace intrusive_ptr_target

} // namespace raw

} // namespace c10

namespace std {
// To allow intrusive_ptr inside std::unordered_map or
// std::unordered_set, we need std::hash
template <class TTarget, class NullType>
struct hash<c10::intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x.get());
  }
};
} // namespace std
