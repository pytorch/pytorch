#pragma once

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>
#include <atomic>
#include <climits>
#include <memory>

namespace pybind11 {
template <typename, typename...>
class class_;
}

namespace c10 {
class intrusive_ptr_target;
namespace raw {
namespace weak_intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
}
namespace intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
}

// constructor tag used by intrusive_ptr constructors
struct DontIncreaseRefcount {};
} // namespace raw
/**
 * intrusive_ptr<T> is an alternative to shared_ptr<T> that has better
 * performance because it does the refcounting intrusively
 * (i.e. in a member of the object itself).
 * Your class T needs to inherit from intrusive_ptr_target to allow it to be
 * used in an intrusive_ptr<T>. Your class's constructor should not allow
 *`this` to escape to other threads or create an intrusive_ptr from `this`.
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
// and make sure that the refcount isn't zero (or, a more subtle
// test for weak_intrusive_ptr<T>, for which the refcount may validly
// be zero, but the weak refcount better not be zero), because that
// tells us if the object was allocated by us.  If it wasn't, no
// intrusive_ptr for you!

class C10_API intrusive_ptr_target {
  // Note [Weak references for intrusive refcounting]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Here's the scheme:
  //
  //  - refcount == number of strong references to the object
  //    weakcount == number of weak references to the object,
  //      plus one more if refcount > 0
  //    An invariant: refcount > 0  =>  weakcount > 0
  //
  //  - c10::StorageImpl stays live as long as there are any strong
  //    or weak pointers to it (weakcount > 0, since strong
  //    references count as a +1 to weakcount)
  //
  //  - finalizers are called and data_ptr is deallocated when refcount == 0
  //
  //  - Once refcount == 0, it can never again be > 0 (the transition
  //    from > 0 to == 0 is monotonic)
  //
  //  - When you access c10::StorageImpl via a weak pointer, you must
  //    atomically increment the use count, if it is greater than 0.
  //    If it is not, you must report that the storage is dead.
  //
  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;

  template <typename T, typename NullType>
  friend class intrusive_ptr;
  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);

  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;
  friend inline void raw::weak_intrusive_ptr::incref(
      intrusive_ptr_target* self);

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;

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
#pragma warning(push)
#pragma warning( \
    disable : 4297) // function assumed not to throw an exception but does
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wexceptions"
#endif
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        // Second condition is there to accommodate
        // unsafe_adapt_non_heap_allocated: since we are doing our own
        // deallocation in that case, it is correct for each
        // expected_decref to have happened (some user code tried to
        // decref and thus free the object, but it didn't happen right
        // away) or not (no user code tried to free the object, and
        // now it's getting destroyed through whatever mechanism the
        // caller of unsafe_adapt_non_heap_allocated wanted to
        // use). We choose our reference count such that the count
        // will not dip below INT_MAX regardless.
        refcount_.load() == 0 || refcount_.load() >= INT_MAX,
        "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it; refcount was ",
        refcount_.load());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        // See ~intrusive_ptr for optimization that will frequently result in 1
        // at destruction time.
        weakcount_.load() == 1 || weakcount_.load() == 0 ||
            weakcount_.load() == INT_MAX - 1 || weakcount_.load() == INT_MAX,
        "Tried to destruct an intrusive_ptr_target that still has weak_intrusive_ptr to it");
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
  }

  constexpr intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}

  // intrusive_ptr_target supports copy and move: but refcount and weakcount
  // don't participate (since they are intrinsic properties of the memory
  // location)
  intrusive_ptr_target(intrusive_ptr_target&& /*other*/) noexcept
      : intrusive_ptr_target() {}

  intrusive_ptr_target& operator=(intrusive_ptr_target&& /*other*/) noexcept {
    return *this;
  }

  intrusive_ptr_target(const intrusive_ptr_target& /*other*/) noexcept
      : intrusive_ptr_target() {}

  intrusive_ptr_target& operator=(
      const intrusive_ptr_target& /*other*/) noexcept {
    return *this;
  }

 private:
  /**
   * This is called when refcount reaches zero.
   * You can override this to release expensive resources.
   * There might still be weak references, so your object might not get
   * destructed yet, but you can assume the object isn't used anymore,
   * i.e. no more calls to methods or accesses to members (we just can't
   * destruct it yet because we need the weakcount accessible).
   *
   * If there are no weak references (i.e. your class is about to be
   * destructed), this function WILL NOT be called.
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

template <class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();
  } else {
    return rhs;
  }
}

// Increment needs to be acquire-release to make use_count() and
// unique() reliable.
inline size_t atomic_refcount_increment(std::atomic<size_t>& refcount) {
  return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}

// weak_use_count() is only used for testing, so we don't need it to
// be reliable. Relaxed should be fine.
inline size_t atomic_weakcount_increment(std::atomic<size_t>& weakcount) {
  return weakcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Both decrements need to be acquire-release for correctness. See
// e.g. std::shared_ptr implementation.
inline size_t atomic_refcount_decrement(std::atomic<size_t>& refcount) {
  return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline size_t atomic_weakcount_decrement(std::atomic<size_t>& weakcount) {
  return weakcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

} // namespace detail

template <class TTarget, class NullType>
class weak_intrusive_ptr;

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
//      "intrusive_ptr can only be used for classes that inherit from
//      intrusive_ptr_target.");
#ifndef _WIN32
  // This static_assert triggers on MSVC
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_base_of<
          TTarget,
          typename std::remove_pointer<decltype(NullType::singleton())>::type>::
          value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;
  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
  friend class weak_intrusive_ptr<TTarget, NullType>;

  // Make pybind11::class_ be a friend class of intrusive_ptr, so that custom
  // smart holder in pybind11 could access the private constructor of
  // intrusive_ptr(T*) which took the ownership of the object. This is required
  // by customer holder macro PYBIND11_DECLARE_HOLDER_TYPE, where it uses
  // intrusive_ptr(TTarget*) to initialize and take ownership of the object. For
  // details, see
  // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
  template <typename, typename...>
  friend class pybind11::class_;

  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_refcount =
          detail::atomic_refcount_increment(target_->refcount_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() &&
        detail::atomic_refcount_decrement(target_->refcount_) == 0) {
      // See comment above about weakcount. As long as refcount>0,
      // weakcount is one larger than the actual number of weak references.
      // So we need to decrement it here.
      bool should_delete =
          target_->weakcount_.load(std::memory_order_acquire) == 1;
      if (!should_delete) {
        // justification for const_cast: release_resources is basically a
        // destructor and a destructor always mutates the object, even for const
        // objects. NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();
        should_delete =
            detail::atomic_weakcount_decrement(target_->weakcount_) == 0;
      }
      if (should_delete) {
        delete target_;
      }
    }
  }

  // raw pointer constructors are not public because we shouldn't make
  // intrusive_ptr out of raw pointers except from inside the make_intrusive(),
  // reclaim() and weak_intrusive_ptr::lock() implementations.

  // This constructor will increase the ref counter for you.
  // This constructor will be used by the make_intrusive(), and also pybind11,
  // which wrap the intrusive_ptr holder around the raw pointer and incref
  // correspondingly (pybind11 requires raw pointer constructor to incref by
  // default).
  explicit intrusive_ptr(TTarget* target)
      : intrusive_ptr(target, raw::DontIncreaseRefcount{}) {
    if (target_ != NullType::singleton()) {
      // We just created result.target_, so we know no other thread has
      // access to it, so we know we needn't care about memory ordering.
      // (On x86_64, a store with memory_order_relaxed generates a plain old
      // `mov`, whereas an atomic increment does a lock-prefixed `add`, which is
      // much more expensive: https://godbolt.org/z/eKPzj8.)
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          target_->refcount_ == 0 && target_->weakcount_ == 0,
          "intrusive_ptr: Newly-created target had non-zero refcounts. Does its "
          "constructor do something strange like incref or create an "
          "intrusive_ptr from `this`?");
      target_->refcount_.store(1, std::memory_order_relaxed);
      target_->weakcount_.store(1, std::memory_order_relaxed);
    }
  }

 public:
  using element_type = TTarget;

  intrusive_ptr() noexcept
      : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

  intrusive_ptr(std::nullptr_t) noexcept
      : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

  // This constructor will not increase the ref counter for you.
  // We use the tagged dispatch mechanism to explicitly mark this constructor
  // to not increase the refcount
  explicit intrusive_ptr(TTarget* target, raw::DontIncreaseRefcount) noexcept
      : target_(target) {}

  explicit intrusive_ptr(std::unique_ptr<TTarget> rhs) noexcept
      : intrusive_ptr(rhs.release()) {}

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(const intrusive_ptr<From, FromNullType>& rhs)
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
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
  intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
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
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
    return target_;
  }

  operator bool() const noexcept {
    return target_ != NullType::singleton();
  }

  void reset() noexcept {
    reset_();
    target_ = NullType::singleton();
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
    return target_->refcount_.load(std::memory_order_acquire);
  }

  size_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount_.load(std::memory_order_acquire);
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
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
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
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        owning_ptr == NullType::singleton() ||
            owning_ptr->refcount_.load() == 0 || owning_ptr->weakcount_.load(),
        "TTarget violates the invariant that refcount > 0  =>  weakcount > 0");
    return intrusive_ptr(owning_ptr, raw::DontIncreaseRefcount{});
  }

  /**
   * Takes an owning pointer to TTarget* and creates an intrusive_ptr
   * representing a new reference, i.e. the raw pointer retains
   * ownership.
   */
  static intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    return ret;
  }

  /**
   * Allocate a heap object with args and wrap it inside a intrusive_ptr and
   * incref. This is a helper function to let make_intrusive() access private
   * intrusive_ptr constructors.
   */
  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    return intrusive_ptr(new TTarget(std::forward<Args>(args)...));
  }

  /**
   * Turn a new instance of TTarget (e.g., literally allocated
   * using new TTarget(...) into an intrusive_ptr.  If possible,
   * use intrusive_ptr::make instead which statically guarantees
   * that the allocation was done properly.
   *
   * At the moment, the only reason this method exists is because
   * pybind11 holder types expect to be able to allocate in
   * this way (because pybind11 handles the new allocation itself).
   */
  static intrusive_ptr unsafe_steal_from_new(TTarget* raw_ptr) {
    return intrusive_ptr(raw_ptr);
  }

  /**
   * Turn an instance of TTarget that should not be reference counted
   * (e.g., allocated into an arena with placement new) into an
   * intrusive_ptr. This is gratuitously unsafe and should only be
   * used if you can guarantee that the pointer will not escape and be
   * refcounted as normal.
   *
   * `expected_decrefs` is a debugging parameter: it indicates the
   * number of strong owners the intrusive_ptr_target in question is
   * expected to get. In most use cases, this will likely be 1.
   *
   * The reason this method exists is for manually sharing
   * StorageImpls across Tensors in the static runtime. It needs
   * access to private intrusive_ptr members so that the refcounts can
   * be initialized to custom values.
   */
  static intrusive_ptr unsafe_adapt_non_heap_allocated(
      TTarget* raw_ptr,
      size_t expected_decrefs) {
    intrusive_ptr result(raw_ptr, raw::DontIncreaseRefcount{});
    // INT_MAX is impractically huge for a reference count, while
    // being in no danger of overflowing size_t. We actually only need to
    // initialize the refcount to 2 -- we are just doing an unbalanced
    // incref to prevent the non-heap-allocated target from being
    // freed, and we are optimizing that incref by directly
    // initializing the refcounts rather than doing an expensive
    // atomic increment. The reason to use INT_MAX is to accommodate
    // the debug assertions in ~intrusive_ptr_target.
#ifdef NDEBUG
    expected_decrefs = 0;
#endif
    result.target_->refcount_.store(
        INT_MAX + expected_decrefs, std::memory_order_relaxed);
    result.target_->weakcount_.store(INT_MAX, std::memory_order_relaxed);
    return result;
  }

  /**
   * Turn a **non-owning raw pointer** to an intrusive_ptr.  It is
   * the moral equivalent of enable_shared_from_this on a shared pointer.
   *
   * This method is only valid for objects that are already live.  If
   * you are looking for the moral equivalent of unique_ptr<T>(T*)
   * constructor, see steal_from_new.
   *
   * TODO: https://github.com/pytorch/pytorch/issues/56482
   */
  static intrusive_ptr unsafe_reclaim_from_nonowning(TTarget* raw_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
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

template <class TTarget1, class NullType1>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    std::nullptr_t) noexcept {
  return lhs.get() == nullptr;
}

template <class TTarget2, class NullType2>
inline bool operator==(
    std::nullptr_t,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return nullptr == rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

template <class TTarget1, class NullType1>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    std::nullptr_t) noexcept {
  return !operator==(lhs, nullptr);
}

template <class TTarget2, class NullType2>
inline bool operator!=(
    std::nullptr_t,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(nullptr, rhs);
}
template <typename T>
struct MaybeOwnedTraits<c10::intrusive_ptr<T>> {
  using owned_type = c10::intrusive_ptr<T>;
  using borrow_type = c10::intrusive_ptr<T>;

  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type::reclaim(from.get());
  }

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.release();
    lhs = borrow_type::reclaim(rhs.get());
  }

  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.release();
  }

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
    return true;
  }
};

template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
 private:
  static_assert(
      std::is_base_of<intrusive_ptr_target, TTarget>::value,
      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
#ifndef _WIN32
  // This static_assert triggers on MSVC
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_base_of<
          TTarget,
          typename std::remove_pointer<decltype(NullType::singleton())>::type>::
          value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class weak_intrusive_ptr;

  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_weakcount =
          detail::atomic_weakcount_increment(target_->weakcount_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_weakcount != 1,
          "weak_intrusive_ptr: Cannot increase weakcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() &&
        detail::atomic_weakcount_decrement(target_->weakcount_) == 0) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
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
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  weak_intrusive_ptr(const weak_intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      const weak_intrusive_ptr<From, FromNullType>& rhs)
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr copy constructor got pointer of wrong type.");
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
      weak_intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move assignment got pointer of wrong type.");
    weak_intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
    return operator=<TTarget, NullType>(rhs);
  }

  weak_intrusive_ptr& operator=(
      const intrusive_ptr<TTarget, NullType>& rhs) & noexcept {
    weak_intrusive_ptr tmp(rhs);
    swap(tmp);
    return *this;
  }

  template <class From, class FromNullType>
  weak_intrusive_ptr& operator=(
      const weak_intrusive_ptr<From, NullType>& rhs) & {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr copy assignment got pointer of wrong type.");
    weak_intrusive_ptr tmp = rhs;
    swap(tmp);
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

  // NB: This should ONLY be used by the std::hash implementation
  // for weak_intrusive_ptr.  Another way you could do this is
  // friend std::hash<weak_intrusive_ptr>, but this triggers two
  // bugs:
  //
  //  (1) It triggers an nvcc bug, where std::hash in a friend class
  //      declaration gets preprocessed into hash, which then cannot
  //      actually be found.  The error in this case looks like:
  //
  //        error: no template named 'hash'; did you mean 'std::hash'?
  //
  //  (2) On OS X, std::hash is declared as a struct, not a class.
  //      This twings:
  //
  //        error: class 'hash' was previously declared as a struct
  //        [-Werror,-Wmismatched-tags]
  //
  // Both of these are work-aroundable, but on the whole, I decided
  // it would be simpler and easier to make work if we just expose
  // an unsafe getter for target_
  //
  TTarget* _unsafe_get_target() const noexcept {
    return target_;
  }

  size_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount_.load(
        std::memory_order_acquire); // refcount, not weakcount!
  }

  size_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount_.load(std::memory_order_acquire);
  }

  bool expired() const noexcept {
    return use_count() == 0;
  }

  intrusive_ptr<TTarget, NullType> lock() const noexcept {
    if (expired()) {
      return intrusive_ptr<TTarget, NullType>();
    } else {
      auto refcount = target_->refcount_.load(std::memory_order_seq_cst);
      do {
        if (refcount == 0) {
          // Object already destructed, no strong references left anymore.
          // Return nullptr.
          return intrusive_ptr<TTarget, NullType>();
        }
      } while (
          !target_->refcount_.compare_exchange_weak(refcount, refcount + 1));
      return intrusive_ptr<TTarget, NullType>(
          target_, raw::DontIncreaseRefcount{});
    }
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
   * This means that the weakcount is not increased.
   * This is the counter-part to weak_intrusive_ptr::release() and the pointer
   * passed in *must* have been created using weak_intrusive_ptr::release().
   */
  static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    // if refcount > 0, weakcount must be >1 for weak references to exist.
    // see weak counting explanation at top of this file.
    // if refcount == 0, weakcount only must be >0.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        owning_weak_ptr == NullType::singleton() ||
            owning_weak_ptr->weakcount_.load() > 1 ||
            (owning_weak_ptr->refcount_.load() == 0 &&
             owning_weak_ptr->weakcount_.load() > 0),
        "weak_intrusive_ptr: Can only weak_intrusive_ptr::reclaim() owning pointers that were created using weak_intrusive_ptr::release().");
    return weak_intrusive_ptr(owning_weak_ptr);
  }

  /**
   * Takes a pointer to TTarget* (may be weak or strong) and creates a
   * new weak_intrusive_ptr representing a new weak reference, i.e.
   * the raw pointer retains ownership.
   */
  static weak_intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    return ret;
  }

  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator<(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator==(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
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

// Alias for documentary purposes, to more easily distinguish
// weak raw intrusive pointers from intrusive pointers.
using weak_intrusive_ptr_target = intrusive_ptr_target;

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
// functions on strong pointers, and weak_intrusive_ptr namespace functions
// on weak pointers.  If you mix it up, you may get an assert failure.
namespace raw {

namespace intrusive_ptr {

// WARNING: Unlike the reclaim() API, it is NOT valid to pass
// NullType::singleton to this function
inline void incref(intrusive_ptr_target* self) {
  if (self) {
    detail::atomic_refcount_increment(self->refcount_);
  }
}

// WARNING: Unlike the reclaim() API, it is NOT valid to pass
// NullType::singleton to this function
inline void decref(intrusive_ptr_target* self) {
  // Let it die
  c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  // NB: Caller still has 'self' pointer, but it's now invalid.
  // If you want more safety, used the actual c10::intrusive_ptr class
}

template <typename T>
inline T* make_weak(T* self) {
  // NB: 'this' is a strong pointer, but we return a weak pointer
  auto ptr = c10::intrusive_ptr<T>::reclaim(self);
  c10::weak_intrusive_ptr<T> wptr(ptr);
  ptr.release();
  return wptr.release();
}

inline size_t use_count(intrusive_ptr_target* self) {
  auto ptr = c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  auto r = ptr.use_count();
  ptr.release();
  return r;
}

} // namespace intrusive_ptr

namespace weak_intrusive_ptr {

inline void incref(weak_intrusive_ptr_target* self) {
  detail::atomic_weakcount_increment(self->weakcount_);
}

inline void decref(weak_intrusive_ptr_target* self) {
  // Let it die
  c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  // NB: You still "have" the 'self' pointer, but it's now invalid.
  // If you want more safety, used the actual c10::weak_intrusive_ptr class
}

template <typename T>
inline T* lock(T* self) {
  auto wptr = c10::weak_intrusive_ptr<T>::reclaim(self);
  auto ptr = wptr.lock();
  wptr.release();
  return ptr.release();
}

// This gives the STRONG refcount of a WEAK pointer
inline size_t use_count(weak_intrusive_ptr_target* self) {
  auto wptr = c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  auto r = wptr.use_count();
  wptr.release();
  return r;
}

} // namespace weak_intrusive_ptr

} // namespace raw

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
    return std::hash<TTarget*>()(x._unsafe_get_target());
  }
};
} // namespace std
