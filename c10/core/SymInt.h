#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <memory>
#include <numeric>
#include <utility>

namespace c10 {

class SymFloat;

// SymInt represents either a regular int64_t, or a symbolic integer
// (represented in a type erased way as SymNode).  The intention is for SymInt
// to represent symbolic sizes that arise when doing shape computation in
// operator kernels. This allows for tracing through programs without baking in
// concrete sizes into kernel calls.
//
// SymInt has an API equivalent to int64_t.  In particular, it is a value type.
// Internally, SymInt is represented in a clever packed way, so that it only
// occupies one word of space; but morally, it is a union between an int64_t
// and an intrusive pointer to SymNodeImpl.
//
// Invariant: the referenced SymNodeImpl is guaranteed to be a SymNode where
// is_int() returns true

class C10_API SymInt {
 public:
  enum Unchecked {
    UNCHECKED,
  };

  /*implicit*/ SymInt(int64_t d) : data_(d) {
    // NB: this relies on exception in constructor inhibiting
    // destructor; otherwise we would attempt to deallocate
    // the garbage data!
    TORCH_CHECK(!is_symbolic());
  };
  SymInt() : data_(0) {}
  SymInt(SymNode n);

  // unchecked c-tor accepting raw `data_`
  // One appropriate use for this is when you are constructing a symint
  // in a situation where you know it is non-negative (or, if it is negative,
  // the negative value is -1; i.e., not user controlled)
  SymInt(Unchecked, int64_t d) : data_(d) {}

  // TODO: these implementations are not optimal because they allocate a
  // temporary and then use the move constructor/assignment
  SymInt(const SymInt& s) : data_(0) {
    if (s.is_symbolic()) {
      *this = SymInt(s.toSymNodeImpl());
    } else {
      data_ = s.data_;
    }
  }
  SymInt(SymInt&& s) noexcept : data_(s.data_) {
    s.data_ = 0;
  }

  SymInt& operator=(const SymInt& s) {
    if (this != &s) {
      if (s.is_symbolic()) {
        *this = SymInt(s.toSymNodeImpl());
      } else {
        data_ = s.data_;
      }
    }
    return *this;
  }
  SymInt& operator=(SymInt&& s) noexcept {
    if (this != &s) {
      release_(); // release the current SymNode if any
      data_ = s.data_;
      if (s.is_symbolic())
        s.data_ = 0;
    };
    return *this;
  }

  SymInt clone() const {
    if (is_symbolic()) {
      return SymInt(toSymNodeImplUnowned()->clone());
    }
    return *this;
  }

  SymNodeImpl* toSymNodeImplUnowned() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(is_symbolic());
    uint64_t unextended_bits = static_cast<uint64_t>(data_) & ~MASK;
    uint64_t sign_bit_mask = 1ULL << (62 - 1);
    // https://stackoverflow.com/questions/42534749/signed-extension-from-24-bit-to-32-bit-in-c
    uint64_t extended_bits = (unextended_bits ^ sign_bit_mask) - sign_bit_mask;
    return static_cast<SymNodeImpl*>(
        reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }

  void release_() {
    if (is_symbolic()) {
      SymNode::reclaim(toSymNodeImplUnowned()); // steal
    }
  }

  SymNodeImpl* release() && {
#ifndef C10_MOBILE
    TORCH_INTERNAL_ASSERT(is_symbolic());
    auto* r = toSymNodeImplUnowned();
    data_ = 0; // transfer ownership
    return r;
#else
    TORCH_INTERNAL_ASSERT(false);
#endif
  }

  SymNode toSymNodeImpl() const;

  ~SymInt() {
    release_();
  }

  // Require the int to be non-symbolic, and if it is symbolic raise an
  // error.  This is safe to use for C++ code that doesn't work for symbolic
  // shapes, and you don't have time to fix it immediately, as if we
  // try to trigger the path in C++ you'll appropriately get an error
  int64_t expect_int() const {
    TORCH_CHECK(!is_symbolic());
    return data_;
  }

  // Insert a guard for the int to be its concrete value, and then return
  // that value.  This operation always works, even if the int is symbolic,
  // so long as we know what the underlying value is (e.g., this won't work
  // if you call it on the size of nonzero output).  Don't blindly put this
  // everywhere; you can cause overspecialization of PyTorch programs with
  // this method.
  //
  // It should be called as guard_int(__FILE__, __LINE__).  The file and line
  // number can be used to diagnose overspecialization.
  int64_t guard_int(const char* file, int64_t line) const;

  // N.B. It's important to keep this definition in the header
  // as we expect if checks to be folded for mobile builds
  // where `is_symbolic` is always false and optimize dead code paths
  C10_ALWAYS_INLINE bool is_symbolic() const {
#ifdef C10_MOBILE
    return false;
#else
    return !check_range(data_);
#endif
  }

  SymInt operator+(const SymInt& sci) const;
  SymInt operator-(const SymInt& sci) const;
  SymInt operator*(const SymInt& sci) const;
  SymInt operator/(const SymInt& sci) const;
  SymInt operator%(const SymInt& sci) const;
  bool operator==(const SymInt& sci) const;
  bool operator!=(const SymInt& p2) const;
  bool operator<(const SymInt& sci) const;
  bool operator<=(const SymInt& sci) const;
  bool operator>(const SymInt& sci) const;
  bool operator>=(const SymInt& sci) const;
  void operator*=(const SymInt& sci);
  void operator+=(const SymInt& sci);
  void operator/=(const SymInt& sci);

  SymInt min(const SymInt& sci) const;
  SymInt max(const SymInt& sci) const;

  SymInt operator*(int64_t sci) const;
  bool operator<(int64_t sci) const;
  bool operator==(int64_t sci) const;
  bool operator!=(int64_t sci) const;
  bool operator<=(int64_t sci) const;
  bool operator>(int64_t sci) const;
  bool operator>=(int64_t sci) const;

  operator SymFloat() const;

  int64_t as_int_unchecked() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!is_symbolic());
    return data_;
  }

  // Return whether the integer is representable as a SymInt.
  static bool check_range(int64_t i) {
    return i > MAX_UNREPRESENTABLE_INT;
  }

 private:
  // Constraints on the internal representation:
  //
  // - Should represent positive and small negative ints
  // - No conversion necessary for operations on ints
  // - Must represent valid 64-bit pointers
  // - Is symbolic test should be FAST (two arithmetic instructions is too
  // much).
  //   This code being a hotpath is based on Strobelight profiles of
  //   is_symbolic().  FB only: https://fburl.com/strobelight/5l50ncxd
  //   (you will need to change the time window).
  //
  // So, the scheme is to reserve large negative numbers (asssuming
  // two's complement):
  //
  // - 0b0.... means we are a positive int
  // - 0b11... means we are a small negative int
  // - 0b10... means we are are a pointer. This means that
  //           [-2^63, -2^62-1] are not representable as ints.
  //           We don't actually need all of this space as on x86_64
  //           as the top 16bits aren't used for anything
  static constexpr uint64_t MASK = 1ULL << 63 | 1ULL << 62 | 1ULL << 61;
  static constexpr uint64_t IS_SYM = 1ULL << 63 | 1ULL << 61;
  // We must manually translate the bit pattern test into a greater
  // than test because compiler doesn't figure it out:
  // https://godbolt.org/z/356aferaW
  static constexpr int64_t MAX_UNREPRESENTABLE_INT =
      -1LL & static_cast<int64_t>(~(1ULL << 62));
  int64_t data_;
};

/// Sum of a list of SymInt; accumulates into the c10::SymInt expression
template <
    typename C,
    typename std::enable_if<
        std::is_same<typename C::value_type, c10::SymInt>::value,
        int>::type = 0>
inline c10::SymInt multiply_integers(const C& container) {
  return std::accumulate(
      container.begin(),
      container.end(),
      c10::SymInt(1),
      [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
}

inline SymInt operator+(int64_t a, const SymInt& b) {
  return c10::SymInt(a) + b;
}
inline SymInt operator-(int64_t a, const SymInt& b) {
  return c10::SymInt(a) - b;
}
inline SymInt operator*(int64_t a, const SymInt& b) {
  return c10::SymInt(a) * b;
}
inline SymInt operator/(int64_t a, const SymInt& b) {
  return c10::SymInt(a) / b;
}
inline SymInt operator%(int64_t a, const SymInt& b) {
  return c10::SymInt(a) % b;
}
inline bool operator==(int64_t a, const SymInt& b) {
  return c10::SymInt(a) == b;
}
inline bool operator!=(int64_t a, const SymInt& b) {
  return c10::SymInt(a) != b;
}
inline bool operator<(int64_t a, const SymInt& b) {
  return c10::SymInt(a) < b;
}
inline bool operator<=(int64_t a, const SymInt& b) {
  return c10::SymInt(a) <= b;
}
inline bool operator>(int64_t a, const SymInt& b) {
  return c10::SymInt(a) > b;
}
inline bool operator>=(int64_t a, const SymInt& b) {
  return c10::SymInt(a) >= b;
}

C10_API std::ostream& operator<<(std::ostream& os, const SymInt& s);
C10_API SymInt operator-(const SymInt& s);
} // namespace c10
