#pragma once

#include <c10/core/SymIntNodeImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <memory>
#include <numeric>

namespace c10 {

class SymFloat;

// `SymInt` is a C++ wrapper class around int64_t data_ which  and is used to
// represent concrete dimension values.
//
// `SymInt` is also a data type in Pytorch that can be used in function schemas
// to enable tracing.
//
// `SymInt` is introduced to enable tracing arithmetic
// operations on symbolic integers (e.g. sizes). Tracing symbolic sizes will
// allow LTC and AOTAutograd representing dynamic shapes in expression graphs
// faithfully without baking in concrete dimension values.
//
// To trace the operations, SymInt will overload arithmetic operators (e.g. +,
// -, *) and will provide overloads taking SymInt for commonly used math
// functions.
//
// SymInt will be extenteded to represent a union structure Union[int64_t,
// SymIntNodeImpl*] which will be implemented as a single packed int64_t field
// named data_.

#ifdef C10_MOBILE
#define SKIP_IS_SYMBOLIC_ON_MOBILE(_) \
  do {                                \
  } while (0)
#else
#define SKIP_IS_SYMBOLIC_ON_MOBILE(X) TORCH_CHECK(X)
#endif

class C10_API SymInt {
 public:
  enum Unchecked {
    UNCHECKED,
  };

  /*implicit*/ SymInt(int64_t d) : data_(d) {
    SKIP_IS_SYMBOLIC_ON_MOBILE(!is_symbolic());
  };
  SymInt() : data_(0) {}

  // unchecked c-tor accepting raw `data_`
  // One appropriate use for this is when you are constructing a symint
  // in a situation where you know it is non-negative (or, if it is negative,
  // the negative value is -1; i.e., not user controlled)
  SymInt(Unchecked, int64_t d) : data_(d) {}

  // TODO: these implementations are not optimal because they allocate a
  // temporary and then use the move constructor/assignment
  SymInt(const SymInt& s) : data_(0) {
    if (s.is_symbolic()) {
      *this = SymInt::toSymInt(s.toSymIntNodeImpl());
    } else {
      data_ = s.data_;
    }
  }
  SymInt(SymInt&& s) : data_(s.data_) {
    s.data_ = 0;
  }

  SymInt& operator=(const SymInt& s) {
    if (this != &s) {
      if (s.is_symbolic()) {
        *this = SymInt::toSymInt(s.toSymIntNodeImpl());
      } else {
        data_ = s.data_;
      }
    }
    return *this;
  }
  SymInt& operator=(SymInt&& s) {
    if (this != &s) {
      release_(); // release the current SymIntNode if any
      data_ = s.data_;
      if (s.is_symbolic())
        s.data_ = 0;
    };
    return *this;
  }

  SymInt clone() const {
#ifndef C10_MOBILE
    if (is_symbolic()) {
      return toSymIntNodeImplUnowned()->clone()->toSymInt();
    }
#else
    TORCH_INTERNAL_ASSERT(!is_symbolic());
#endif
    return *this;
  }

#ifndef C10_MOBILE
  SymIntNodeImpl* toSymIntNodeImplUnowned() const {
    uint64_t unextended_bits = static_cast<uint64_t>(data_) & ~MASK;
    uint64_t sign_bit_mask = 1ULL << (62 - 1);
    // https://stackoverflow.com/questions/42534749/signed-extension-from-24-bit-to-32-bit-in-c
    uint64_t extended_bits = (unextended_bits ^ sign_bit_mask) - sign_bit_mask;
    return static_cast<SymIntNodeImpl*>(
        reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }

  void release_() {
    if (is_symbolic()) {
      SymIntNode::reclaim(toSymIntNodeImplUnowned()); // steal
    }
  }

  SymIntNodeImpl* release() && {
    TORCH_INTERNAL_ASSERT(is_symbolic());
    auto* r = toSymIntNodeImplUnowned();
    data_ = 0; // transfer ownership
    return r;
  }
#else
  void release_() {}

  SymIntNodeImpl* release() && {
    TORCH_INTERNAL_ASSERT(false);
  }
#endif

  SymIntNode toSymIntNodeImpl() const;
  static c10::SymInt toSymInt(SymIntNode sin);

  ~SymInt() {
    release_();
  }

  // Require the int to be non-symbolic, and if it is symbolic raise an
  // error.  This is safe to use for C++ code that doesn't work for symbolic
  // shapes, and you don't have time to fix it immediately, as if we
  // try to trigger the path in C++ you'll appropriately get an error
  int64_t expect_int() const {
    SKIP_IS_SYMBOLIC_ON_MOBILE(!is_symbolic());
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
  // where `is_symbolic` is always false
  C10_ALWAYS_INLINE bool is_symbolic() const {
#ifdef C10_MOBILE
    return false;
#else
    return (MASK & static_cast<uint64_t>(this->data_)) == IS_SYM;
#endif
  }

  SymInt operator+(SymInt sci) const;
  SymInt operator-(SymInt sci) const;
  SymInt operator*(SymInt sci) const;
  SymInt operator/(SymInt sci) const;
  SymInt operator%(SymInt sci) const;
  bool operator==(SymInt sci) const;
  bool operator!=(SymInt p2) const;
  bool operator<(SymInt sci) const;
  bool operator<=(SymInt sci) const;
  bool operator>(SymInt sci) const;
  bool operator>=(SymInt sci) const;
  void operator*=(SymInt sci);
  void operator+=(SymInt sci);

  SymInt operator*(int64_t sci) const;
  bool operator<(int64_t sci) const;
  bool operator==(int64_t sci) const;
  bool operator!=(int64_t sci) const;
  bool operator<=(int64_t sci) const;
  bool operator>(int64_t sci) const;
  bool operator>=(int64_t sci) const;

  operator SymFloat() const;

  int64_t as_int_unchecked() const {
    return data_;
  }

  // Return whether the integer is representable as a SymInt.
  static bool check_range(int64_t i) {
    return i > MIN_INT;
  }

 private:
  // Constraints on the internal representation:
  // - Should represent positive and small negative ints
  // - No conversion necessary for operations on ints.
  // - Must represent valid 64-bit pointers
  //
  // So, the scheme is to reserve large negative numbers:
  // - 0b0.... means we are a positive int (following two's complement)
  // - 0b11... means we are a negative int (following two's complement)
  // - 0b10... means we are are a pointer. This means that
  //           [-2^63, -2^62-1] are not representable as ints.
  //           We don't actually need all of this space as on x86_64
  //           as the top 16bits aren't used for anything
  static constexpr uint64_t MASK = 1ULL << 63 | 1ULL << 62;
  static constexpr uint64_t IS_SYM = 1ULL << 63;
  // Since we use the top two bits to determine whether something is symbolic,
  // we cannot represent symbolic indices that are large enough to use those
  // bits. This will probably never happen.
  static constexpr uint64_t MAX_SYM_IDX = 1ULL << 62;
  // Since 0b10... is reserved for symbolic indices, any integers lower than
  // this value would collide with our representation.
  static constexpr int64_t MIN_INT = -1LL & static_cast<int64_t>(~(1ULL << 62));
  int64_t data_;
};

#undef SKIP_IS_SYMBOLIC_ON_MOBILE

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
      [](c10::SymInt a, c10::SymInt b) { return a * b; });
}

C10_API std::ostream& operator<<(std::ostream& os, SymInt s);
} // namespace c10
