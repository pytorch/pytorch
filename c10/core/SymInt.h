#pragma once

#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <numeric>
#include <type_traits>

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
    if (is_heap_allocated()) {
      // Large negative number, heap allocate it
      promote_to_negative();
    }
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
    if (s.is_heap_allocated()) {
      *this = SymInt(s.toSymNode());
    } else {
      data_ = s.data_;
    }
  }
  SymInt(SymInt&& s) noexcept : data_(s.data_) {
    s.data_ = 0;
  }

  SymInt& operator=(const SymInt& s) {
    if (this != &s) {
      if (s.is_heap_allocated()) {
        *this = SymInt(s.toSymNode());
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
      if (s.is_heap_allocated())
        s.data_ = 0;
    };
    return *this;
  }

  SymNodeImpl* toSymNodeImplUnowned() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(is_heap_allocated());
    uint64_t unextended_bits = static_cast<uint64_t>(data_) & ~MASK;
    uint64_t sign_bit_mask = 1ULL << (62 - 1);
    // https://stackoverflow.com/questions/42534749/signed-extension-from-24-bit-to-32-bit-in-c
    uint64_t extended_bits = (unextended_bits ^ sign_bit_mask) - sign_bit_mask;
    return static_cast<SymNodeImpl*>(
        reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }

  void release_() {
    if (is_heap_allocated()) {
      SymNode::reclaim(toSymNodeImplUnowned()); // steal
    }
  }

  SymNodeImpl* release() && {
#ifndef C10_MOBILE
    TORCH_INTERNAL_ASSERT(is_heap_allocated());
    auto* r = toSymNodeImplUnowned();
    data_ = 0; // transfer ownership
    return r;
#else
    TORCH_INTERNAL_ASSERT(false);
#endif
  }

  // Only valid if is_heap_allocated()
  SymNode toSymNode() const;

  // Guaranteed to return a SymNode, wrapping using base if necessary
  SymNode wrap_node(const SymNode& base) const;

  ~SymInt() {
    release_();
  }

  // Require the int to be non-symbolic, and if it is symbolic raise an
  // error.  This is safe to use for C++ code that doesn't work for symbolic
  // shapes, and you don't have time to fix it immediately, as if we
  // try to trigger the path in C++ you'll appropriately get an error
  int64_t expect_int() const {
    if (auto r = maybe_as_int()) {
      return *r;
    }
    TORCH_CHECK(false, "expected int but got ", *this);
  }

  // Test if we have a hint for this int (e.g., guard_int would work).
  // Most of the time this is true; it is only false when you have
  // an unbacked SymInt.
  bool has_hint() const;

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

  // Insert a guard that this SymInt must be size-like, returning true if
  // the integer actually is >= 0.  Unlike manually performing a >= 0 test,
  // if the SymInt in question is an unbacked SymInt (or, potentially in the
  // future, if it contains unbacked SymInts), we will also treat the
  // unbacked SymInt as statically testing >= 2 (which will prevent us from
  // choking on, e.g., contiguity chekcs.)
  bool expect_size(const char* file, int64_t line) const;

  // Distinguish actual symbolic values from constants stored on the heap
  bool is_symbolic() const {
    return is_heap_allocated() &&
        !toSymNodeImplUnowned()->constant_int().has_value();
  }

  // N.B. It's important to keep this definition in the header
  // as we expect if checks to be folded for mobile builds
  // where `is_heap_allocated` is always false and optimize dead code paths
  C10_ALWAYS_INLINE bool is_heap_allocated() const {
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
  void operator*=(const SymInt& sci);
  void operator+=(const SymInt& sci);
  void operator/=(const SymInt& sci);

  SymInt clone() const;

  SymBool sym_eq(const SymInt&) const;
  SymBool sym_ne(const SymInt&) const;
  SymBool sym_lt(const SymInt&) const;
  SymBool sym_le(const SymInt&) const;
  SymBool sym_gt(const SymInt&) const;
  SymBool sym_ge(const SymInt&) const;

  bool operator==(const SymInt& o) const {
    return sym_eq(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator!=(const SymInt& o) const {
    return sym_ne(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<(const SymInt& o) const {
    return sym_lt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<=(const SymInt& o) const {
    return sym_le(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>(const SymInt& o) const {
    return sym_gt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>=(const SymInt& o) const {
    return sym_ge(o).guard_bool(__FILE__, __LINE__);
  }

  SymInt min(const SymInt& sci) const;
  SymInt max(const SymInt& sci) const;

  operator SymFloat() const;

  // Don't use this.  Prefer maybe_as_int instead
  int64_t as_int_unchecked() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!is_heap_allocated());
    return data_;
  }

  c10::optional<int64_t> maybe_as_int() const {
    if (!is_heap_allocated()) {
      return c10::make_optional(data_);
    }
    auto* node = toSymNodeImplUnowned();
    if (auto c = node->constant_int()) {
      return c;
    }
    return node->maybe_as_int();
  }

  // Return whether the integer is directly coercible to a SymInt
  // without requiring heap allocation.  You don't need to use this
  // to check if you can pass an integer to SymInt; this is guaranteed
  // to work (it just might heap allocate!)
  static bool check_range(int64_t i) {
    return i > MAX_UNREPRESENTABLE_INT;
  }

  // Return the min representable integer as a SymInt without
  // heap allocation.  For quantities that count bytes (or larger),
  // this is still much larger than you need, so you may consider
  // using this as a more efficient version of MIN_INT
  static constexpr int64_t min_representable_int() {
    return MAX_UNREPRESENTABLE_INT + 1;
  }

 private:
  void promote_to_negative();

  // Constraints on the internal representation:
  //
  // - Should represent positive and small negative ints
  // - No conversion necessary for operations on ints
  // - Must represent valid 64-bit pointers
  // - Is symbolic test should be FAST (two arithmetic instructions is too
  // much).
  //   This code being a hotpath is based on Strobelight profiles of
  //   is_heap_allocated().  FB only: https://fburl.com/strobelight/5l50ncxd
  //   (you will need to change the time window).
  //
  // So, the scheme is to reserve large negative numbers (assuming
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

template <
    typename Iter,
    typename = std::enable_if_t<std::is_same<
        typename std::iterator_traits<Iter>::value_type,
        c10::SymInt>::value>>
inline c10::SymInt multiply_integers(Iter begin, Iter end) {
  return std::accumulate(
      begin,
      end,
      c10::SymInt(1),
      [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
}

#define DECLARE_SYMINT_OP_INTONLY(scalar_t, RetTy)      \
  C10_API RetTy operator%(const SymInt& a, scalar_t b); \
  C10_API RetTy operator%(scalar_t a, const SymInt& b);

#define DECLARE_SYMINT_OP(scalar_t, RetTy)              \
  C10_API RetTy operator+(const SymInt& a, scalar_t b); \
  C10_API RetTy operator-(const SymInt& a, scalar_t b); \
  C10_API RetTy operator*(const SymInt& a, scalar_t b); \
  C10_API RetTy operator/(const SymInt& a, scalar_t b); \
  C10_API RetTy operator+(scalar_t a, const SymInt& b); \
  C10_API RetTy operator-(scalar_t a, const SymInt& b); \
  C10_API RetTy operator*(scalar_t a, const SymInt& b); \
  C10_API RetTy operator/(scalar_t a, const SymInt& b); \
  C10_API bool operator==(const SymInt& a, scalar_t b); \
  C10_API bool operator!=(const SymInt& a, scalar_t b); \
  C10_API bool operator<(const SymInt& a, scalar_t b);  \
  C10_API bool operator<=(const SymInt& a, scalar_t b); \
  C10_API bool operator>(const SymInt& a, scalar_t b);  \
  C10_API bool operator>=(const SymInt& a, scalar_t b); \
  C10_API bool operator==(scalar_t a, const SymInt& b); \
  C10_API bool operator!=(scalar_t a, const SymInt& b); \
  C10_API bool operator<(scalar_t a, const SymInt& b);  \
  C10_API bool operator<=(scalar_t a, const SymInt& b); \
  C10_API bool operator>(scalar_t a, const SymInt& b);  \
  C10_API bool operator>=(scalar_t a, const SymInt& b);

DECLARE_SYMINT_OP_INTONLY(int64_t, SymInt)
DECLARE_SYMINT_OP_INTONLY(int32_t, SymInt)
DECLARE_SYMINT_OP_INTONLY(uint64_t, SymInt)
DECLARE_SYMINT_OP_INTONLY(uint32_t, SymInt)
DECLARE_SYMINT_OP(int64_t, SymInt)
DECLARE_SYMINT_OP(int32_t, SymInt) // make sure constants work
DECLARE_SYMINT_OP(uint64_t, SymInt)
DECLARE_SYMINT_OP(uint32_t, SymInt)
DECLARE_SYMINT_OP(double, SymFloat)
DECLARE_SYMINT_OP(float, SymFloat) // just for completeness

// On OSX size_t is different than uint64_t so we have to
// define it separately
#if defined(__APPLE__)
DECLARE_SYMINT_OP_INTONLY(size_t, SymInt)
DECLARE_SYMINT_OP(size_t, SymInt)
#endif

#undef DECLARE_SYMINT_OP

C10_API std::ostream& operator<<(std::ostream& os, const SymInt& s);
C10_API SymInt operator-(const SymInt& s);
} // namespace c10
