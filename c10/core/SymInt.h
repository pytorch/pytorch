#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/SymIntNodeImpl.h>

#include <memory>

namespace c10 {

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
class C10_API SymInt {
 public:
  /*implicit*/ SymInt(int64_t d) : data_(d){};
  SymInt() = default;

  SymIntNodeImpl* toSymIntNodeImplUnowned() const {
    uint64_t unextended_bits = static_cast<uint64_t>(data_) & ~MASK;
    uint64_t sign_bit_mask = 1ULL << (62 - 1);
    // https://stackoverflow.com/questions/42534749/signed-extension-from-24-bit-to-32-bit-in-c
    uint64_t extended_bits = (unextended_bits ^ sign_bit_mask) - sign_bit_mask;
    return static_cast<SymIntNodeImpl*>(reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }

  ~SymInt() {
    if (is_symbolic()) {
      SymIntNode::reclaim(toSymIntNodeImplUnowned());  // steal
    }
  }

  int64_t expect_int() const {
    TORCH_CHECK(!is_symbolic());
    return data_;
  }

  bool is_symbolic() const {
    return (MASK & static_cast<uint64_t>(this->data_)) == IS_SYM;
  }

  SymInt operator+(SymInt sci) const;
  SymInt operator*(SymInt sci) const;
  bool operator==(SymInt sci) const;
  bool operator!=(SymInt p2) const;
  bool operator<(SymInt sci) const;
  void operator*=(SymInt sci);

  SymInt operator*(int64_t sci) const;
  bool operator<(int64_t sci) const;
  bool operator==(int64_t sci) const;
  bool operator!=(int64_t sci) const;

  SymIntNode toSymIntNodeImpl() const;
  static c10::SymInt toSymInt(SymIntNode sin);

  int64_t as_int_unchecked() const {
    return data_;
  }

  // This is needed for interoperability with IValue
  // TODO: this is wrong
  int64_t data() const {
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

C10_API std::ostream& operator<<(std::ostream& os, SymInt s);
} // namespace c10
