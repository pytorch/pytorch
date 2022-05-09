#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <type_traits>


namespace c10 {

class SymbolicIntNode;

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
// To trace the operations, SymInt will overload arithmetic operators (e.g. +, -, *)
// and will provide overloads taking SymInt for commonly used math functions.
//
// SymInt will be extenteded to represent a union structure Union[int64_t, SymbolicIntNode*]
// which will be implemented as a single packed int64_t field named data_.
//
// data_ can be either a plain int64_t or (1 << 63 | `index`). `index` points to
// SymbolicIntNode* that will be responsible for constructing an IR node for
// a traced operation to represent it in LTC or Fx graphs.
class TORCH_API SymInt {
    public:
        SymInt() = default;

        SymInt(int64_t d):
        data_(d) {};

        int64_t expect_int() const {
            TORCH_CHECK(!is_symbolic());
            return data_;
        }

        int64_t unchecked_expect_int() const {
            return data_;
        }

        bool is_symbolic() const {
            return SYM_TAG_MASK & static_cast<uint64_t>(this->data_);
        }

        bool operator==(const SymInt& p2) const
        {
            return data_ == p2.data_;
        }

        SymInt operator+(SymInt sci) const {
            TORCH_CHECK(!this->is_symbolic() && !sci.is_symbolic(), "Symbolic Add isn't supported yet");
            return data_ + sci.data_;
        }

        SymInt operator*(SymInt other) const {
          if (C10_UNLIKELY(is_symbolic() || other.is_symbolic())) {
            return slow_mul(other);
          } else {
            return data_ * other.data_;
          }
        }

        SymInt max(SymInt other) const {
          if (C10_UNLIKELY(is_symbolic() || other.is_symbolic())) {
            return slow_max(other);
          } else {
            return std::max(data_, other.data_);
          }
        }

        std::shared_ptr<SymbolicIntNode> toSymbolicIntNode();
        static c10::SymInt toSymInt(std::shared_ptr<SymbolicIntNode> sin);

        // This is needed for interoperability with IValue
        // TODO: this is very innocuously named, doesn't look very good
        int64_t data() const {
            return data_;
        }

    private:
        const static int64_t SYM_TAG_MASK = 1LL << 63;
        int64_t data_;

        SymInt slow_mul(SymInt other) const;
        SymInt slow_max(SymInt other) const;
};

static_assert(std::is_trivial<SymInt>::value, "SymInt must be trivial");

TORCH_API std::ostream& operator<<(std::ostream& os, SymInt s);
}
