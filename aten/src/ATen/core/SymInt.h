#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

namespace c10 {

// SymInt is a C++ wrapper class used by LTC and AOTAutograd for tracing
// arithmetic operations on symbolic integers (e.g. sizes).
// 
// `SymInt` is also a data type in Pytorch that can be used in function schemas
// to enable tracing.
//
// To trace the operations, SymInt overloads arithmetic operators (e.g. +, -, *)
// and provides overloads taking SymInt for commonly used math functions.
//
// SymInt is semantically a union structure Union[int64_t, SymbolicIntNode*]
// implemented as a single packed int64_t field named data_.
//
// data_ can be either a plain int64_t or (1 << 63 | `index`). `index` points to
// SymbolicIntNode* that is responsible for constructing an IR node for
// a traced operation to represent it in LTC or Fx graphs.
class TORCH_API SymInt {
    public:

    SymInt(int64_t d):
    data_(d) {};

    int64_t data_;

    int64_t expect_int() {
        // we are dealing with concrete ints only for now
        return data_;
    }

    bool is_symbolic() {
        return false;
    }

    bool operator==(const SymInt& p2)
    {
        TORCH_INTERNAL_ASSERT("NYI");
        return false;
    }

    SymInt operator+(SymInt sci) {
        return data_ + sci.data_;
    }
};

TORCH_API std::ostream& operator<<(std::ostream& os, SymInt s);
}
