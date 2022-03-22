#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

namespace c10 {

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
