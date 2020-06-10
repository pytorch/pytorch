#include <c10/core/CompileTimeFunctionPointer.h>

namespace test_is_compile_time_function_pointer {
static_assert(!c10::is_compile_time_function_pointer<void()>::value, "");

void dummy() {}
static_assert(c10::is_compile_time_function_pointer<decltype(TORCH_FN(dummy))>::value, "");
}

namespace test_access_through_type {
    void dummy() {}
    using dummy_ptr = decltype(TORCH_FN(dummy));
    static_assert(c10::is_compile_time_function_pointer<dummy_ptr>::value, "");
    static_assert(dummy_ptr::func_ptr() == &dummy, "");
    static_assert(std::is_same<void(), dummy_ptr::FuncType>::value, "");
}

namespace test_access_through_value {
    void dummy() {}
    constexpr auto dummy_ptr = TORCH_FN(dummy);
    static_assert(dummy_ptr.func_ptr() == &dummy, "");
    static_assert(std::is_same<void(), decltype(dummy_ptr)::FuncType>::value, "");
}
