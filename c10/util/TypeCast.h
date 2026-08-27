#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <c10/util/overflows.h>
#include <c10/util/safe_conv.h>

#include <torch/headeronly/util/TypeCast.h>

namespace c10 {
[[noreturn]] C10_API void report_overflow(const char* name);
} // namespace c10
