#pragma once

// Wrapper around our vendored fmt that avoids leaking macros to libraries that
// link in c10.

// Include this file to use fmt, don't include fmt directly.

#define FMT_HEADER_ONLY
#include <c10/util/fmt/format.h>
#include <c10/util/fmt/ostream.h>
#include <c10/util/fmt/ranges.h>
#include <c10/util/fmt/compile.h>
#include <c10/util/fmt/color.h>
#include <c10/util/fmt/chrono.h>
#undef FMT_HEADER_ONLY
