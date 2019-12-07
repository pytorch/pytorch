#include <c10/util/C++17.h>

using std::max;
using std::min;

static_assert(min(3, 5) == 3, "");
static_assert(min(5, 3) == 3, "");
static_assert(min(3, 3) == 3, "");
static_assert(min(3.0, 3.1) == 3.0, "");

static_assert(max(3, 5) == 5, "");
static_assert(max(5, 3) == 5, "");
static_assert(max(3, 3) == 3, "");
static_assert(max(3.0, 3.1) == 3.1, "");
