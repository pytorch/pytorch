#include <c10/util/Optional.h>

#include <type_traits>

static_assert(C10_IS_TRIVIALLY_COPYABLE(c10::optional<int>), "c10::optional<int> should be trivially copyable");
static_assert(C10_IS_TRIVIALLY_COPYABLE(c10::optional<bool>), "c10::optional<bool> should be trivially copyable");
