#include <c10/util/Optional.h>

#include <type_traits>

static_assert(std::is_trivially_copyable<c10::optional<int>>::value, "c10::optional<int> should be trivially copyable");
static_assert(std::is_trivially_copyable<c10::optional<bool>>::value, "c10::optional<bool> should be trivially copyable");
