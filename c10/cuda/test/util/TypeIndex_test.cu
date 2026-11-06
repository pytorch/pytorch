#include <c10/util/TypeIndex.h>

#include <string_view>

namespace {
using c10::util::get_fully_qualified_type_name;

template <char C>
struct CharTag {};

static_assert(get_fully_qualified_type_name<int>() == "int");
static_assert(get_fully_qualified_type_name<double>() == "double");
static_assert(get_fully_qualified_type_name<CharTag<';'>>().ends_with(">"));
} // namespace
