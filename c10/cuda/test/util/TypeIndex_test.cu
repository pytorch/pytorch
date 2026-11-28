#include <c10/util/TypeIndex.h>

#include <string_view>

// nvcc's MSVC-mode front end formats __FUNCSIG__ differently.
// TypeIndex_test.cpp covers MSVC.
#if !defined(_MSC_VER)
namespace {
using c10::util::get_fully_qualified_type_name;

template <char C>
struct CharTag {};

static_assert(get_fully_qualified_type_name<int>() == "int");
static_assert(get_fully_qualified_type_name<double>() == "double");
static_assert(
    get_fully_qualified_type_name<CharTag<';'>>().ends_with("CharTag<';'>"));
} // namespace
#endif
