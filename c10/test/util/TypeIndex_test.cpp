#include <c10/util/TypeIndex.h>

using c10::util::get_type_index;

namespace {

static_assert(get_type_index<int>() == get_type_index<int>(), "");
static_assert(get_type_index<float>() == get_type_index<float>(), "");
static_assert(get_type_index<int>() != get_type_index<float>(), "");
static_assert(get_type_index<int(double, double)>() == get_type_index<int(double, double)>(), "");
static_assert(get_type_index<int(double, double)>() != get_type_index<int(double)>(), "");
static_assert(get_type_index<int(double, double)>() == get_type_index<int(*)(double, double)>(), "");
static_assert(get_type_index<std::function<int(double, double)>>() == get_type_index<std::function<int(double, double)>>(), "");
static_assert(get_type_index<std::function<int(double, double)>>() != get_type_index<std::function<int(double)>>(), "");

static_assert(get_type_index<int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() == get_type_index<int&&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int>(), "");
static_assert(get_type_index<const int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() != get_type_index<int*>(), "");
static_assert(get_type_index<int*>() != get_type_index<int**>(), "");
static_assert(get_type_index<int(double&, double)>() != get_type_index<int(double, double)>(), "");

}
