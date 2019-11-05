#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeIndex.h>

using c10::util::get_type_index;

namespace {

static_assert(get_type_index<int>() == get_type_index<int>(), "");
static_assert(get_type_index<float>() == get_type_index<float>(), "");
static_assert(get_type_index<int>() != get_type_index<float>(), "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int(double, double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() != get_type_index<int(double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int (*)(double, double)>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() ==
        get_type_index<std::function<int(double, double)>>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() !=
        get_type_index<std::function<int(double)>>(),
    "");

static_assert(get_type_index<int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() == get_type_index<int&&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int>(), "");
static_assert(get_type_index<const int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() != get_type_index<int*>(), "");
static_assert(get_type_index<int*>() != get_type_index<int**>(), "");
static_assert(
    get_type_index<int(double&, double)>() !=
        get_type_index<int(double, double)>(),
    "");

struct Dummy final {};
struct Functor final {
  int64_t operator()(uint32_t, Dummy&&, const Dummy&) const;
};
static_assert(
    get_type_index<int64_t(uint32_t, Dummy&&, const Dummy&)>() ==
        get_type_index<
            c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");

} // namespace
