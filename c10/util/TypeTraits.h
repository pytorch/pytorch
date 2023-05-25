#pragma once

#include <c10/util/C++17.h>
#include <functional>

namespace c10 {
namespace guts {

/**
 * is_equality_comparable<T> is true_type iff the equality operator is defined
 * for T.
 */
template <class T, class Enable = void>
struct is_equality_comparable : std::false_type {};
template <class T>
struct is_equality_comparable<
    T,
    void_t<decltype(std::declval<T&>() == std::declval<T&>())>>
    : std::true_type {};
template <class T>
using is_equality_comparable_t = typename is_equality_comparable<T>::type;

/**
 * is_hashable<T> is true_type iff std::hash is defined for T
 */
template <class T, class Enable = void>
struct is_hashable : std::false_type {};
template <class T>
struct is_hashable<T, void_t<decltype(std::hash<T>()(std::declval<T&>()))>>
    : std::true_type {};
template <class T>
using is_hashable_t = typename is_hashable<T>::type;

/**
 * is_function_type<T> is true_type iff T is a plain function type (i.e.
 * "Result(Args...)")
 */
template <class T>
struct is_function_type : std::false_type {};
template <class Result, class... Args>
struct is_function_type<Result(Args...)> : std::true_type {};
template <class T>
using is_function_type_t = typename is_function_type<T>::type;

/**
 * is_instantiation_of<T, I> is true_type iff I is a template instantiation of T
 * (e.g. vector<int> is an instantiation of vector) Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template <template <class...> class Template, class T>
using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;

namespace detail {
/**
 * strip_class: helper to remove the class type from pointers to `operator()`.
 */

template <typename T>
struct strip_class {};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...)> {
  using type = Result(Args...);
};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...) const> {
  using type = Result(Args...);
};
template <typename T>
using strip_class_t = typename strip_class<T>::type;
} // namespace detail

/**
 * Evaluates to true_type, iff the given class is a Functor
 * (i.e. has a call operator with some set of arguments)
 */

template <class Functor, class Enable = void>
struct is_functor : std::false_type {};
template <class Functor>
struct is_functor<
    Functor,
    std::enable_if_t<is_function_type<
        detail::strip_class_t<decltype(&Functor::operator())>>::value>>
    : std::true_type {};

/**
 * lambda_is_stateless<T> is true iff the lambda type T is stateless
 * (i.e. does not have a closure).
 * Example:
 *  auto stateless_lambda = [] (int a) {return a;};
 *  lambda_is_stateless<decltype(stateless_lambda)> // true
 *  auto stateful_lambda = [&] (int a) {return a;};
 *  lambda_is_stateless<decltype(stateful_lambda)> // false
 */
namespace detail {
template <class LambdaType, class FuncType>
struct is_stateless_lambda__ final {
  static_assert(
      !std::is_same<LambdaType, LambdaType>::value,
      "Base case shouldn't be hit");
};
// implementation idea: According to the C++ standard, stateless lambdas are
// convertible to function pointers
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...) const>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...)>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};

// case where LambdaType is not even a functor
template <class LambdaType, class Enable = void>
struct is_stateless_lambda_ final : std::false_type {};
// case where LambdaType is a functor
template <class LambdaType>
struct is_stateless_lambda_<
    LambdaType,
    std::enable_if_t<is_functor<LambdaType>::value>>
    : is_stateless_lambda__<LambdaType, decltype(&LambdaType::operator())> {};
} // namespace detail
template <class T>
using is_stateless_lambda = detail::is_stateless_lambda_<std::decay_t<T>>;

/**
 * is_type_condition<C> is true_type iff C<...> is a type trait representing a
 * condition (i.e. has a constexpr static bool ::value member) Example:
 *   is_type_condition<std::is_reference>  // true
 */
template <template <class> class C, class Enable = void>
struct is_type_condition : std::false_type {};
template <template <class> class C>
struct is_type_condition<
    C,
    std::enable_if_t<
        std::is_same<bool, std::remove_cv_t<decltype(C<int>::value)>>::value>>
    : std::true_type {};

/**
 * is_fundamental<T> is true_type iff the lambda type T is a fundamental type
 * (that is, arithmetic type, void, or nullptr_t). Example: is_fundamental<int>
 * // true We define it here to resolve a MSVC bug. See
 * https://github.com/pytorch/pytorch/issues/30932 for details.
 */
template <class T>
struct is_fundamental : std::is_fundamental<T> {};
} // namespace guts
} // namespace c10
