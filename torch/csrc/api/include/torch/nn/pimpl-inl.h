// This class exists  only to do SFINAE on abstract types `T` that are really
// `ModuleHolder<ModuleType>`, because there's no good way to say that `T` is a
// `ModuleHolder` over some unknown type `ModuleType`. With this, you can do
// `enable_if_t<is_base_of_v<ModuleHolderIndicator, T>>`.
struct ModuleHolderIndicator {};

// A type trait that is true for types that are `ModuleHolder`s.
template <typename T>
using is_module_holder = std::is_base_of<ModuleHolderIndicator, decay_t<T>>;

template <typename T>
using disable_if_module_holder_t = disable_if_t<is_module_holder<T>::value>;

// A collection of templates that answer the question whether a type `T` is a
// `ModuleHolder`, and if so whether its contained type is of type `C`. This is
// tricky because it is hard to short circuit in template metaprogramming. A
// naive and incorrect solution to this problem would be something like
// `disable_if<is_module_holder<T>::value && typename T::ContainedType == C>`.
// This would disable all types that are not `ModuleHolder`s, because even
// though the `is_module_holder<T>::value` may be `false` for such types the
// `T::ContainedType` access would be ill-formed and thus fail the whole
// expression by the rules of SFINAE. Instead we have to use template
// specialization to statically branch on the first condition
// (`is_module_holder<T>`) and are only then allowed to query
// `T::ContainedType` in the branch for which the condition was true.

// Base template.
template <bool is_module_holder_value, typename T, typename C>
struct is_module_holder_of_impl;

// False branch. `T` is not a `ModuleHolder` and thus not a `ModuleHolder` with
// contained type `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<false, T, C> : std::false_type {};

// True branch. `T` is a `ModuleHolder` and thus we can legit access its
// `ContainedType` and compare it against `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<true, T, C>
    : std::is_same<typename T::ContainedType, C> {};

// Helper template.
template <typename T, typename C>
struct is_module_holder_of : is_module_holder_of_impl<
                                 is_module_holder<T>::value,
                                 decay_t<T>,
                                 decay_t<C>> {};

// A collection of templates that allow deducing the return type of the
// `forward()` method, but only if a module actually has a `forward()` method,
// and otherwise deduces to the type `void`.

template <bool has_forward_value, typename C, typename... Args>
struct return_type_of_forward_impl;

template <typename C, typename... Args>
struct return_type_of_forward_impl<true, C, Args...> {
  using type = decltype(::std::declval<C>().forward(::std::declval<Args>()...));
};

template <typename C, typename... Args>
struct return_type_of_forward_impl<false, C, Args...> {
  using type = void;
};

template <typename C, typename... Args>
using return_type_of_forward = return_type_of_forward_impl<
    torch::detail::has_forward<C>::value,
    C,
    Args...>;

template <typename C, typename... Args>
using return_type_of_forward_t =
    typename return_type_of_forward<C, Args...>::type;
