// Type trait utils
template <typename Type, bool is_volatile>
struct MaybeVolatile;

template <typename Type>
struct MaybeVolatile<Type, true> {
  using type = volatile Type;
};

template <typename Type>
struct MaybeVolatile<Type, false> {
  using type = Type;
};

template <typename... Types>
struct TypeList {};

template <int idx, typename T, typename... Types>
struct TypeSelector {
  using type = typename TypeSelector<idx - 1, Types...>::type;
};

template <typename T, typename... Types>
struct TypeSelector<0, T, Types...> {
  using type = T;
};

template <typename T0, typename T1>
struct IsSameType {
  static constexpr bool value = false;
};

template <typename T0>
struct IsSameType<T0, T0> {
  static constexpr bool value = true;
};

template <typename T>
struct IsPointerType {
  static constexpr bool value = false;
};

template <typename T>
struct IsPointerType<T*> {
  static constexpr bool value = true;
};
