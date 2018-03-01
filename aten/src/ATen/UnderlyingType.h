template<typename T>
struct UnderlyingType;

#define SPECIALIZE_UNDERLYING_TYPE(underlying, type, _) \
template<> UnderlyingType<type> { using type = underlying; }
AT_FORALL_SCALAR_TYPES(SPECIALIZE_UNDERLYING_TYPE)
#undef SPECIALIZE_UNDERLYING_TYPE
