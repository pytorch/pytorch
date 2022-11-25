// std::tuple-like type
template <typename... Types>
struct Tuple;

#define TUPLE_INCREMENT_PTR(idx)                                        \
  do {                                                                  \
    static_assert(                                                      \
        IsPointerType<T##idx>::value, "Invalid for non-pointer types"); \
    val##idx += offset;                                                 \
  } while (0)

template <typename T0>
struct Tuple<T0> {
  T0 val0;

  Tuple() = default;

  __device__ Tuple(T0 _val0) : val0(_val0) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
  }
};

template <typename T0, typename T1>
struct Tuple<T0, T1> {
  T0 val0;
  T1 val1;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1) : val0(_val0), val1(_val1) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
  }
};

template <typename T0, typename T1, typename T2>
struct Tuple<T0, T1, T2> {
  T0 val0;
  T1 val1;
  T2 val2;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2)
      : val0(_val0), val1(_val1), val2(_val2) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
  }
};

template <typename T0, typename T1, typename T2, typename T3>
struct Tuple<T0, T1, T2, T3> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2, T3 _val3)
      : val0(_val0), val1(_val1), val2(_val2), val3(_val3) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
  }
};

template <typename T0, typename T1, typename T2, typename T3, typename T4>
struct Tuple<T0, T1, T2, T3, T4> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2, T3 _val3, T4 _val4)
      : val0(_val0), val1(_val1), val2(_val2), val3(_val3), val4(_val4) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5>
struct Tuple<T0, T1, T2, T3, T4, T5> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2, T3 _val3, T4 _val4, T5 _val5)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6>
struct Tuple<T0, T1, T2, T3, T4, T5, T6> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;
  T6 val6;

  Tuple() = default;

  __device__ Tuple(
      T0 _val0,
      T1 _val1,
      T2 _val2,
      T3 _val3,
      T4 _val4,
      T5 _val5,
      T6 _val6)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5),
        val6(_val6) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
    TUPLE_INCREMENT_PTR(6);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7>
struct Tuple<T0, T1, T2, T3, T4, T5, T6, T7> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;
  T6 val6;
  T7 val7;

  Tuple() = default;

  __device__ Tuple(
      T0 _val0,
      T1 _val1,
      T2 _val2,
      T3 _val3,
      T4 _val4,
      T5 _val5,
      T6 _val6,
      T7 _val7)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5),
        val6(_val6),
        val7(_val7) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
    TUPLE_INCREMENT_PTR(6);
    TUPLE_INCREMENT_PTR(7);
  }
};

#undef TUPLE_INCREMENT_PTR

// Accessor for Tuple
template <int idx>
struct get;

#define DEFINE_TUPLE_GET(idx)                              \
  template <>                                              \
  struct get<idx> {                                        \
    template <typename Tuple>                              \
    __device__ auto& operator()(Tuple& vals) {             \
      return vals.val##idx;                                \
    }                                                      \
    template <typename Tuple>                              \
    __device__ const auto& operator()(const Tuple& vals) { \
      return vals.val##idx;                                \
    }                                                      \
  };

DEFINE_TUPLE_GET(0);
DEFINE_TUPLE_GET(1);
DEFINE_TUPLE_GET(2);
DEFINE_TUPLE_GET(3);
DEFINE_TUPLE_GET(4);
DEFINE_TUPLE_GET(5);
DEFINE_TUPLE_GET(6);
DEFINE_TUPLE_GET(7);
#undef DEFINE_TUPLE_GET

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset = 0);

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset = 0);

template <typename DstType>
__inline__ __device__ static void setTuple(
    DstType& dst,
    typename DstType::template ValType<0> src);

template <typename... Types>
class LocalTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;

  template <int idx>
  using ValType = typename TypeSelector<idx, Types...>::type;

  LocalTuple() = default;

  __device__ explicit LocalTuple(Types... args) : vals_(args...) {}

  __device__ LocalTuple(const LocalTuple& other) : vals_(other.vals_) {}

  template <template <typename...> typename TupleType>
  __device__ LocalTuple(const TupleType<Types...>& other) {
    copyTuple(*this, other);
  }

  __device__ LocalTuple& operator=(const LocalTuple<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <template <typename...> typename TupleType>
  __device__ LocalTuple& operator=(const TupleType<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <int val_idx>
  __device__ auto& val(nvfuser_index_t ptr_offset = 0) {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

  template <int val_idx>
  __device__ const auto& val(nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

 private:
  Tuple<Types...> vals_;
};

template <bool is_volatile, typename... Types>
class PtrTupleBase {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;
  template <int idx>
  using ValType = typename TypeSelector<idx, Types...>::type;
  template <int val_idx>
  using TypeIMaybeVolatile = typename MaybeVolatile<
      typename TypeSelector<val_idx, Types...>::type,
      is_volatile>::type;

  __device__ PtrTupleBase(Types*... args) : vals_(args...) {}

  __device__ PtrTupleBase(const PtrTupleBase& other) : vals_(other.vals_) {}

  // Note: this is a deep copy
  __device__ PtrTupleBase& operator=(
      const PtrTupleBase<is_volatile, Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <template <typename...> typename TupleType>
  __device__ PtrTupleBase& operator=(const TupleType<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <int val_idx>
  __device__ TypeIMaybeVolatile<val_idx>& val(nvfuser_index_t ptr_offset = 0) {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return ((TypeIMaybeVolatile<val_idx>*)get<val_idx>()(vals_))[ptr_offset];
  }

  template <int val_idx>
  __device__ const TypeIMaybeVolatile<val_idx>& val(
      nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return ((TypeIMaybeVolatile<val_idx>*)get<val_idx>()(vals_))[ptr_offset];
  }

  __device__ void operator+=(nvfuser_index_t ptr_offset) {
    vals_ += ptr_offset;
  }

 private:
  Tuple<Types*...> vals_;
};

template <typename... Types>
class RefTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;
  template <int idx>
  using ValType = typename TypeSelector<idx, Types...>::type;

  __device__ RefTuple(Types&... args) : vals_(args...) {}

  __device__ RefTuple(const RefTuple& other) : vals_(other.vals_) {}

  template <template <typename...> typename TupleType>
  __device__ RefTuple(const TupleType<Types...>& other) {
    copyTuple(*this, other);
  }

  __device__ RefTuple& operator=(const RefTuple<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <template <typename...> typename TupleType>
  __device__ RefTuple& operator=(const TupleType<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <int val_idx>
  __device__ auto& val(nvfuser_index_t ptr_offset = 0) {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

  template <int val_idx>
  __device__ const auto& val(nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

 private:
  Tuple<Types&...> vals_;
};

template <typename DstType, typename SrcType, int num_vals>
struct TupleCopy {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset) {
    static_assert(
        IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::
            value,
        "Invalid value types");
    TupleCopy<DstType, SrcType, num_vals - 1>::copy(
        dst, dst_offset, src, src_offset);
    dst.val<num_vals - 1>(dst_offset) = src.val<num_vals - 1>(src_offset);
  }
};

template <typename DstType, typename SrcType>
struct TupleCopy<DstType, SrcType, 0> {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset) {}
};

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset) {
  static_assert(
      IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::value,
      "Invalid value types");
  TupleCopy<DstType, SrcType, DstType::num_vals>::copy(
      dst, dst_offset, src, src_offset);
};

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset) {
  copyTuple<DstType, SrcType>(dst, 0, src, src_offset);
};

template <typename DstType, int num_vals>
struct TupleSet {
  __inline__ __device__ static void set(
      DstType& dst,
      nvfuser_index_t dst_offset,
      typename DstType::template ValType<0> src) {
    static_assert(
        IsSameType<
            typename DstType::template ValType<num_vals - 1>,
            typename DstType::template ValType<0>>::value,
        "Invalid value types");
    TupleSet<DstType, num_vals - 1>::set(dst, dst_offset, src);
    dst.val<num_vals - 1>(dst_offset) = src;
  }
};

template <typename DstType>
struct TupleSet<DstType, 0> {
  __inline__ __device__ static void set(
      DstType& dst,
      nvfuser_index_t dst_offset,
      typename DstType::template ValType<0> src) {}
};

template <typename DstType>
__inline__ __device__ static void setTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    typename DstType::template ValType<0> src) {
  TupleSet<DstType, DstType::num_vals>::set(dst, dst_offset, src);
};

template <typename DstType>
__inline__ __device__ static void setTuple(
    DstType& dst,
    typename DstType::template ValType<0> src) {
  setTuple(dst, 0, src);
};

template <typename DstType, typename SrcType, typename PredType, int num_vals>
struct PredicatedTupleCopy {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset,
      const PredType& pred) {
    static_assert(
        IsSameType<typename PredType::template ValType<num_vals - 1>, bool>::
            value,
        "Invalid predicate type");
    PredicatedTupleCopy<DstType, SrcType, PredType, num_vals - 1>::copy(
        dst, dst_offset, src, src_offset, pred);
    if (pred.val<num_vals - 1>(0)) {
      dst.val<num_vals - 1>(dst_offset) = src.val<num_vals - 1>(src_offset);
    }
  }
};

template <typename DstType, typename SrcType, typename PredType>
struct PredicatedTupleCopy<DstType, SrcType, PredType, 0> {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset,
      const PredType& pred) {}
};

template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyTupleIf(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset,
    const PredType& pred) {
  static_assert(
      IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::value,
      "Invalid value types");
  static_assert(
      PredType::num_vals == DstType::num_vals, "Invalid predicate type");
  PredicatedTupleCopy<DstType, SrcType, PredType, DstType::num_vals>::copy(
      dst, dst_offset, src, src_offset, pred);
};

template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyTupleIf(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset,
    const PredType& pred) {
  copyTupleIf(dst, 0, src, src_offset, pred);
};

template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyTupleIf(
    DstType& dst,
    const SrcType& src,
    const PredType& pred) {
  copyTupleIf(dst, 0, src, 0, pred);
};

// Can a generic const and non-const RefTupe be defined?
template <typename... Types>
class ConstRefTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;

  __device__ ConstRefTuple(const Types&... args) : vals_(args...) {}

  __device__ ConstRefTuple(const ConstRefTuple& other) : vals_(other.vals_) {}

  template <template <typename...> typename TupleType>
  __device__ ConstRefTuple(const TupleType<Types...>& other) {
    copyTuple(*this, other);
  }

  template <int val_idx>
  __device__ const auto& val(nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

 private:
  Tuple<const Types&...> vals_;
};

template <typename... Types>
using PtrTuple = PtrTupleBase<false, Types...>;

template <typename... Types>
using VolatilePtrTuple = PtrTupleBase<true, Types...>;

// Define a LocalTuple of NumVals values of type Type
template <int NumVals, typename Type>
struct MakeLocalTuple;

template <typename Type>
struct MakeLocalTuple<1, Type> {
  using type = LocalTuple<Type>;
};

template <typename Type>
struct MakeLocalTuple<2, Type> {
  using type = LocalTuple<Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<3, Type> {
  using type = LocalTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<4, Type> {
  using type = LocalTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<5, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<6, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<7, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<8, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <int NumVals, typename Type>
struct MakeRefTuple;

template <typename Type>
struct MakeRefTuple<1, Type> {
  using type = RefTuple<Type>;
};

template <typename Type>
struct MakeRefTuple<2, Type> {
  using type = RefTuple<Type, Type>;
};

template <typename Type>
struct MakeRefTuple<3, Type> {
  using type = RefTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<4, Type> {
  using type = RefTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<5, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<6, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<7, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<8, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <int NumVals, typename Type>
struct MakeConstRefTuple;

template <typename Type>
struct MakeConstRefTuple<1, Type> {
  using type = ConstRefTuple<Type>;
};

template <typename Type>
struct MakeConstRefTuple<2, Type> {
  using type = ConstRefTuple<Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<3, Type> {
  using type = ConstRefTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<4, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<5, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<6, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<7, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<8, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <int NumVals, typename Type>
struct MakeVolatilePtrTuple;

template <typename Type>
struct MakeVolatilePtrTuple<1, Type> {
  using type = VolatilePtrTuple<Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<2, Type> {
  using type = VolatilePtrTuple<Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<3, Type> {
  using type = VolatilePtrTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<4, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<5, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<6, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<7, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<8, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

// Utility definitions. Currently only used with LocalTuple

template <int idx, typename BinaryFunc, typename... DataTypes>
struct TupleBinaryOp {
  static __inline__ __device__ void apply(
      BinaryFunc func,
      const LocalTuple<DataTypes...>& lhs,
      const LocalTuple<DataTypes...>& rhs,
      LocalTuple<DataTypes...>& result) {
    TupleBinaryOp<idx - 1, BinaryFunc, DataTypes...>::apply(
        func, lhs, rhs, result);
    result.val<idx - 1>(0) = func(lhs.val<idx - 1>(0), rhs.val<idx - 1>(0));
  }
};

template <typename BinaryFunc, typename... DataTypes>
struct TupleBinaryOp<0, BinaryFunc, DataTypes...> {
  static __inline__ __device__ void apply(
      BinaryFunc func,
      const LocalTuple<DataTypes...>& lhs,
      const LocalTuple<DataTypes...>& rhs,
      LocalTuple<DataTypes...>& result) {}
};

template <typename BinaryFunc, typename... DataTypes>
__inline__ __device__ LocalTuple<DataTypes...> apply(
    BinaryFunc func,
    const LocalTuple<DataTypes...>& lhs,
    const LocalTuple<DataTypes...>& rhs) {
  LocalTuple<DataTypes...> result = lhs;
  TupleBinaryOp<sizeof...(DataTypes), BinaryFunc, DataTypes...>::apply(
      func, result, rhs, result);
  return result;
}

template <typename... BoolTypes>
__inline__ __device__ LocalTuple<BoolTypes...> operator&&(
    const LocalTuple<BoolTypes...>& lhs,
    const LocalTuple<BoolTypes...>& rhs) {
  return apply([](bool x, bool y) { return x && y; }, lhs, rhs);
}

template <typename... BoolTypes>
__inline__ __device__ LocalTuple<BoolTypes...> operator&&(
    bool lhs,
    const LocalTuple<BoolTypes...>& rhs) {
  LocalTuple<BoolTypes...> lhs_tuple;
  setTuple(lhs_tuple, lhs);
  return lhs_tuple && rhs;
}

template <typename... BoolTypes>
__inline__ __device__ LocalTuple<BoolTypes...> operator&&(
    const LocalTuple<BoolTypes...>& lhs,
    bool rhs) {
  LocalTuple<BoolTypes...> rhs_tuple;
  setTuple(rhs_tuple, rhs);
  return lhs && rhs_tuple;
}
