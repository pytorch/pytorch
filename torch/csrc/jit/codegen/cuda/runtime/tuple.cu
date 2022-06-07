// std::tuple-like type
template <typename... Types>
struct Tuple;

template <typename T0>
struct Tuple<T0> {
  T0 val0;

  __device__ Tuple(T0 _val0) : val0(_val0) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    static_assert(IsPointerType<T0>::value, "Invalid for non-pointer types");
    val0 += offset;
  }
};

template <typename T0, typename T1>
struct Tuple<T0, T1> {
  T0 val0;
  T1 val1;

  __device__ Tuple(T0 _val0, T1 _val1) : val0(_val0), val1(_val1) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    static_assert(IsPointerType<T0>::value, "Invalid for non-pointer types");
    static_assert(IsPointerType<T1>::value, "Invalid for non-pointer types");
    val0 += offset;
    val1 += offset;
  }
};

template <typename T0, typename T1, typename T2>
struct Tuple<T0, T1, T2> {
  T0 val0;
  T1 val1;
  T2 val2;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2)
      : val0(_val0), val1(_val1), val2(_val2) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    static_assert(IsPointerType<T0>::value, "Invalid for non-pointer types");
    static_assert(IsPointerType<T1>::value, "Invalid for non-pointer types");
    static_assert(IsPointerType<T2>::value, "Invalid for non-pointer types");
    val0 += offset;
    val1 += offset;
    val2 += offset;
  }
};

// Accessor for Tuple
template <int idx>
struct get;

template <>
struct get<0> {
  template <typename Tuple>
  __device__ auto& operator()(Tuple& vals) {
    return vals.val0;
  }
  template <typename Tuple>
  __device__ const auto& operator()(const Tuple& vals) {
    return vals.val0;
  }
};

template <>
struct get<1> {
  template <typename Tuple>
  __device__ auto& operator()(Tuple& vals) {
    return vals.val1;
  }
  template <typename Tuple>
  __device__ const auto& operator()(const Tuple& vals) {
    return vals.val1;
  }
};

template <>
struct get<2> {
  template <typename Tuple>
  __device__ auto& operator()(Tuple& vals) {
    return vals.val2;
  }
  template <typename Tuple>
  __device__ const auto& operator()(const Tuple& vals) {
    return vals.val2;
  }
};

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

template <typename... Types>
class LocalTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;

  __device__ LocalTuple(Types... args) : vals_(args...) {}

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

template <typename DstType, typename SrcType>
struct TupleCopy<DstType, SrcType, 1> {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset) {
    static_assert(
        IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::
            value,
        "Invalid value types");
    dst.val<0>(dst_offset) = src.val<0>(src_offset);
  }
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
  copyTuple(dst, 0, src, src_offset);
};

template <typename... Types>
using PtrTuple = PtrTupleBase<false, Types...>;

template <typename... Types>
using VolatilePtrTuple = PtrTupleBase<true, Types...>;
