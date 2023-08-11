namespace fused_reduction {

// Tuple of Welford avg, var and N parameters.
//
// Template parameters:
// - DataTypeT: Type of avg and var
// - IndexTypeT: Type of N
// - MakeTuple: Template template parameter to define Tuple types
// (e.g., MakeLocalTuple>
template <
    int NumVals,
    typename DataTypeT,
    typename IndexTypeT,
    template <int, typename>
    typename MakeTuple>
struct WelfordTripletTuple {
  static constexpr int num_vals = NumVals;
  using DataType = DataTypeT;
  using IndexType = IndexTypeT;
  using DataTuple = typename MakeTuple<NumVals, DataType>::type;
  using IndexTuple = typename MakeTuple<NumVals, IndexType>::type;

  DataTuple avg;
  DataTuple var;
  IndexTuple N;

  WelfordTripletTuple(
      const DataTuple& avg,
      const DataTuple& var,
      const IndexTuple& N)
      : avg(avg), var(var), N(N) {}
};

template <int NumVals, typename DataType, typename IndexType>
using LocalWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataType, IndexType, MakeLocalTuple>;

template <int NumVals, typename DataType, typename IndexType>
using RefWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataType, IndexType, MakeRefTuple>;

template <int NumVals, typename DataType, typename IndexType>
using ConstRefWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataType, IndexType, MakeConstRefTuple>;

template <int NumVals, typename DataTypeT, typename IndexTypeT>
using VolatilePtrWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataTypeT, IndexTypeT, MakeVolatilePtrTuple>;

// Advance pointer offsets of WelfordTripleTuple. Only valid when the
// values are pointer values.
template <typename WelfordTripletTupleType>
__inline__ __device__ static void operator+=(
    WelfordTripletTupleType& triplet,
    nvfuser_index_t offset) {
  triplet.avg += offset;
  triplet.var += offset;
  triplet.N += offset;
}

// Copy each of the triplet tuples
template <typename DstType, typename SrcType>
__inline__ __device__ static void copyWelfordTripletTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset = 0) {
  copyTuple(dst.avg, dst_offset, src.avg, src_offset);
  copyTuple(dst.var, dst_offset, src.var, src_offset);
  copyTuple(dst.N, dst_offset, src.N, src_offset);
}

// Copy each of the triplet tuples
template <typename DstType, typename SrcType>
__inline__ __device__ static void copyWelfordTripletTuple(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset = 0) {
  copyWelfordTripletTuple(dst, 0, src, src_offset);
}

// Copy each of the triplet tuples
template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyWelfordTripletTupleIf(
    DstType& dst,
    const SrcType& src,
    const PredType& pred) {
  copyTupleIf(dst.avg, src.avg, pred);
  copyTupleIf(dst.var, src.var, pred);
  copyTupleIf(dst.N, src.N, pred);
}

} // namespace fused_reduction
