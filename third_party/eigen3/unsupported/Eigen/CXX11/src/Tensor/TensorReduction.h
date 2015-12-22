// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H

namespace Eigen {

/** \class TensorReduction
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reduction class.
  *
  */

namespace internal {
template<typename Op, typename Dims, typename XprType>
struct traits<TensorReductionOp<Op, Dims, XprType> >
 : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
};

template<typename Op, typename Dims, typename XprType>
struct eval<TensorReductionOp<Op, Dims, XprType>, Eigen::Dense>
{
  typedef const TensorReductionOp<Op, Dims, XprType>& type;
};

template<typename Op, typename Dims, typename XprType>
struct nested<TensorReductionOp<Op, Dims, XprType>, 1, typename eval<TensorReductionOp<Op, Dims, XprType> >::type>
{
  typedef TensorReductionOp<Op, Dims, XprType> type;
};


template <typename OutputDims> struct DimInitializer {
  template <typename InputDims, typename ReducedDims> EIGEN_DEVICE_FUNC
  static void run(const InputDims& input_dims,
                  const array<bool, internal::array_size<InputDims>::value>& reduced,
                  OutputDims* output_dims, ReducedDims* reduced_dims) {
    const int NumInputDims = internal::array_size<InputDims>::value;
    int outputIndex = 0;
    int reduceIndex = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (reduced[i]) {
        (*reduced_dims)[reduceIndex] = input_dims[i];
        ++reduceIndex;
      } else {
        (*output_dims)[outputIndex] = input_dims[i];
        ++outputIndex;
      }
    }
  }
};

template <> struct DimInitializer<Sizes<> > {
  template <typename InputDims, typename Index, size_t Rank> EIGEN_DEVICE_FUNC
  static void run(const InputDims& input_dims, const array<bool, Rank>&,
                  Sizes<>*, array<Index, Rank>* reduced_dims) {
    const int NumInputDims = internal::array_size<InputDims>::value;
    for (int i = 0; i < NumInputDims; ++i) {
      (*reduced_dims)[i] = input_dims[i];
    }
  }
};


template <typename ReducedDims, int NumTensorDims, int Layout>
struct are_inner_most_dims {
  static const bool value = false;
};
template <typename ReducedDims, int NumTensorDims, int Layout>
struct preserve_inner_most_dims {
  static const bool value = false;
};

#if defined(EIGEN_HAS_CONSTEXPR) && defined(EIGEN_HAS_VARIADIC_TEMPLATES)
template <typename ReducedDims, int NumTensorDims>
struct are_inner_most_dims<ReducedDims, NumTensorDims, ColMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>();
  static const bool tmp2 = index_statically_eq<ReducedDims>(0, 0);
  static const bool tmp3 = index_statically_eq<ReducedDims>(array_size<ReducedDims>::value-1, array_size<ReducedDims>::value-1);
  static const bool value = tmp1 & tmp2 & tmp3;
};
template <typename ReducedDims, int NumTensorDims>
struct are_inner_most_dims<ReducedDims, NumTensorDims, RowMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>();
  static const bool tmp2 = index_statically_eq<ReducedDims>(0, NumTensorDims - array_size<ReducedDims>::value);
  static const bool tmp3 = index_statically_eq<ReducedDims>(array_size<ReducedDims>::value - 1, NumTensorDims - 1);
  static const bool value = tmp1 & tmp2 & tmp3;

};
template <typename ReducedDims, int NumTensorDims>
struct preserve_inner_most_dims<ReducedDims, NumTensorDims, ColMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>();
  static const bool tmp2 = index_statically_gt<ReducedDims>(0, 0);
  static const bool value = tmp1 & tmp2;

};
template <typename ReducedDims, int NumTensorDims>
struct preserve_inner_most_dims<ReducedDims, NumTensorDims, RowMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>();
  static const bool tmp2 = index_statically_lt<ReducedDims>(array_size<ReducedDims>::value - 1, NumTensorDims - 1);
  static const bool value = tmp1 & tmp2;
};
#endif


template <int DimIndex, typename Self, typename Op>
struct GenericDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    EIGEN_STATIC_ASSERT(DimIndex > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (int j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      GenericDimReducer<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
  }
};
template <typename Self, typename Op>
struct GenericDimReducer<0, Self, Op> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    for (int j = 0; j < self.m_reducedDims[0]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[0];
      reducer.reduce(self.m_impl.coeff(input), accum);
    }
  }
};
template <typename Self, typename Op>
struct GenericDimReducer<-1, Self, Op> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index index, Op& reducer, typename Self::CoeffReturnType* accum) {
    reducer.reduce(self.m_impl.coeff(index), accum);
  }
};

template <typename Self, typename Op, bool Vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct InnerMostDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Self::CoeffReturnType reduce(const Self& self, typename Self::Index firstIndex, typename Self::Index numValuesToReduce, Op& reducer) {
    typename Self::CoeffReturnType accum = reducer.initialize();
    for (typename Self::Index j = 0; j < numValuesToReduce; ++j) {
      reducer.reduce(self.m_impl.coeff(firstIndex + j), &accum);
    }
    return reducer.finalize(accum);
  }
};

template <typename Self, typename Op>
struct InnerMostDimReducer<Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Self::CoeffReturnType reduce(const Self& self, typename Self::Index firstIndex, typename Self::Index numValuesToReduce, Op& reducer) {
    const int packetSize = internal::unpacket_traits<typename Self::PacketReturnType>::size;
    const typename Self::Index VectorizedSize = (numValuesToReduce / packetSize) * packetSize;
    typename Self::PacketReturnType p = reducer.template initializePacket<typename Self::PacketReturnType>();
    for (typename Self::Index j = 0; j < VectorizedSize; j += packetSize) {
      reducer.reducePacket(self.m_impl.template packet<Unaligned>(firstIndex + j), &p);
    }
    typename Self::CoeffReturnType accum = reducer.initialize();
    for (typename Self::Index j = VectorizedSize; j < numValuesToReduce; ++j) {
      reducer.reduce(self.m_impl.coeff(firstIndex + j), &accum);
    }
    return reducer.finalizeBoth(accum, p);
  }
};

template <int DimIndex, typename Self, typename Op, bool vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct InnerMostDimPreserver {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self&, typename Self::Index, Op&, typename Self::PacketReturnType*) {
    eigen_assert(false && "should never be called");
  }
};

template <int DimIndex, typename Self, typename Op>
struct InnerMostDimPreserver<DimIndex, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    EIGEN_STATIC_ASSERT(DimIndex > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (typename Self::Index j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      InnerMostDimPreserver<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
  }
};

template <typename Self, typename Op>
struct InnerMostDimPreserver<0, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    for (typename Self::Index j = 0; j < self.m_reducedDims[0]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[0];
      reducer.reducePacket(self.m_impl.template packet<Unaligned>(input), accum);
    }
  }
};
template <typename Self, typename Op>
struct InnerMostDimPreserver<-1, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self&, typename Self::Index, Op&, typename Self::PacketReturnType*) {
    eigen_assert(false && "should never be called");
  }
};

// Default full reducer
template <typename Self, typename Op, typename Device, bool Vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct FullReducer {
  static const bool HasOptimizedImplementation = false;

  static EIGEN_DEVICE_FUNC void run(const Self& self, Op& reducer, const Device&, typename Self::CoeffReturnType* output) {
    const typename Self::Index num_coeffs = array_prod(self.m_impl.dimensions());
    *output = InnerMostDimReducer<Self, Op>::reduce(self, 0, num_coeffs, reducer);
  }
};


#ifdef EIGEN_USE_THREADS
// Multithreaded full reducers
template <typename Eval, typename Op, bool Vectorizable = (Eval::InputPacketAccess & Op::PacketAccess)>
struct FullReducerShard {
  static void run(const Eval& eval, typename Eval::Index firstIndex, typename Eval::Index numValuesToReduce, Op& reducer, FullReducerShard* shard) {

    shard->saccum = reducer.initialize();
    for (typename Eval::Index j = 0; j < numValuesToReduce; ++j) {
      reducer.reduce(eval.m_impl.coeff(firstIndex + j), &shard->saccum);
    }
  }

  typename Eval::CoeffReturnType saccum;
};

template <typename Eval, typename Op>
struct FullReducerShard<Eval, Op, true> {
  static void run(const Eval& eval, typename Eval::Index firstIndex, typename Eval::Index numValuesToReduce, Op& reducer, FullReducerShard* shard) {

    const int packetSize = internal::unpacket_traits<typename Eval::PacketReturnType>::size;
    const typename Eval::Index VectorizedSize = (numValuesToReduce / packetSize) * packetSize;

    shard->paccum = reducer.template initializePacket<typename Eval::PacketReturnType>();
    for (typename Eval::Index j = 0; j < VectorizedSize; j += packetSize) {
      reducer.reducePacket(eval.m_impl.template packet<Unaligned>(firstIndex + j), &shard->paccum);
    }
    shard->saccum = reducer.initialize();
    for (typename Eval::Index j = VectorizedSize; j < numValuesToReduce; ++j) {
      reducer.reduce(eval.m_impl.coeff(firstIndex + j), &shard->saccum);
    }
  }

  typename Eval::PacketReturnType paccum;
  typename Eval::CoeffReturnType saccum;
};


template <typename Self, typename Op>
struct FullReducer<Self, Op, ThreadPoolDevice, false> {
  static const bool HasOptimizedImplementation = !Op::IsStateful;

  // launch one reducer per thread and accumulate the result.
  static void run(const Self& self, Op& reducer, const ThreadPoolDevice& device, typename Self::CoeffReturnType* output) {
    typedef typename Self::Index Index;
    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const Index blocksize = std::floor<Index>(static_cast<float>(num_coeffs)/device.numThreads());
    const Index numblocks = blocksize > 0 ? num_coeffs / blocksize : 0;
    eigen_assert(num_coeffs >= numblocks * blocksize);

    std::vector<Notification*> results;
    results.reserve(numblocks);
    std::vector<FullReducerShard<Self, Op, false> > shards;
    shards.resize(numblocks);
    for (Index i = 0; i < numblocks; ++i) {
      results.push_back(device.enqueue(&FullReducerShard<Self, Op, false>::run, self, i*blocksize, blocksize, reducer, &shards[i]));
    }

    FullReducerShard<Self, Op, false> finalShard;
    if (numblocks * blocksize < num_coeffs) {
      FullReducerShard<Self, Op, false>::run(self, numblocks * blocksize, num_coeffs - numblocks * blocksize, reducer, &finalShard);
    } else {
      finalShard.saccum = reducer.initialize();
    }

    for (Index i = 0; i < numblocks; ++i) {
      wait_until_ready(results[i]);
      delete results[i];
    }

    for (Index i = 0; i < numblocks; ++i) {
      reducer.reduce(shards[i].saccum, &finalShard.saccum);
    }
    *output = reducer.finalize(finalShard.saccum);
  }
};

template <typename Self, typename Op>
struct FullReducer<Self, Op, ThreadPoolDevice, true> {
  static const bool HasOptimizedImplementation = !Op::IsStateful;

  // launch one reducer per thread and accumulate the result.
  static void run(const Self& self, Op& reducer, const ThreadPoolDevice& device, typename Self::CoeffReturnType* output) {
    typedef typename Self::Index Index;
    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const Index blocksize = std::floor<Index>(static_cast<float>(num_coeffs)/device.numThreads());
    const Index numblocks = blocksize > 0 ? num_coeffs / blocksize : 0;
    eigen_assert(num_coeffs >= numblocks * blocksize);

    std::vector<Notification*> results;
    results.reserve(numblocks);
    std::vector<FullReducerShard<Self, Op, true> > shards;
    shards.resize(numblocks);
    for (Index i = 0; i < numblocks; ++i) {
      results.push_back(device.enqueue(&FullReducerShard<Self, Op, true>::run, self, i*blocksize, blocksize, reducer, &shards[i]));
    }

    FullReducerShard<Self, Op, true> finalShard;
    if (numblocks * blocksize < num_coeffs) {
      FullReducerShard<Self, Op, true>::run(self, numblocks * blocksize, num_coeffs - numblocks * blocksize, reducer, &finalShard);
    } else {
      finalShard.paccum = reducer.template initializePacket<typename Self::PacketReturnType>();
      finalShard.saccum = reducer.initialize();
    }

    for (Index i = 0; i < numblocks; ++i) {
      wait_until_ready(results[i]);
      delete results[i];
    }

    for (Index i = 0; i < numblocks; ++i) {
      reducer.reducePacket(shards[i].paccum, &finalShard.paccum);
      reducer.reduce(shards[i].saccum, &finalShard.saccum);
    }

    *output = reducer.finalizeBoth(finalShard.saccum, finalShard.paccum);
  }
};
#endif


#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
template <int B, int N, typename S, typename R, typename I>
__global__ void FullReductionKernel(R, const S, I, typename S::CoeffReturnType*);
#endif

}  // end namespace internal


template <typename Op, typename Dims, typename XprType>
class TensorReductionOp : public TensorBase<TensorReductionOp<Op, Dims, XprType>, ReadOnlyAccessors> {
  public:
    typedef typename Eigen::internal::traits<TensorReductionOp>::Scalar Scalar;
    typedef typename Eigen::internal::traits<TensorReductionOp>::Packet Packet;
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
    typedef typename internal::remove_const<typename XprType::PacketReturnType>::type PacketReturnType;
    typedef typename Eigen::internal::nested<TensorReductionOp>::type Nested;
    typedef typename Eigen::internal::traits<TensorReductionOp>::StorageKind StorageKind;
    typedef typename Eigen::internal::traits<TensorReductionOp>::Index Index;

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReductionOp(const XprType& expr, const Dims& dims) : m_expr(expr), m_dims(dims)
    { }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReductionOp(const XprType& expr, const Dims& dims, const Op& reducer) : m_expr(expr), m_dims(dims), m_reducer(reducer)
    { }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const XprType& expression() const { return m_expr; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const Dims& dims() const { return m_dims; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const Op& reducer() const { return m_reducer; }

  protected:
    typename XprType::Nested m_expr;
    const Dims m_dims;
    const Op m_reducer;
};


// Eval as rvalue
template<typename Op, typename Dims, typename ArgType, typename Device>
struct TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType>, Device>
{
  typedef TensorReductionOp<Op, Dims, ArgType> XprType;
  typedef typename XprType::Index Index;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions InputDimensions;
  static const int NumInputDims = internal::array_size<InputDimensions>::value;
  static const int NumReducedDims = internal::array_size<Dims>::value;
  static const int NumOutputDims = NumInputDims - NumReducedDims;
  typedef typename internal::conditional<NumOutputDims==0, Sizes<>, DSizes<Index, NumOutputDims> >::type Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType>, Device> Self;
  static const bool InputPacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess;

  enum {
    IsAligned = false,
    PacketAccess = Self::InputPacketAccess && Op::PacketAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  static const bool ReducingInnerMostDims = internal::are_inner_most_dims<Dims, NumInputDims, Layout>::value;
  static const bool PreservingInnerMostDims = internal::preserve_inner_most_dims<Dims, NumInputDims, Layout>::value;
  static const bool RunningFullReduction = (NumOutputDims==0);

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_reducer(op.reducer()), m_result(NULL), m_device(device)
  {
    EIGEN_STATIC_ASSERT(NumInputDims >= NumReducedDims, YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT((!ReducingInnerMostDims | !PreservingInnerMostDims | (NumReducedDims == NumInputDims)),
                        YOU_MADE_A_PROGRAMMING_MISTAKE);

    // Bitmap indicating if an input dimension is reduced or not.
    array<bool, NumInputDims> reduced;
    for (int i = 0; i < NumInputDims; ++i) {
      reduced[i] = false;
    }
    for (int i = 0; i < NumReducedDims; ++i) {
      eigen_assert(op.dims()[i] >= 0);
      eigen_assert(op.dims()[i] < NumInputDims);
      reduced[op.dims()[i]] = true;
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    internal::DimInitializer<Dimensions>::run(input_dims, reduced, &m_dimensions, &m_reducedDims);

    // Precompute output strides.
    if (NumOutputDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
	m_outputStrides[0] = 1;
	for (int i = 1; i < NumOutputDims; ++i) {
	  m_outputStrides[i] = m_outputStrides[i - 1] * m_dimensions[i - 1];
	}
      } else {
	m_outputStrides[NumOutputDims - 1] = 1;
	for (int i = NumOutputDims - 2; i >= 0; --i) {
	  m_outputStrides[i] = m_outputStrides[i + 1] * m_dimensions[i + 1];
	}
      }
    }
    
    // Precompute input strides.
    if (NumInputDims > 0) {
      array<Index, NumInputDims> input_strides;
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
	input_strides[0] = 1;
	for (int i = 1; i < NumInputDims; ++i) {
	  input_strides[i] = input_strides[i-1] * input_dims[i-1];
	}
      } else {
	input_strides[NumInputDims - 1] = 1;
	for (int i = NumInputDims - 2; i >= 0; --i) {
	  input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
	}
      }
      
      int outputIndex = 0;
      int reduceIndex = 0;
      for (int i = 0; i < NumInputDims; ++i) {
	if (reduced[i]) {
	  m_reducedStrides[reduceIndex] = input_strides[i];
	  ++reduceIndex;
	} else {
	  m_preservedStrides[outputIndex] = input_strides[i];
	  ++outputIndex;
	}
      }
    }

    // Special case for full reductions
    if (NumOutputDims == 0) {
      m_preservedStrides[0] = internal::array_prod(input_dims);
    }
  }
  
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
  typedef typename internal::remove_const<typename XprType::PacketReturnType>::type PacketReturnType;

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    m_impl.evalSubExprsIfNeeded(NULL);

    // Use the FullReducer if possible.
    if (RunningFullReduction && internal::FullReducer<Self, Op, Device>::HasOptimizedImplementation &&
        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
         (internal::array_prod(m_impl.dimensions()) > 1024 * 1024))) {

      bool need_assign = false;
      if (!data) {
        m_result = static_cast<CoeffReturnType*>(m_device.allocate(sizeof(CoeffReturnType)));
        data = m_result;
        need_assign = true;
      }

      Op reducer(m_reducer);
      internal::FullReducer<Self, Op, Device>::run(*this, reducer, m_device, data);
      return need_assign;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
    if (m_result) {
      m_device.deallocate(m_result);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    if (RunningFullReduction && m_result) {
      return *m_result;
    }
    Op reducer(m_reducer);
    if (ReducingInnerMostDims || RunningFullReduction) {
      const Index num_values_to_reduce =
	(static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_preservedStrides[0] : m_preservedStrides[NumPreservedStrides - 1];
      return internal::InnerMostDimReducer<Self, Op>::reduce(*this, firstInput(index),
                                                             num_values_to_reduce, reducer);
    } else {
      typename Self::CoeffReturnType accum = reducer.initialize();
      internal::GenericDimReducer<NumReducedDims-1, Self, Op>::reduce(*this, firstInput(index), reducer, &accum);
      return reducer.finalize(accum);
    }
  }

  // TODO(bsteiner): provide a more efficient implementation.
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index + packetSize - 1 < dimensions().TotalSize());

    EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    if (ReducingInnerMostDims) {
      const Index num_values_to_reduce =
	(static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_preservedStrides[0] : m_preservedStrides[NumPreservedStrides - 1];
      const Index firstIndex = firstInput(index);
      for (Index i = 0; i < packetSize; ++i) {
        Op reducer(m_reducer);
        values[i] = internal::InnerMostDimReducer<Self, Op>::reduce(*this, firstIndex + i * num_values_to_reduce,
                                                                    num_values_to_reduce, reducer);
      }
    } else if (PreservingInnerMostDims) {
      const Index firstIndex = firstInput(index);
      const int innermost_dim = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? 0 : NumOutputDims - 1;
      // TBD: extend this the the n innermost dimensions that we preserve.
      if (((firstIndex % m_dimensions[innermost_dim]) + packetSize - 1) < m_dimensions[innermost_dim]) {
        Op reducer(m_reducer);
        typename Self::PacketReturnType accum = reducer.template initializePacket<typename Self::PacketReturnType>();
        internal::InnerMostDimPreserver<NumReducedDims-1, Self, Op>::reduce(*this, firstIndex, reducer, &accum);
        return reducer.finalizePacket(accum);
      } else {
        for (int i = 0; i < packetSize; ++i) {
          values[i] = coeff(index + i);
        }
      }
    } else {
      for (int i = 0; i < packetSize; ++i) {
        values[i] = coeff(index + i);
      }
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

  private:
  template <int, typename, typename> friend struct internal::GenericDimReducer;
  template <typename, typename, bool> friend struct internal::InnerMostDimReducer;
  template <int, typename, typename, bool> friend struct internal::InnerMostDimPreserver;
  template <typename S, typename O, typename D, bool V> friend struct internal::FullReducer;
#ifdef EIGEN_USE_THREADS
  template <typename S, typename O, bool V> friend struct internal::FullReducerShard;
#endif
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
  template <int B, int N, typename S, typename R, typename I> friend void internal::FullReductionKernel(R, const S, I, typename S::CoeffReturnType*);
#endif

  // Returns the Index in the input tensor of the first value that needs to be
  // used to compute the reduction at output index "index".
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index firstInput(Index index) const {
    if (ReducingInnerMostDims) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        return index * m_preservedStrides[0];
      } else {
        return index * m_preservedStrides[NumPreservedStrides - 1];
      }
    }
    // TBD: optimize the case where we preserve the innermost dimensions.
    Index startInput = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumOutputDims - 1; i > 0; --i) {
        // This is index_i in the output tensor.
        const Index idx = index / m_outputStrides[i];
        startInput += idx * m_preservedStrides[i];
        index -= idx * m_outputStrides[i];
      }
      if (PreservingInnerMostDims) {
        eigen_assert(m_preservedStrides[0] == 1);
        startInput += index;
      } else {
        startInput += index * m_preservedStrides[0];
      }
    } else {
      for (int i = 0; i < NumOutputDims - 1; ++i) {
        // This is index_i in the output tensor.
        const Index idx = index / m_outputStrides[i];
        startInput += idx * m_preservedStrides[i];
        index -= idx * m_outputStrides[i];
      }
      if (PreservingInnerMostDims) {
        eigen_assert(m_preservedStrides[NumPreservedStrides - 1] == 1);
        startInput += index;
      } else {
        startInput += index * m_preservedStrides[NumPreservedStrides - 1];
      }
    }
    return startInput;
  }

  // Dimensions of the output of the operation.
  Dimensions m_dimensions;
  // Precomputed strides for the output tensor.
  array<Index, NumOutputDims> m_outputStrides;
  // Subset of strides of the input tensor for the non-reduced dimensions.
  // Indexed by output dimensions.
  static const int NumPreservedStrides = max_n_1<NumOutputDims>::size;
  array<Index, NumPreservedStrides> m_preservedStrides;

  // Subset of strides of the input tensor for the reduced dimensions.
  // Indexed by reduced dimensions.
  array<Index, NumReducedDims> m_reducedStrides;
  // Size of the input dimensions that are reduced.
  // Indexed by reduced dimensions.
  array<Index, NumReducedDims> m_reducedDims;

  // Evaluator for the input expression.
  TensorEvaluator<ArgType, Device> m_impl;

  // Operation to apply for computing the reduction.
  Op m_reducer;

  // For full reductions
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
#else
  static const bool RunningOnGPU = false;
#endif
  CoeffReturnType* m_result;

  const Device& m_device;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
