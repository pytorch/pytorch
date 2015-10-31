// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
#define EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H

namespace Eigen {

/** \class TensorKChippingReshaping
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A chip is a thin slice, corresponding to a column or a row in a 2-d tensor.
  *
  *
  */

namespace internal {
template<DenseIndex DimId, typename XprType>
struct traits<TensorChippingOp<DimId, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions - 1;
  static const int Layout = XprTraits::Layout;
};

template<DenseIndex DimId, typename XprType>
struct eval<TensorChippingOp<DimId, XprType>, Eigen::Dense>
{
  typedef const TensorChippingOp<DimId, XprType>& type;
};

template<DenseIndex DimId, typename XprType>
struct nested<TensorChippingOp<DimId, XprType>, 1, typename eval<TensorChippingOp<DimId, XprType> >::type>
{
  typedef TensorChippingOp<DimId, XprType> type;
};

template <DenseIndex DimId>
struct DimensionId
{
  DimensionId(DenseIndex dim) {
    eigen_assert(dim == DimId);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex actualDim() const {
    return DimId;
  }
};
template <>
struct DimensionId<Dynamic>
{
  DimensionId(DenseIndex dim) : actual_dim(dim) {
    eigen_assert(dim >= 0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex actualDim() const {
    return actual_dim;
  }
 private:
  const DenseIndex actual_dim;
};


}  // end namespace internal



template<DenseIndex DimId, typename XprType>
class TensorChippingOp : public TensorBase<TensorChippingOp<DimId, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorChippingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorChippingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorChippingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorChippingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorChippingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorChippingOp(const XprType& expr, const Index offset, const Index dim)
      : m_xpr(expr), m_offset(offset), m_dim(dim) {
  }

  EIGEN_DEVICE_FUNC
  const Index offset() const { return m_offset; }
  EIGEN_DEVICE_FUNC
  const Index dim() const { return m_dim.actualDim(); }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename XprType::Nested>::type&
  expression() const { return m_xpr; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorChippingOp& operator = (const TensorChippingOp& other)
  {
    typedef TensorAssignOp<TensorChippingOp, const TensorChippingOp> Assign;
    Assign assign(*this, other);
    internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    return *this;
  }

  template<typename OtherDerived>
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorChippingOp& operator = (const OtherDerived& other)
  {
    typedef TensorAssignOp<TensorChippingOp, const OtherDerived> Assign;
    Assign assign(*this, other);
    internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    return *this;
  }

  protected:
    typename XprType::Nested m_xpr;
    const Index m_offset;
    const internal::DimensionId<DimId> m_dim;
};


// Eval as rvalue
template<DenseIndex DimId, typename ArgType, typename Device>
struct TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims = NumInputDims-1;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;

  enum {
    // Alignment can't be guaranteed at compile time since it depends on the
    // slice offsets.
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dim(op.dim()), m_device(device)
  {
    // We could also support the case where NumInputDims==1 if needed.
    EIGEN_STATIC_ASSERT(NumInputDims >= 2, YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(NumInputDims > m_dim.actualDim());

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    eigen_assert(op.offset() < input_dims[m_dim.actualDim()]);

    int j = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (i != m_dim.actualDim()) {
        m_dimensions[j] = input_dims[i];
        ++j;
      }
    }

    m_stride = 1;
    m_inputStride = 1;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = 0; i < m_dim.actualDim(); ++i) {
        m_stride *= input_dims[i];
        m_inputStride *= input_dims[i];
      }
    } else {
      for (int i = NumInputDims-1; i > m_dim.actualDim(); --i) {
        m_stride *= input_dims[i];
        m_inputStride *= input_dims[i];
      }
    }
    m_inputStride *= input_dims[m_dim.actualDim()];
    m_inputOffset = m_stride * op.offset();
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_impl.coeff(srcCoeff(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) && m_dim.actualDim() == 0) ||
	(static_cast<int>(Layout) == static_cast<int>(RowMajor) && m_dim.actualDim() == NumInputDims-1)) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(m_stride == 1);
      Index inputIndex = index * m_inputStride + m_inputOffset;
      EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[packetSize];
      for (int i = 0; i < packetSize; ++i) {
        values[i] = m_impl.coeff(inputIndex);
        inputIndex += m_inputStride;
      }
      PacketReturnType rslt = internal::pload<PacketReturnType>(values);
      return rslt;
    } else if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) && m_dim.actualDim() == NumInputDims - 1) ||
	       (static_cast<int>(Layout) == static_cast<int>(RowMajor) && m_dim.actualDim() == 0)) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(m_stride > index);
      return m_impl.template packet<LoadMode>(index + m_inputOffset);
    } else {
      const Index idx = index / m_stride;
      const Index rem = index - idx * m_stride;
      if (rem + packetSize <= m_stride) {
        Index inputIndex = idx * m_inputStride + m_inputOffset + rem;
        return m_impl.template packet<LoadMode>(inputIndex);
      } else {
        // Cross the stride boundary. Fallback to slow path.
        EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[packetSize];
        for (int i = 0; i < packetSize; ++i) {
          values[i] = coeff(index);
          ++index;
        }
        PacketReturnType rslt = internal::pload<PacketReturnType>(values);
        return rslt;
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar* data() const {
    Scalar* result = m_impl.data();
    if (((static_cast<int>(Layout) == static_cast<int>(ColMajor) && m_dim.actualDim() == NumDims) ||
         (static_cast<int>(Layout) == static_cast<int>(RowMajor) && m_dim.actualDim() == 0)) &&
        result) {
      return result + m_inputOffset;
    } else {
      return NULL;
    }
  }

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeff(Index index) const
  {
    Index inputIndex;
    if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) && m_dim.actualDim() == 0) ||
	(static_cast<int>(Layout) == static_cast<int>(RowMajor) && m_dim.actualDim() == NumInputDims-1)) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(m_stride == 1);
      inputIndex = index * m_inputStride + m_inputOffset;
    } else if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) && m_dim.actualDim() == NumInputDims-1) ||
	       (static_cast<int>(Layout) == static_cast<int>(RowMajor) && m_dim.actualDim() == 0)) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(m_stride > index);
      inputIndex = index + m_inputOffset;
    } else {
      const Index idx = index / m_stride;
      inputIndex = idx * m_inputStride + m_inputOffset;
      index -= idx * m_stride;
      inputIndex += index;
    }
    return inputIndex;
  }

  Dimensions m_dimensions;
  Index m_stride;
  Index m_inputOffset;
  Index m_inputStride;
  TensorEvaluator<ArgType, Device> m_impl;
  const internal::DimensionId<DimId> m_dim;
  const Device& m_device;
};


// Eval as lvalue
template<DenseIndex DimId, typename ArgType, typename Device>
struct TensorEvaluator<TensorChippingOp<DimId, ArgType>, Device>
  : public TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device> Base;
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims = NumInputDims-1;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : Base(op, device)
    { }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    return this->m_impl.coeffRef(this->srcCoeff(index));
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    static const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)

    if ((static_cast<int>(this->Layout) == static_cast<int>(ColMajor) && this->m_dim.actualDim() == 0) ||
	(static_cast<int>(this->Layout) == static_cast<int>(RowMajor) && this->m_dim.actualDim() == NumInputDims-1)) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(this->m_stride == 1);
      EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[packetSize];
      internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
      Index inputIndex = index * this->m_inputStride + this->m_inputOffset;
      for (int i = 0; i < packetSize; ++i) {
        this->m_impl.coeffRef(inputIndex) = values[i];
        inputIndex += this->m_inputStride;
      }
    } else if ((static_cast<int>(this->Layout) == static_cast<int>(ColMajor) && this->m_dim.actualDim() == NumInputDims-1) ||
	       (static_cast<int>(this->Layout) == static_cast<int>(RowMajor) && this->m_dim.actualDim() == 0)) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(this->m_stride > index);
      this->m_impl.template writePacket<StoreMode>(index + this->m_inputOffset, x);
    } else {
      const Index idx = index / this->m_stride;
      const Index rem = index - idx * this->m_stride;
      if (rem + packetSize <= this->m_stride) {
        const Index inputIndex = idx * this->m_inputStride + this->m_inputOffset + rem;
        this->m_impl.template writePacket<StoreMode>(inputIndex, x);
      } else {
        // Cross stride boundary. Fallback to slow path.
        EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[packetSize];
        internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
        for (int i = 0; i < packetSize; ++i) {
          this->coeffRef(index) = values[i];
          ++index;
        }
      }
    }
  }
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
