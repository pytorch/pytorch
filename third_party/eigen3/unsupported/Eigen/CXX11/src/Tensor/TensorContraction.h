// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H

namespace Eigen {

/** \class TensorContraction
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor contraction class.
  *
  *
  */
namespace internal {

enum {
  Rhs = 0,
  Lhs = 1,
};

/*
 * Implementation of the Eigen blas_data_mapper class for tensors.
 */
template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size, bool inner_dim_contiguous>
class SimpleTensorContractionMapper {
  public:
  EIGEN_DEVICE_FUNC
  SimpleTensorContractionMapper(const Tensor& tensor,
                                const nocontract_t& nocontract_strides,
                                const nocontract_t& ij_strides,
                                const contract_t& contract_strides,
                                const contract_t& k_strides) :
      m_tensor(tensor),
      m_nocontract_strides(nocontract_strides),
      m_ij_strides(ij_strides),
      m_contract_strides(contract_strides),
      m_k_strides(k_strides) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE void prefetch(Index /*i*/) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row) const {
    // column major assumption
    return operator()(row, 0);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index col) const {
    return m_tensor.coeff(computeIndex(row, col));
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Index computeIndex(Index row, Index col) const {
    const bool left = (side == Lhs);
    Index nocontract_val = left ? row : col;
    Index linidx = 0;
    for (int i = static_cast<int>(array_size<nocontract_t>::value) - 1; i > 0; i--) {
      const Index idx = nocontract_val / m_ij_strides[i];
      linidx += idx * m_nocontract_strides[i];
      nocontract_val -= idx * m_ij_strides[i];
    }
    if (array_size<typename Tensor::Dimensions>::value > array_size<contract_t>::value) {
      if (side == Lhs && inner_dim_contiguous) {
        eigen_assert(m_nocontract_strides[0] == 1);
        linidx += nocontract_val;
      } else {
        linidx += nocontract_val * m_nocontract_strides[0];
      }
    }

    Index contract_val = left ? col : row;
    for (int i = static_cast<int>(array_size<contract_t>::value) - 1; i > 0; i--) {
      const Index idx = contract_val / m_k_strides[i];
      linidx += idx * m_contract_strides[i];
      contract_val -= idx * m_k_strides[i];
    }

    if(array_size<contract_t>::value > 0) {
        if (side == Rhs && inner_dim_contiguous) {
            eigen_assert(m_contract_strides[0] == 1);
            linidx += contract_val;
        } else {
            linidx += contract_val * m_contract_strides[0];
        }
    }

    return linidx;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE IndexPair<Index> computeIndexPair(Index row, Index col, const Index distance) const {
    const bool left = (side == Lhs);
    Index nocontract_val[2] = {left ? row : col, left ? row + distance : col};
    Index linidx[2] = {0, 0};
    for (int i = static_cast<int>(array_size<nocontract_t>::value) - 1; i > 0; i--) {
      const Index idx0 = nocontract_val[0] / m_ij_strides[i];
      const Index idx1 = nocontract_val[1] / m_ij_strides[i];
      linidx[0] += idx0 * m_nocontract_strides[i];
      linidx[1] += idx1 * m_nocontract_strides[i];
      nocontract_val[0] -= idx0 * m_ij_strides[i];
      nocontract_val[1] -= idx1 * m_ij_strides[i];
    }
    if (array_size<typename Tensor::Dimensions>::value > array_size<contract_t>::value) {
      if (side == Lhs && inner_dim_contiguous) {
        eigen_assert(m_nocontract_strides[0] == 1);
        linidx[0] += nocontract_val[0];
        linidx[1] += nocontract_val[1];
      } else {
        linidx[0] += nocontract_val[0] * m_nocontract_strides[0];
        linidx[1] += nocontract_val[1] * m_nocontract_strides[0];
      }
    }

    Index contract_val[2] = {left ? col : row, left ? col : row + distance};
    for (int i = static_cast<int>(array_size<contract_t>::value) - 1; i > 0; i--) {
      const Index idx0 = contract_val[0] / m_k_strides[i];
      const Index idx1 = contract_val[1] / m_k_strides[i];
      linidx[0] += idx0 * m_contract_strides[i];
      linidx[1] += idx1 * m_contract_strides[i];
      contract_val[0] -= idx0 * m_k_strides[i];
      contract_val[1] -= idx1 * m_k_strides[i];
    }

    if (side == Rhs && inner_dim_contiguous) {
      eigen_assert(m_contract_strides[0] == 1);
      linidx[0] += contract_val[0];
      linidx[1] += contract_val[1];
    } else {
      linidx[0] += contract_val[0] * m_contract_strides[0];
      linidx[1] += contract_val[1] * m_contract_strides[0];
    }
    return IndexPair<Index>(linidx[0], linidx[1]);
  }

  Index firstAligned(Index size) const {
    return size;
  }
  Index stride() const {
    return 1;
  }

 protected:
  const Tensor m_tensor;
  const nocontract_t m_nocontract_strides;
  const nocontract_t m_ij_strides;
  const contract_t m_contract_strides;
  const contract_t m_k_strides;
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size, bool inner_dim_contiguous,
         bool inner_dim_reordered, int Alignment>
  class BaseTensorContractionMapper : public SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous>
{
 public:
  typedef SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous> ParentMapper;

  EIGEN_DEVICE_FUNC
  BaseTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
  ParentMapper(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadPacket(Index i, Index j) const {
    // whole method makes column major assumption

    // don't need to add offsets for now (because operator handles that)
    // current code assumes packet size must be a multiple of 2
    EIGEN_STATIC_ASSERT(packet_size % 2 == 0, YOU_MADE_A_PROGRAMMING_MISTAKE);

    if (Tensor::PacketAccess && inner_dim_contiguous && !inner_dim_reordered) {
      const Index index = this->computeIndex(i, j);
      eigen_assert(this->computeIndex(i+packet_size-1, j) == index + packet_size-1);
      return this->m_tensor.template packet<Alignment>(index);
    }

    const IndexPair<Index> indexPair = this->computeIndexPair(i, j, packet_size - 1);
    const Index first = indexPair.first;
    const Index last = indexPair.second;

    // We can always do optimized packet reads from left hand side right now, because
    // the vertical matrix dimension on the left hand side is never contracting.
    // On the right hand side we need to check if the contracting dimensions may have
    // been shuffled first.
    if (Tensor::PacketAccess &&
        (side == Lhs || internal::array_size<contract_t>::value <= 1 || !inner_dim_reordered) &&
        (last - first) == (packet_size - 1)) {

      return this->m_tensor.template packet<Alignment>(first);
    }

    EIGEN_ALIGN_MAX Scalar data[packet_size];

    data[0] = this->m_tensor.coeff(first);
    for (Index k = 1; k < packet_size - 1; k += 2) {
      const IndexPair<Index> internal_pair = this->computeIndexPair(i + k, j, 1);
      data[k] = this->m_tensor.coeff(internal_pair.first);
      data[k + 1] = this->m_tensor.coeff(internal_pair.second);
    }
    data[packet_size - 1] = this->m_tensor.coeff(last);

    return pload<Packet>(data);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE HalfPacket loadHalfPacket(Index i, Index j) const {
    // whole method makes column major assumption

    // don't need to add offsets for now (because operator handles that)
    const Index half_packet_size = unpacket_traits<HalfPacket>::size;
    if (half_packet_size == packet_size) {
      return loadPacket(i, j);
    }
    EIGEN_ALIGN_MAX Scalar data[half_packet_size];
    for (Index k = 0; k < half_packet_size; k++) {
      data[k] = operator()(i + k, j);
    }
    return pload<HalfPacket>(data);
  }
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         bool inner_dim_contiguous,
         bool inner_dim_reordered, int Alignment>
class BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered, Alignment> : public SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous>
{
 public:
  typedef SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous> ParentMapper;

  EIGEN_DEVICE_FUNC
  BaseTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
  ParentMapper(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadPacket(Index i, Index j) const {
    EIGEN_ALIGN_MAX Scalar data[1];
    data[0] = this->m_tensor.coeff(this->computeIndex(i, j));
    return pload<typename packet_traits<Scalar>::type>(data);
  }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadHalfPacket(Index i, Index j) const {
    return loadPacket(i, j);
  }
};

template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper;

template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper {
 public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  typedef TensorContractionInputMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> ParentMapper;
  typedef TensorContractionSubMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> Self;
  typedef Self LinearMapper;

  EIGEN_DEVICE_FUNC TensorContractionSubMapper(const ParentMapper& base_mapper, Index vert_offset, Index horiz_offset)
      : m_base_mapper(base_mapper), m_vert_offset(vert_offset), m_horiz_offset(horiz_offset) { }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_base_mapper(i + m_vert_offset, m_horiz_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i, Index j) const {
    return m_base_mapper(i + m_vert_offset, j + m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    return m_base_mapper.loadPacket(i + m_vert_offset, m_horiz_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index j) const {
   return m_base_mapper.loadPacket(i + m_vert_offset, j + m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE HalfPacket loadHalfPacket(Index i) const {
    return m_base_mapper.loadHalfPacket(i + m_vert_offset, m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void storePacket(Index i, Packet p) const {
    m_base_mapper.storePacket(i + m_vert_offset, m_horiz_offset, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(m_base_mapper, i + m_vert_offset, j + m_horiz_offset);
  }

  template <typename PacketT, int AlignmentType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketT load(Index i) const {
    EIGEN_STATIC_ASSERT((internal::is_same<PacketT, Packet>::value), YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT((AlignmentType == Aligned || Alignment == Unaligned), YOU_MADE_A_PROGRAMMING_MISTAKE);
    return loadPacket(i);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC bool aligned(Index) const {
    return false;
  }

 private:
  const ParentMapper& m_base_mapper;
  const Index m_vert_offset;
  const Index m_horiz_offset;
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper
  : public BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> {

 public:
  typedef BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> Base;
  typedef TensorContractionSubMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> SubMapper;
  typedef SubMapper VectorMapper;

  EIGEN_DEVICE_FUNC TensorContractionInputMapper(const Tensor& tensor,
                               const nocontract_t& nocontract_strides,
                               const nocontract_t& ij_strides,
                               const contract_t& contract_strides,
                               const contract_t& k_strides)
      : Base(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE SubMapper getSubMapper(Index i, Index j) const {
    return SubMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE VectorMapper getVectorMapper(Index i, Index j) const {
    return VectorMapper(*this, i, j);
  }
};



template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct traits<TensorContractionOp<Dimensions, LhsXprType, RhsXprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename internal::promote_storage_type<typename LhsXprType::Scalar,
                                                  typename RhsXprType::Scalar>::ret Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename promote_storage_type<typename traits<LhsXprType>::StorageKind,
                                        typename traits<RhsXprType>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<LhsXprType>::Index,
                                      typename traits<RhsXprType>::Index>::type Index;
  typedef typename LhsXprType::Nested LhsNested;
  typedef typename RhsXprType::Nested RhsNested;
  typedef typename remove_reference<LhsNested>::type _LhsNested;
  typedef typename remove_reference<RhsNested>::type _RhsNested;

  // From NumDims below.
  static const int NumDimensions = max_n_1<traits<RhsXprType>::NumDimensions + traits<RhsXprType>::NumDimensions - 2 * array_size<Dimensions>::value>::size;
  static const int Layout = traits<LhsXprType>::Layout;

  enum {
    Flags = 0,
  };
};

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct eval<TensorContractionOp<Dimensions, LhsXprType, RhsXprType>, Eigen::Dense>
{
  typedef const TensorContractionOp<Dimensions, LhsXprType, RhsXprType>& type;
};

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct nested<TensorContractionOp<Dimensions, LhsXprType, RhsXprType>, 1, typename eval<TensorContractionOp<Dimensions, LhsXprType, RhsXprType> >::type>
{
  typedef TensorContractionOp<Dimensions, LhsXprType, RhsXprType> type;
};

template<typename Indices_, typename LeftArgType_, typename RightArgType_, typename Device_>
struct traits<TensorEvaluator<const TensorContractionOp<Indices_, LeftArgType_, RightArgType_>, Device_> > {
  typedef Indices_ Indices;
  typedef LeftArgType_ LeftArgType;
  typedef RightArgType_ RightArgType;
  typedef Device_ Device;

  // From NumDims below.
  static const int NumDimensions = max_n_1<traits<LeftArgType_>::NumDimensions + traits<RightArgType_>::NumDimensions - 2 * array_size<Indices_>::value>::size;
};

}  // end namespace internal

template<typename Indices, typename LhsXprType, typename RhsXprType>
class TensorContractionOp : public TensorBase<TensorContractionOp<Indices, LhsXprType, RhsXprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorContractionOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorContractionOp>::Packet Packet;
  typedef typename internal::promote_storage_type<typename LhsXprType::CoeffReturnType,
                                                  typename RhsXprType::CoeffReturnType>::ret CoeffReturnType;
  typedef typename internal::promote_storage_type<typename LhsXprType::PacketReturnType,
                                                  typename RhsXprType::PacketReturnType>::ret PacketReturnType;
  typedef typename Eigen::internal::nested<TensorContractionOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorContractionOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorContractionOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionOp(
      const LhsXprType& lhs, const RhsXprType& rhs, const Indices& dims)
      : m_lhs_xpr(lhs), m_rhs_xpr(rhs), m_indices(dims) {}

  EIGEN_DEVICE_FUNC
  const Indices& indices() const { return m_indices; }

  /** \returns the nested expressions */
  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename LhsXprType::Nested>::type&
  lhsExpression() const { return m_lhs_xpr; }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename RhsXprType::Nested>::type&
  rhsExpression() const { return m_rhs_xpr; }

  protected:
    typename LhsXprType::Nested m_lhs_xpr;
    typename RhsXprType::Nested m_rhs_xpr;
    const Indices m_indices;
};


template<typename Derived>
struct TensorContractionEvaluatorBase
{
  typedef typename internal::traits<Derived>::Indices Indices;
  typedef typename internal::traits<Derived>::LeftArgType LeftArgType;
  typedef typename internal::traits<Derived>::RightArgType RightArgType;
  typedef typename internal::traits<Derived>::Device Device;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Packet Packet;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  enum {
    IsAligned = true,
    PacketAccess = (internal::packet_traits<Scalar>::size > 1),
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;
  static const int NumDims = max_n_1<LDims + RDims - 2 * ContractDims>::size;

  typedef array<Index, LDims> left_dim_mapper_t;
  typedef array<Index, RDims> right_dim_mapper_t;
  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, max_n_1<LDims - ContractDims>::size> left_nocontract_t;
  typedef array<Index, max_n_1<RDims - ContractDims>::size> right_nocontract_t;

  typedef DSizes<Index, NumDims> Dimensions;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  TensorContractionEvaluatorBase(const XprType& op, const Device& device)
    : m_leftImpl(choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(),
                          op.lhsExpression(), op.rhsExpression()), device),
    m_rightImpl(choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(),
                          op.rhsExpression(), op.lhsExpression()), device),
        m_device(device),
        m_result(NULL) {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<LeftArgType, Device>::Layout) ==
			   static_cast<int>(TensorEvaluator<RightArgType, Device>::Layout)),
                        YOU_MADE_A_PROGRAMMING_MISTAKE);


    DSizes<Index, LDims> eval_left_dims;
    DSizes<Index, RDims> eval_right_dims;
    array<IndexPair<Index>, ContractDims> eval_op_indices;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      // For ColMajor, we keep using the existing dimensions
      for (int i = 0; i < LDims; i++) {
        eval_left_dims[i] = m_leftImpl.dimensions()[i];
      }
      for (int i = 0; i < RDims; i++) {
        eval_right_dims[i] = m_rightImpl.dimensions()[i];
      }
      // We keep the pairs of contracting indices.
      for (int i = 0; i < ContractDims; i++) {
        eval_op_indices[i].first = op.indices()[i].first;
        eval_op_indices[i].second = op.indices()[i].second;
      }
    } else {
      // For RowMajor, we need to reverse the existing dimensions
      for (int i = 0; i < LDims; i++) {
        eval_left_dims[i] = m_leftImpl.dimensions()[LDims - i - 1];
      }
      for (int i = 0; i < RDims; i++) {
        eval_right_dims[i] = m_rightImpl.dimensions()[RDims - i - 1];
      }
      // We need to flip all the pairs of contracting indices as well as
      // reversing the dimensions.
      for (int i = 0; i < ContractDims; i++) {
        eval_op_indices[i].first = LDims - 1 - op.indices()[i].second;
        eval_op_indices[i].second = RDims - 1 - op.indices()[i].first;
      }
    }

    array<Index, LDims> lhs_strides;
    lhs_strides[0] = 1;
    for (int i = 0; i < LDims-1; ++i) {
      lhs_strides[i+1] = lhs_strides[i] * eval_left_dims[i];
    }

    array<Index, RDims> rhs_strides;
    rhs_strides[0] = 1;
    for (int i = 0; i < RDims-1; ++i) {
      rhs_strides[i+1] = rhs_strides[i] * eval_right_dims[i];
    }

    m_i_strides[0] = 1;
    m_j_strides[0] = 1;
    if(ContractDims) {
        m_k_strides[0] = 1;
    }

    m_i_size = 1;
    m_j_size = 1;
    m_k_size = 1;

    // To compute the dimension, we simply concatenate the non-contracting
    // dimensions of the left and then the right tensor. Additionally, we also
    // compute the strides corresponding to the left non-contracting
    // dimensions and right non-contracting dimensions.
    m_lhs_inner_dim_contiguous = true;
    int dim_idx = 0;
    unsigned int nocontract_idx = 0;

    for (int i = 0; i < LDims; i++) {
      // find if we are contracting on index i of left tensor
      bool contracting = false;
      for (int j = 0; j < ContractDims; j++) {
        if (eval_op_indices[j].first == i) {
          contracting = true;
          break;
        }
      }
      if (!contracting) {
        // add dimension size to output dimensions
        m_dimensions[dim_idx] = eval_left_dims[i];
        m_left_nocontract_strides[nocontract_idx] = lhs_strides[i];
        if (dim_idx != i) {
          m_lhs_inner_dim_contiguous = false;
        }
        if (nocontract_idx+1 < internal::array_size<left_nocontract_t>::value) {
          m_i_strides[nocontract_idx+1] =
              m_i_strides[nocontract_idx] * eval_left_dims[i];
        } else {
          m_i_size = m_i_strides[nocontract_idx] * eval_left_dims[i];
        }
        dim_idx++;
        nocontract_idx++;
      }
    }

    nocontract_idx = 0;
    for (int i = 0; i < RDims; i++) {
      bool contracting = false;
      // find if we are contracting on index i of right tensor
      for (int j = 0; j < ContractDims; j++) {
        if (eval_op_indices[j].second == i) {
          contracting = true;
          break;
        }
      }
      if (!contracting) {
        m_dimensions[dim_idx] = eval_right_dims[i];
        if (nocontract_idx+1 < internal::array_size<right_nocontract_t>::value) {
          m_j_strides[nocontract_idx+1] =
              m_j_strides[nocontract_idx] * eval_right_dims[i];
        } else {
          m_j_size = m_j_strides[nocontract_idx] * eval_right_dims[i];
        }
        m_right_nocontract_strides[nocontract_idx] = rhs_strides[i];
        dim_idx++;
        nocontract_idx++;
      }
    }

    // Now compute the strides corresponding to the contracting dimensions. We
    // assumed above that non-contracting axes are represented in the same order
    // in the matrix as they are in the tensor. This is not the case for
    // contracting axes. As the contracting axes must be of the same size in
    // each tensor, we'll only look at the first tensor here.
    m_rhs_inner_dim_contiguous = true;
    m_rhs_inner_dim_reordered = false;
    for (int i = 0; i < ContractDims; i++) {
      Index left = eval_op_indices[i].first;
      Index right = eval_op_indices[i].second;

      Index size = eval_left_dims[left];
      eigen_assert(size == eval_right_dims[right] &&
                   "Contraction axes must be same size");

      if (i+1 < static_cast<int>(internal::array_size<contract_t>::value)) {
        m_k_strides[i+1] = m_k_strides[i] * size;
      } else {
        m_k_size = m_k_strides[i] * size;
      }
      m_left_contracting_strides[i] = lhs_strides[left];
      m_right_contracting_strides[i] = rhs_strides[right];

      if (i > 0 && right < eval_op_indices[i-1].second) {
        m_rhs_inner_dim_reordered = true;
      }
      if (right != i) {
        m_rhs_inner_dim_contiguous = false;
      }
    }

    // Scalar case. We represent the result as a 1d tensor of size 1.
    if (LDims + RDims == 2 * ContractDims) {
      m_dimensions[0] = 1;
    }

    // If the layout is RowMajor, we need to reverse the m_dimensions
    if (static_cast<int>(Layout) == static_cast<int>(RowMajor)) {
      for (int i = 0, j = NumDims - 1; i < j; i++, j--) {
        numext::swap(m_dimensions[i], m_dimensions[j]);
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
    m_leftImpl.evalSubExprsIfNeeded(NULL);
    m_rightImpl.evalSubExprsIfNeeded(NULL);
    if (data) {
      evalTo(data);
      return false;
    } else {
      m_result = static_cast<Scalar *>(m_device.allocate(dimensions().TotalSize() * sizeof(Scalar)));
      evalTo(m_result);
      return true;
    }
  }

  EIGEN_DEVICE_FUNC void evalTo(Scalar* buffer) const {
    if (this->m_lhs_inner_dim_contiguous) {
      if (this->m_rhs_inner_dim_contiguous) {
        if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<true, true, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<true, true, false, Unaligned>(buffer);
        }
      }
      else {
       if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<true, false, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<true, false, false, Unaligned>(buffer);
        }
      }
    }
    else {
      if (this->m_rhs_inner_dim_contiguous) {
        if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<false, true, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<false, true, false, Unaligned>(buffer);
        }
      }
      else {
       if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<false, false, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<false, false, false, Unaligned>(buffer);
        }
      }
    }
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalGemv(Scalar* buffer) const {
    const Index rows = m_i_size;
    const Index cols = m_k_size;

    typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
    typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;
    typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
    typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;
    const Index lhs_packet_size = internal::packet_traits<LhsScalar>::size;
    const Index rhs_packet_size = internal::packet_traits<RhsScalar>::size;
    typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                   LeftEvaluator, left_nocontract_t,
                                                   contract_t, lhs_packet_size,
                                                   lhs_inner_dim_contiguous,
                                                   false, Unaligned> LhsMapper;

    typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                   RightEvaluator, right_nocontract_t,
                                                   contract_t, rhs_packet_size,
                                                   rhs_inner_dim_contiguous,
                                                   rhs_inner_dim_reordered, Unaligned> RhsMapper;

    LhsMapper lhs(m_leftImpl, m_left_nocontract_strides, m_i_strides,
                  m_left_contracting_strides, m_k_strides);
    RhsMapper rhs(m_rightImpl, m_right_nocontract_strides, m_j_strides,
                  m_right_contracting_strides, m_k_strides);

    const Scalar alpha(1);
    const Index resIncr(1);

    // zero out the result buffer (which must be of size at least rows * sizeof(Scalar)
    m_device.memset(buffer, 0, rows * sizeof(Scalar));

    internal::general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,false,RhsScalar,RhsMapper,false>::run(
        rows, cols, lhs, rhs,
        buffer, resIncr, alpha);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_leftImpl.cleanup();
    m_rightImpl.cleanup();

    if (m_result != NULL) {
      m_device.deallocate(m_result);
      m_result = NULL;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    return m_result[index];
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const {
    return internal::ploadt<Packet, LoadMode>(m_result + index);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

  protected:
  // Prevent assignment
  TensorContractionEvaluatorBase& operator = (const TensorContractionEvaluatorBase&);
  Dimensions m_dimensions;

  contract_t m_k_strides;
  contract_t m_left_contracting_strides;
  contract_t m_right_contracting_strides;

  bool m_lhs_inner_dim_contiguous;
  bool m_rhs_inner_dim_contiguous;
  bool m_rhs_inner_dim_reordered;

  left_nocontract_t m_i_strides;
  right_nocontract_t m_j_strides;
  left_nocontract_t m_left_nocontract_strides;
  right_nocontract_t m_right_nocontract_strides;

  Index m_i_size;
  Index m_j_size;
  Index m_k_size;

  TensorEvaluator<EvalLeftArgType, Device> m_leftImpl;
  TensorEvaluator<EvalRightArgType, Device> m_rightImpl;
  const Device& m_device;
  Scalar* m_result;
};


// evaluator for default device
template<typename Indices, typename LeftArgType, typename RightArgType, typename Device>
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> :
    public TensorContractionEvaluatorBase<
      TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> > {
  typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> Self;
  typedef TensorContractionEvaluatorBase<Self> Base;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Packet Packet;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  enum {
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;

  typedef array<Index, LDims> left_dim_mapper_t;
  typedef array<Index, RDims> right_dim_mapper_t;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, max_n_1<LDims - ContractDims>::size> left_nocontract_t;
  typedef array<Index, max_n_1<RDims - ContractDims>::size> right_nocontract_t;

  static const int NumDims = max_n_1<LDims + RDims - 2 * ContractDims>::size;

  // Could we use NumDimensions here?
  typedef DSizes<Index, NumDims> Dimensions;


  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device) :
      Base(op, device) { }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalProduct(Scalar* buffer) const {
    if (this->m_j_size == 1) {
      this->template evalGemv<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
      return;
    }

    evalGemm<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  EIGEN_DEVICE_FUNC void evalGemm(Scalar* buffer) const {
    // columns in left side, rows in right side
    const Index k = this->m_k_size;

    // rows in left side
    const Index m = this->m_i_size;

    // columns in right side
    const Index n = this->m_j_size;

    // zero out the result buffer (which must be of size at least m * n * sizeof(Scalar)
    this->m_device.memset(buffer, 0, m * n * sizeof(Scalar));

    // define mr, nr, and all of my data mapper types
    typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
    typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;
    typedef typename internal::gebp_traits<LhsScalar, RhsScalar> Traits;

    const Index nr = Traits::nr;
    const Index mr = Traits::mr;

    typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
    typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

    const Index lhs_packet_size = internal::packet_traits<LhsScalar>::size;
    const Index rhs_packet_size = internal::packet_traits<RhsScalar>::size;

    typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                   LeftEvaluator, left_nocontract_t,
                                                   contract_t, lhs_packet_size,
                                                   lhs_inner_dim_contiguous,
                                                   false, Unaligned> LhsMapper;

    typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                   RightEvaluator, right_nocontract_t,
                                                   contract_t, rhs_packet_size,
                                                   rhs_inner_dim_contiguous,
                                                   rhs_inner_dim_reordered, Unaligned> RhsMapper;

    typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;

    // Declare GEBP packing and kernel structs
    internal::gemm_pack_lhs<LhsScalar, Index, typename LhsMapper::SubMapper, mr, Traits::LhsProgress, ColMajor> pack_lhs;
    internal::gemm_pack_rhs<RhsScalar, Index, typename RhsMapper::SubMapper, nr, ColMajor> pack_rhs;

    internal::gebp_kernel<LhsScalar, RhsScalar, Index, OutputMapper, mr, nr, false, false> gebp;

    // initialize data mappers
    LhsMapper lhs(this->m_leftImpl, this->m_left_nocontract_strides, this->m_i_strides,
                  this->m_left_contracting_strides, this->m_k_strides);

    RhsMapper rhs(this->m_rightImpl, this->m_right_nocontract_strides, this->m_j_strides,
                  this->m_right_contracting_strides, this->m_k_strides);

    OutputMapper output(buffer, m);

    typedef typename internal::gemm_blocking_space<ColMajor, LhsScalar, RhsScalar, Dynamic, Dynamic, Dynamic> BlockingType;

    // Sizes of the blocks to load in cache. See the Goto paper for details.
    BlockingType blocking(m, n, k, 1, true);
    const Index kc = blocking.kc();
    const Index mc = numext::mini(m, blocking.mc());
    const Index nc = numext::mini(n, blocking.nc());
    const Index sizeA = mc * kc;
    const Index sizeB = kc * nc;

    LhsScalar* blockA = static_cast<LhsScalar *>(this->m_device.allocate(sizeA * sizeof(LhsScalar)));
    RhsScalar* blockB = static_cast<RhsScalar *>(this->m_device.allocate(sizeB * sizeof(RhsScalar)));

    for(Index i2=0; i2<m; i2+=mc)
    {
      const Index actual_mc = numext::mini(i2+mc,m)-i2;
      for (Index k2 = 0; k2 < k; k2 += kc) {
        // make sure we don't overshoot right edge of left matrix, then pack vertical panel
        const Index actual_kc = numext::mini(k2 + kc, k) - k2;
        pack_lhs(blockA, lhs.getSubMapper(i2, k2), actual_kc, actual_mc, 0, 0);

        // series of horizontal blocks
        for (Index j2 = 0; j2 < n; j2 += nc) {
          // make sure we don't overshoot right edge of right matrix, then pack block
          const Index actual_nc = numext::mini(j2 + nc, n) - j2;
          pack_rhs(blockB, rhs.getSubMapper(k2, j2), actual_kc, actual_nc, 0, 0);

          // call gebp (matrix kernel)
          // The parameters here are copied from Eigen's GEMM implementation
          gebp(output.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, 1.0, -1, -1, 0, 0);
        }
      }
    }

    this->m_device.deallocate(blockA);
    this->m_device.deallocate(blockB);
  }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H
