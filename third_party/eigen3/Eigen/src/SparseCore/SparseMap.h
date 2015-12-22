// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_MAP_H
#define EIGEN_SPARSE_MAP_H

namespace Eigen {

namespace internal {

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
struct traits<Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
  : public traits<SparseMatrix<MatScalar,MatOptions,MatIndex> >
{
  typedef SparseMatrix<MatScalar,MatOptions,MatIndex> PlainObjectType;
  typedef traits<PlainObjectType> TraitsBase;
  enum {
    Flags = TraitsBase::Flags & (~NestByRefBit)
  };
};

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
struct traits<Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
  : public traits<SparseMatrix<MatScalar,MatOptions,MatIndex> >
{
  typedef SparseMatrix<MatScalar,MatOptions,MatIndex> PlainObjectType;
  typedef traits<PlainObjectType> TraitsBase;
  enum {
    Flags = TraitsBase::Flags & (~ (NestByRefBit | LvalueBit))
  };
};

} // end namespace internal
  
template<typename Derived,
         int Level = internal::accessors_level<Derived>::has_write_access ? WriteAccessors : ReadOnlyAccessors
> class SparseMapBase;

template<typename Derived>
class SparseMapBase<Derived,ReadOnlyAccessors>
  : public SparseCompressedBase<Derived>
{
  public:
    typedef SparseCompressedBase<Derived> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::StorageIndex StorageIndex;
    enum { IsRowMajor = Base::IsRowMajor };
    using Base::operator=;
  protected:
    
    typedef typename internal::conditional<
                         bool(internal::is_lvalue<Derived>::value),
                         Scalar *, const Scalar *>::type ScalarPointer;
    typedef typename internal::conditional<
                         bool(internal::is_lvalue<Derived>::value),
                         StorageIndex *, const StorageIndex *>::type IndexPointer;

    Index   m_outerSize;
    Index   m_innerSize;
    Array<StorageIndex,2,1>  m_zero_nnz;
    IndexPointer  m_outerIndex;
    IndexPointer  m_innerIndices;
    ScalarPointer m_values;
    IndexPointer  m_innerNonZeros;

  public:

    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }
    inline Index innerSize() const { return m_innerSize; }
    inline Index outerSize() const { return m_outerSize; }
    inline Index nonZeros() const { return m_zero_nnz[1]; }
    
    bool isCompressed() const { return m_innerNonZeros==0; }

    //----------------------------------------
    // direct access interface
    inline const Scalar* valuePtr() const { return m_values; }
    inline const StorageIndex* innerIndexPtr() const { return m_innerIndices; }
    inline const StorageIndex* outerIndexPtr() const { return m_outerIndex; }
    inline const StorageIndex* innerNonZeroPtr() const { return m_innerNonZeros; }
    //----------------------------------------

    inline Scalar coeff(Index row, Index col) const
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = m_outerIndex[outer];
      Index end = isCompressed() ? m_outerIndex[outer+1] : start + m_innerNonZeros[outer];
      if (start==end)
        return Scalar(0);
      else if (end>0 && inner==m_innerIndices[end-1])
        return m_values[end-1];
      // ^^  optimization: let's first check if it is the last coefficient
      // (very common in high level algorithms)

      const StorageIndex* r = std::lower_bound(&m_innerIndices[start],&m_innerIndices[end-1],inner);
      const Index id = r-&m_innerIndices[0];
      return ((*r==inner) && (id<end)) ? m_values[id] : Scalar(0);
    }

    inline SparseMapBase(Index rows, Index cols, Index nnz, IndexPointer outerIndexPtr, IndexPointer innerIndexPtr,
                              ScalarPointer valuePtr, IndexPointer innerNonZerosPtr = 0)
      : m_outerSize(IsRowMajor?rows:cols), m_innerSize(IsRowMajor?cols:rows), m_zero_nnz(0,internal::convert_index<StorageIndex>(nnz)), m_outerIndex(outerIndexPtr),
        m_innerIndices(innerIndexPtr), m_values(valuePtr), m_innerNonZeros(innerNonZerosPtr)
    {}

    // for vectors
    inline SparseMapBase(Index size, Index nnz, IndexPointer innerIndexPtr, ScalarPointer valuePtr)
      : m_outerSize(1), m_innerSize(size), m_zero_nnz(0,internal::convert_index<StorageIndex>(nnz)), m_outerIndex(m_zero_nnz.data()),
        m_innerIndices(innerIndexPtr), m_values(valuePtr), m_innerNonZeros(0)
    {}

    /** Empty destructor */
    inline ~SparseMapBase() {}

  protected:
    inline SparseMapBase() {}
};

template<typename Derived>
class SparseMapBase<Derived,WriteAccessors>
  : public SparseMapBase<Derived,ReadOnlyAccessors>
{
    typedef MapBase<Derived, ReadOnlyAccessors> ReadOnlyMapBase;
    
  public:
    typedef SparseMapBase<Derived, ReadOnlyAccessors> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::StorageIndex StorageIndex;
    enum { IsRowMajor = Base::IsRowMajor };
    
    using Base::operator=;

  public:
    
    //----------------------------------------
    // direct access interface
    using Base::valuePtr;
    using Base::innerIndexPtr;
    using Base::outerIndexPtr;
    using Base::innerNonZeroPtr;
    inline Scalar* valuePtr()       { return Base::m_values; }
    inline StorageIndex* innerIndexPtr()   { return Base::m_innerIndices; }
    inline StorageIndex* outerIndexPtr()   { return Base::m_outerIndex; }
    inline StorageIndex* innerNonZeroPtr() { return Base::m_innerNonZeros; }
    //----------------------------------------

    inline Scalar& coeffRef(Index row, Index col)
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = Base::m_outerIndex[outer];
      Index end = Base::isCompressed() ? Base::m_outerIndex[outer+1] : start + Base::m_innerNonZeros[outer];
      eigen_assert(end>=start && "you probably called coeffRef on a non finalized matrix");
      eigen_assert(end>start && "coeffRef cannot be called on a zero coefficient");
      Index* r = std::lower_bound(&Base::m_innerIndices[start],&Base::m_innerIndices[end],inner);
      const Index id = r - &Base::m_innerIndices[0];
      eigen_assert((*r==inner) && (id<end) && "coeffRef cannot be called on a zero coefficient");
      return const_cast<Scalar*>(Base::m_values)[id];
    }
    
    inline SparseMapBase(Index rows, Index cols, Index nnz, StorageIndex* outerIndexPtr, StorageIndex* innerIndexPtr,
                              Scalar* valuePtr, StorageIndex* innerNonZerosPtr = 0)
      : Base(rows, cols, nnz, outerIndexPtr, innerIndexPtr, valuePtr, innerNonZerosPtr)
    {}

    // for vectors
    inline SparseMapBase(Index size, Index nnz, StorageIndex* innerIndexPtr, Scalar* valuePtr)
      : Base(size, nnz, innerIndexPtr, valuePtr)
    {}

    /** Empty destructor */
    inline ~SparseMapBase() {}

  protected:
    inline SparseMapBase() {}
};

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
class Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType>
  : public SparseMapBase<Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
{
  public:
    typedef SparseMapBase<Map> Base;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Map)
    enum { IsRowMajor = Base::IsRowMajor };

  public:

    inline Map(Index rows, Index cols, Index nnz, StorageIndex* outerIndexPtr,
               StorageIndex* innerIndexPtr, Scalar* valuePtr, StorageIndex* innerNonZerosPtr = 0)
      : Base(rows, cols, nnz, outerIndexPtr, innerIndexPtr, valuePtr, innerNonZerosPtr)
    {}

    /** Empty destructor */
    inline ~Map() {}
};

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
class Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType>
  : public SparseMapBase<Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
{
  public:
    typedef SparseMapBase<Map> Base;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Map)
    enum { IsRowMajor = Base::IsRowMajor };

  public:

    inline Map(Index rows, Index cols, Index nnz, const StorageIndex* outerIndexPtr,
               const StorageIndex* innerIndexPtr, const Scalar* valuePtr, const StorageIndex* innerNonZerosPtr = 0)
      : Base(rows, cols, nnz, outerIndexPtr, innerIndexPtr, valuePtr, innerNonZerosPtr)
    {}

    /** Empty destructor */
    inline ~Map() {}
};

namespace internal {

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
struct evaluator<Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
  : evaluator<SparseCompressedBase<Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> > >
{
  typedef evaluator<SparseCompressedBase<Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> > > Base;
  typedef Map<SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> XprType;  
  evaluator() : Base() {}
  explicit evaluator(const XprType &mat) : Base(mat) {}
};

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
struct evaluator<Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
  : evaluator<SparseCompressedBase<Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> > >
{
  typedef evaluator<SparseCompressedBase<Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> > > Base;
  typedef Map<const SparseMatrix<MatScalar,MatOptions,MatIndex>, Options, StrideType> XprType;  
  evaluator() : Base() {}
  explicit evaluator(const XprType &mat) : Base(mat) {}
};

}

} // end namespace Eigen

#endif // EIGEN_SPARSE_MAP_H
