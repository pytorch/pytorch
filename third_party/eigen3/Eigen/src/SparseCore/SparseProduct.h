// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEPRODUCT_H
#define EIGEN_SPARSEPRODUCT_H

namespace Eigen { 

/** \returns an expression of the product of two sparse matrices.
  * By default a conservative product preserving the symbolic non zeros is performed.
  * The automatic pruning of the small values can be achieved by calling the pruned() function
  * in which case a totally different product algorithm is employed:
  * \code
  * C = (A*B).pruned();             // supress numerical zeros (exact)
  * C = (A*B).pruned(ref);
  * C = (A*B).pruned(ref,epsilon);
  * \endcode
  * where \c ref is a meaningful non zero reference value.
  * */
template<typename Derived>
template<typename OtherDerived>
inline const Product<Derived,OtherDerived>
SparseMatrixBase<Derived>::operator*(const SparseMatrixBase<OtherDerived> &other) const
{
  return Product<Derived,OtherDerived>(derived(), other.derived());
}

namespace internal {

// sparse * sparse
template<typename Lhs, typename Rhs, int ProductType>
struct generic_product_impl<Lhs, Rhs, SparseShape, SparseShape, ProductType>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs)
  {
    typedef typename nested_eval<Lhs,Dynamic>::type LhsNested;
    typedef typename nested_eval<Rhs,Dynamic>::type RhsNested;
    LhsNested lhsNested(lhs);
    RhsNested rhsNested(rhs);
    internal::conservative_sparse_sparse_product_selector<typename remove_all<LhsNested>::type,
                                                          typename remove_all<RhsNested>::type, Dest>::run(lhsNested,rhsNested,dst);
  }
};

// sparse * sparse-triangular
template<typename Lhs, typename Rhs, int ProductType>
struct generic_product_impl<Lhs, Rhs, SparseShape, SparseTriangularShape, ProductType>
 : public generic_product_impl<Lhs, Rhs, SparseShape, SparseShape, ProductType>
{};

// sparse-triangular * sparse
template<typename Lhs, typename Rhs, int ProductType>
struct generic_product_impl<Lhs, Rhs, SparseTriangularShape, SparseShape, ProductType>
 : public generic_product_impl<Lhs, Rhs, SparseShape, SparseShape, ProductType>
{};

template<typename Lhs, typename Rhs, int Options>
struct evaluator<SparseView<Product<Lhs, Rhs, Options> > > 
 : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>
{
  typedef SparseView<Product<Lhs, Rhs, Options> > XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef evaluator<PlainObject> Base;
  
  explicit evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    using std::abs;
    ::new (static_cast<Base*>(this)) Base(m_result);
    typedef typename nested_eval<Lhs,Dynamic>::type LhsNested;
    typedef typename nested_eval<Rhs,Dynamic>::type RhsNested;
    LhsNested lhsNested(xpr.nestedExpression().lhs());
    RhsNested rhsNested(xpr.nestedExpression().rhs());
    
    internal::sparse_sparse_product_with_pruning_selector<typename remove_all<LhsNested>::type,
                                                          typename remove_all<RhsNested>::type, PlainObject>::run(lhsNested,rhsNested,m_result,
                                                                                                                  abs(xpr.reference())*xpr.epsilon());
  }
  
protected:  
  PlainObject m_result;
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SPARSEPRODUCT_H
