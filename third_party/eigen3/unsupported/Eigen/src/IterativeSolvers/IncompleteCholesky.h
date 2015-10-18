// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INCOMPLETE_CHOlESKY_H
#define EIGEN_INCOMPLETE_CHOlESKY_H
#include "Eigen/src/IterativeLinearSolvers/IncompleteLUT.h" 
#include <Eigen/OrderingMethods>
#include <list>

namespace Eigen {  
/** 
 * \brief Modified Incomplete Cholesky with dual threshold
 * 
 * References : C-J. Lin and J. J. Moré, Incomplete Cholesky Factorizations with
 *              Limited memory, SIAM J. Sci. Comput.  21(1), pp. 24-45, 1999
 * 
 * \tparam _MatrixType The type of the sparse matrix. It should be a symmetric 
 *                     matrix. It is advised to give  a row-oriented sparse matrix 
 * \tparam _UpLo The triangular part of the matrix to reference. 
 * \tparam _OrderingType 
 */

template <typename Scalar, int _UpLo = Lower, typename _OrderingType = AMDOrdering<int> >
class IncompleteCholesky : public SparseSolverBase<IncompleteCholesky<Scalar,_UpLo,_OrderingType> >
{
  protected:
    typedef SparseSolverBase<IncompleteCholesky<Scalar,_UpLo,_OrderingType> > Base;
    using Base::m_isInitialized;
  public:
    typedef typename NumTraits<Scalar>::Real RealScalar; 
    typedef _OrderingType OrderingType;
    typedef typename OrderingType::PermutationType PermutationType;
    typedef typename PermutationType::StorageIndex StorageIndex; 
    typedef SparseMatrix<Scalar,ColMajor,StorageIndex> FactorType;
    typedef FactorType MatrixType;
    typedef Matrix<Scalar,Dynamic,1> VectorSx;
    typedef Matrix<RealScalar,Dynamic,1> VectorRx;
    typedef Matrix<StorageIndex,Dynamic, 1> VectorIx;
    typedef std::vector<std::list<StorageIndex> > VectorList; 
    enum { UpLo = _UpLo };
  public:
    IncompleteCholesky() : m_initialShift(1e-3),m_factorizationIsOk(false) {}
    
    template<typename MatrixType>
    IncompleteCholesky(const MatrixType& matrix) : m_initialShift(1e-3),m_factorizationIsOk(false)
    {
      compute(matrix);
    }
    
    Index rows() const { return m_L.rows(); }
    
    Index cols() const { return m_L.cols(); }
    

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "IncompleteLLT is not initialized.");
      return m_info;
    }
    
    /** 
     * \brief Set the initial shift parameter
     */
    void setInitialShift(RealScalar shift) { m_initialShift = shift; }
    
    /**
    * \brief Computes the fill reducing permutation vector. 
    */
    template<typename MatrixType>
    void analyzePattern(const MatrixType& mat)
    {
      OrderingType ord; 
      PermutationType pinv;
      ord(mat.template selfadjointView<UpLo>(), pinv); 
      if(pinv.size()>0) m_perm = pinv.inverse();
      else              m_perm.resize(0);
      m_analysisIsOk = true; 
    }
    
    template<typename MatrixType>
    void factorize(const MatrixType& amat);
    
    template<typename MatrixType>
    void compute(const MatrixType& matrix)
    {
      analyzePattern(matrix); 
      factorize(matrix);
    }
    
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
      eigen_assert(m_factorizationIsOk && "factorize() should be called first");
      if (m_perm.rows() == b.rows())  x = m_perm * b;
      else                            x = b;
      x = m_scale.asDiagonal() * x;
      x = m_L.template triangularView<Lower>().solve(x);
      x = m_L.adjoint().template triangularView<Upper>().solve(x);
      x = m_scale.asDiagonal() * x;
      if (m_perm.rows() == b.rows())
        x = m_perm.inverse() * x;
      
    }

  protected:
    FactorType m_L;              // The lower part stored in CSC
    VectorRx m_scale;            // The vector for scaling the matrix 
    RealScalar m_initialShift;   // The initial shift parameter
    bool m_analysisIsOk; 
    bool m_factorizationIsOk; 
    ComputationInfo m_info;
    PermutationType m_perm; 
    
  private:
    inline void updateList(Ref<const VectorIx> colPtr, Ref<VectorIx> rowIdx, Ref<VectorSx> vals, const Index& col, const Index& jk, VectorIx& firstElt, VectorList& listCol); 
}; 

template<typename Scalar, int _UpLo, typename OrderingType>
template<typename _MatrixType>
void IncompleteCholesky<Scalar,_UpLo, OrderingType>::factorize(const _MatrixType& mat)
{
  using std::sqrt;
  eigen_assert(m_analysisIsOk && "analyzePattern() should be called first"); 
    
  // Dropping strategy : Keep only the p largest elements per column, where p is the number of elements in the column of the original matrix. Other strategies will be added
  
  m_L.resize(mat.rows(), mat.cols());
  
  // Apply the fill-reducing permutation computed in analyzePattern()
  if (m_perm.rows() == mat.rows() ) // To detect the null permutation
  {
    // The temporary is needed to make sure that the diagonal entry is properly sorted
    FactorType tmp(mat.rows(), mat.cols());
    tmp = mat.template selfadjointView<_UpLo>().twistedBy(m_perm);
    m_L.template selfadjointView<Lower>() = tmp.template selfadjointView<Lower>();
  }
  else
  {
    m_L.template selfadjointView<Lower>() = mat.template selfadjointView<_UpLo>();
  }
  
  Index n = m_L.cols(); 
  Index nnz = m_L.nonZeros();
  Map<VectorSx> vals(m_L.valuePtr(), nnz);         //values
  Map<VectorIx> rowIdx(m_L.innerIndexPtr(), nnz);  //Row indices
  Map<VectorIx> colPtr( m_L.outerIndexPtr(), n+1); // Pointer to the beginning of each row
  VectorIx firstElt(n-1); // for each j, points to the next entry in vals that will be used in the factorization
  VectorList listCol(n);  // listCol(j) is a linked list of columns to update column j
  VectorSx col_vals(n);   // Store a  nonzero values in each column
  VectorIx col_irow(n);   // Row indices of nonzero elements in each column
  VectorIx col_pattern(n);
  col_pattern.fill(-1);
  StorageIndex col_nnz;
  
  
  // Computes the scaling factors 
  m_scale.resize(n);
  m_scale.setZero();
  for (Index j = 0; j < n; j++)
    for (Index k = colPtr[j]; k < colPtr[j+1]; k++)
    {
      m_scale(j) += numext::abs2(vals(k));
      if(rowIdx[k]!=j)
        m_scale(rowIdx[k]) += numext::abs2(vals(k));
    }
  
  m_scale = m_scale.cwiseSqrt().cwiseSqrt();
  
  // Scale and compute the shift for the matrix 
  RealScalar mindiag = NumTraits<RealScalar>::highest();
  for (Index j = 0; j < n; j++)
  {
    for (Index k = colPtr[j]; k < colPtr[j+1]; k++)
      vals[k] /= (m_scale(j)*m_scale(rowIdx[k]));
    eigen_internal_assert(rowIdx[colPtr[j]]==j && "IncompleteCholesky: only the lower triangular part must be stored");
    mindiag = numext::mini(numext::real(vals[colPtr[j]]), mindiag);
  }
  
  RealScalar shift = 0;
  if(mindiag <= RealScalar(0.))
    shift = m_initialShift - mindiag;

  // Apply the shift to the diagonal elements of the matrix
  for (Index j = 0; j < n; j++)
    vals[colPtr[j]] += shift;
  
  // jki version of the Cholesky factorization 
  for (Index j=0; j < n; ++j)
  {  
    // Left-looking factorization of the j-th column
    // First, load the j-th column into col_vals 
    Scalar diag = vals[colPtr[j]];  // It is assumed that only the lower part is stored
    col_nnz = 0;
    for (Index i = colPtr[j] + 1; i < colPtr[j+1]; i++)
    {
      StorageIndex l = rowIdx[i];
      col_vals(col_nnz) = vals[i];
      col_irow(col_nnz) = l;
      col_pattern(l) = col_nnz;
      col_nnz++;
    }
    {
      typename std::list<StorageIndex>::iterator k; 
      // Browse all previous columns that will update column j
      for(k = listCol[j].begin(); k != listCol[j].end(); k++) 
      {
        Index jk = firstElt(*k); // First element to use in the column 
        eigen_internal_assert(rowIdx[jk]==j);
        Scalar v_j_jk = numext::conj(vals[jk]);
        
        jk += 1; 
        for (Index i = jk; i < colPtr[*k+1]; i++)
        {
          StorageIndex l = rowIdx[i];
          if(col_pattern[l]<0)
          {
            col_vals(col_nnz) = vals[i] * v_j_jk;
            col_irow[col_nnz] = l;
            col_pattern(l) = col_nnz;
            col_nnz++;
          }
          else
            col_vals(col_pattern[l]) -= vals[i] * v_j_jk;
        }
        updateList(colPtr,rowIdx,vals, *k, jk, firstElt, listCol);
      }
    }
    
    // Scale the current column
    if(numext::real(diag) <= 0) 
    {
      std::cerr << "\nNegative diagonal during Incomplete factorization at position " << j << " (value = " << diag << ")\n";
      m_info = NumericalIssue; 
      return; 
    }
    
    RealScalar rdiag = sqrt(numext::real(diag));
    vals[colPtr[j]] = rdiag;
    for (Index k = 0; k<col_nnz; ++k)
    {
      Index i = col_irow[k];
      //Scale
      col_vals(k) /= rdiag;
      //Update the remaining diagonals with col_vals
      vals[colPtr[i]] -= numext::abs2(col_vals(k));
    }
    // Select the largest p elements
    // p is the original number of elements in the column (without the diagonal)
    Index p = colPtr[j+1] - colPtr[j] - 1 ; 
    Ref<VectorSx> cvals = col_vals.head(col_nnz);
    Ref<VectorIx> cirow = col_irow.head(col_nnz);
    internal::QuickSplit(cvals,cirow, p); 
    // Insert the largest p elements in the matrix
    Index cpt = 0; 
    for (Index i = colPtr[j]+1; i < colPtr[j+1]; i++)
    {
      vals[i] = col_vals(cpt); 
      rowIdx[i] = col_irow(cpt);
      // restore col_pattern:
      col_pattern(col_irow(cpt)) = -1;
      cpt++; 
    }
    // Get the first smallest row index and put it after the diagonal element
    Index jk = colPtr(j)+1;
    updateList(colPtr,rowIdx,vals,j,jk,firstElt,listCol); 
  }
  m_factorizationIsOk = true; 
  m_isInitialized = true;
  m_info = Success; 
}

template<typename Scalar, int _UpLo, typename OrderingType>
inline void IncompleteCholesky<Scalar,_UpLo, OrderingType>::updateList(Ref<const VectorIx> colPtr, Ref<VectorIx> rowIdx, Ref<VectorSx> vals, const Index& col, const Index& jk, VectorIx& firstElt, VectorList& listCol)
{
  if (jk < colPtr(col+1) )
  {
    Index p = colPtr(col+1) - jk;
    Index minpos; 
    rowIdx.segment(jk,p).minCoeff(&minpos);
    minpos += jk;
    if (rowIdx(minpos) != rowIdx(jk))
    {
      //Swap
      std::swap(rowIdx(jk),rowIdx(minpos));
      std::swap(vals(jk),vals(minpos));
    }
    firstElt(col) = internal::convert_index<StorageIndex,Index>(jk);
    listCol[rowIdx(jk)].push_back(internal::convert_index<StorageIndex,Index>(col));
  }
}

} // end namespace Eigen 

#endif
