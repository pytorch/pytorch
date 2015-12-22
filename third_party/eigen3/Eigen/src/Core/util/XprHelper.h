// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_XPRHELPER_H
#define EIGEN_XPRHELPER_H

// just a workaround because GCC seems to not really like empty structs
// FIXME: gcc 4.3 generates bad code when strict-aliasing is enabled
// so currently we simply disable this optimization for gcc 4.3
#if EIGEN_COMP_GNUC && !EIGEN_GNUC_AT(4,3)
  #define EIGEN_EMPTY_STRUCT_CTOR(X) \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X() {} \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X(const X& ) {}
#else
  #define EIGEN_EMPTY_STRUCT_CTOR(X)
#endif

namespace Eigen {

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE DenseIndex;

/**
 * \brief The Index type as used for the API.
 * \details To change this, \c \#define the preprocessor symbol \c EIGEN_DEFAULT_DENSE_INDEX_TYPE.
 * \sa \ref TopicPreprocessorDirectives, StorageIndex.
 */

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE Index;

namespace internal {

template<typename IndexDest, typename IndexSrc>
EIGEN_DEVICE_FUNC
inline IndexDest convert_index(const IndexSrc& idx) {
  // for sizeof(IndexDest)>=sizeof(IndexSrc) compilers should be able to optimize this away:
  eigen_internal_assert(idx <= NumTraits<IndexDest>::highest() && "Index value to big for target type");
  return IndexDest(idx);
}


//classes inheriting no_assignment_operator don't generate a default operator=.
class no_assignment_operator
{
  private:
    no_assignment_operator& operator=(const no_assignment_operator&);
};

/** \internal return the index type with the largest number of bits */
template<typename I1, typename I2>
struct promote_index_type
{
  typedef typename conditional<(sizeof(I1)<sizeof(I2)), I2, I1>::type type;
};

/** \internal If the template parameter Value is Dynamic, this class is just a wrapper around a T variable that
  * can be accessed using value() and setValue().
  * Otherwise, this class is an empty structure and value() just returns the template parameter Value.
  */
template<typename T, int Value> class variable_if_dynamic
{
  public:
    EIGEN_EMPTY_STRUCT_CTOR(variable_if_dynamic)
    EIGEN_DEVICE_FUNC explicit variable_if_dynamic(T v) { EIGEN_ONLY_USED_FOR_DEBUG(v); eigen_assert(v == T(Value)); }
    EIGEN_DEVICE_FUNC static T value() { return T(Value); }
    EIGEN_DEVICE_FUNC void setValue(T) {}
};

template<typename T> class variable_if_dynamic<T, Dynamic>
{
    T m_value;
    EIGEN_DEVICE_FUNC variable_if_dynamic() { eigen_assert(false); }
  public:
    EIGEN_DEVICE_FUNC explicit variable_if_dynamic(T value) : m_value(value) {}
    EIGEN_DEVICE_FUNC T value() const { return m_value; }
    EIGEN_DEVICE_FUNC void setValue(T value) { m_value = value; }
};

/** \internal like variable_if_dynamic but for DynamicIndex
  */
template<typename T, int Value> class variable_if_dynamicindex
{
  public:
    EIGEN_EMPTY_STRUCT_CTOR(variable_if_dynamicindex)
    EIGEN_DEVICE_FUNC explicit variable_if_dynamicindex(T v) { EIGEN_ONLY_USED_FOR_DEBUG(v); eigen_assert(v == T(Value)); }
    EIGEN_DEVICE_FUNC static T value() { return T(Value); }
    EIGEN_DEVICE_FUNC void setValue(T) {}
};

template<typename T> class variable_if_dynamicindex<T, DynamicIndex>
{
    T m_value;
    EIGEN_DEVICE_FUNC variable_if_dynamicindex() { eigen_assert(false); }
  public:
    EIGEN_DEVICE_FUNC explicit variable_if_dynamicindex(T value) : m_value(value) {}
    EIGEN_DEVICE_FUNC T value() const { return m_value; }
    EIGEN_DEVICE_FUNC void setValue(T value) { m_value = value; }
};

template<typename T> struct functor_traits
{
  enum
  {
    Cost = 10,
    PacketAccess = false,
    IsRepeatable = false
  };
};

template<typename T> struct packet_traits;

template<typename T> struct unpacket_traits
{
  typedef T type;
  typedef T half;
  enum
  {
    size = 1,
    alignment = 1
  };
};

template<int Size, typename PacketType,
         bool Stop = Size==Dynamic || (Size%unpacket_traits<PacketType>::size)==0 || is_same<PacketType,typename unpacket_traits<PacketType>::half>::value>
struct find_best_packet_helper;

template< int Size, typename PacketType>
struct find_best_packet_helper<Size,PacketType,true>
{
  typedef PacketType type;
};

template<int Size, typename PacketType>
struct find_best_packet_helper<Size,PacketType,false>
{
  typedef typename find_best_packet_helper<Size,typename unpacket_traits<PacketType>::half>::type type;
};

template<typename T, int Size>
struct find_best_packet
{
  typedef typename find_best_packet_helper<Size,typename packet_traits<T>::type>::type type;
};

#if EIGEN_MAX_STATIC_ALIGN_BYTES>0
template<int ArrayBytes, int AlignmentBytes,
         bool Match     =  bool((ArrayBytes%AlignmentBytes)==0),
         bool TryHalf   =  bool(AlignmentBytes>EIGEN_MIN_ALIGN_BYTES) >
struct compute_default_alignment_helper
{
  enum { value = 0 };
};

template<int ArrayBytes, int AlignmentBytes, bool TryHalf>
struct compute_default_alignment_helper<ArrayBytes, AlignmentBytes, true, TryHalf> // Match
{
  enum { value = AlignmentBytes };
};

template<int ArrayBytes, int AlignmentBytes>
struct compute_default_alignment_helper<ArrayBytes, AlignmentBytes, false, true> // Try-half
{
  // current packet too large, try with an half-packet
  enum { value = compute_default_alignment_helper<ArrayBytes, AlignmentBytes/2>::value };
};
#else
// If static alignment is disabled, no need to bother.
// This also avoids a division by zero in "bool Match =  bool((ArrayBytes%AlignmentBytes)==0)"
template<int ArrayBytes, int AlignmentBytes>
struct compute_default_alignment_helper
{
  enum { value = 0 };
};
#endif

template<typename T, int Size> struct compute_default_alignment {
  enum { value = compute_default_alignment_helper<Size*sizeof(T),EIGEN_MAX_STATIC_ALIGN_BYTES>::value };
};

template<typename T> struct compute_default_alignment<T,Dynamic> {
  enum { value = EIGEN_MAX_ALIGN_BYTES };
};

template<typename _Scalar, int _Rows, int _Cols,
         int _Options = AutoAlign |
                          ( (_Rows==1 && _Cols!=1) ? RowMajor
                          : (_Cols==1 && _Rows!=1) ? ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
         int _MaxRows = _Rows,
         int _MaxCols = _Cols
> class make_proper_matrix_type
{
    enum {
      IsColVector = _Cols==1 && _Rows!=1,
      IsRowVector = _Rows==1 && _Cols!=1,
      Options = IsColVector ? (_Options | ColMajor) & ~RowMajor
              : IsRowVector ? (_Options | RowMajor) & ~ColMajor
              : _Options
    };
  public:
    typedef Matrix<_Scalar, _Rows, _Cols, Options, _MaxRows, _MaxCols> type;
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class compute_matrix_flags
{
    enum { row_major_bit = Options&RowMajor ? RowMajorBit : 0 };
  public:
    // FIXME currently we still have to handle DirectAccessBit at the expression level to handle DenseCoeffsBase<>
    // and then propagate this information to the evaluator's flags.
    // However, I (Gael) think that DirectAccessBit should only matter at the evaluation stage.
    enum { ret = DirectAccessBit | LvalueBit | NestByRefBit | row_major_bit };
};

template<int _Rows, int _Cols> struct size_at_compile_time
{
  enum { ret = (_Rows==Dynamic || _Cols==Dynamic) ? Dynamic : _Rows * _Cols };
};

template<typename XprType> struct size_of_xpr_at_compile_time
{
  enum { ret = size_at_compile_time<traits<XprType>::RowsAtCompileTime,traits<XprType>::ColsAtCompileTime>::ret };
};

/* plain_matrix_type : the difference from eval is that plain_matrix_type is always a plain matrix type,
 * whereas eval is a const reference in the case of a matrix
 */

template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct plain_matrix_type;
template<typename T, typename BaseClassType, int Flags> struct plain_matrix_type_dense;
template<typename T> struct plain_matrix_type<T,Dense>
{
  typedef typename plain_matrix_type_dense<T,typename traits<T>::XprKind, traits<T>::Flags>::type type;
};
template<typename T> struct plain_matrix_type<T,DiagonalShape>
{
  typedef typename T::PlainObject type;
};

template<typename T, int Flags> struct plain_matrix_type_dense<T,MatrixXpr,Flags>
{
  typedef Matrix<typename traits<T>::Scalar,
                traits<T>::RowsAtCompileTime,
                traits<T>::ColsAtCompileTime,
                AutoAlign | (Flags&RowMajorBit ? RowMajor : ColMajor),
                traits<T>::MaxRowsAtCompileTime,
                traits<T>::MaxColsAtCompileTime
          > type;
};

template<typename T, int Flags> struct plain_matrix_type_dense<T,ArrayXpr,Flags>
{
  typedef Array<typename traits<T>::Scalar,
                traits<T>::RowsAtCompileTime,
                traits<T>::ColsAtCompileTime,
                AutoAlign | (Flags&RowMajorBit ? RowMajor : ColMajor),
                traits<T>::MaxRowsAtCompileTime,
                traits<T>::MaxColsAtCompileTime
          > type;
};

/* eval : the return type of eval(). For matrices, this is just a const reference
 * in order to avoid a useless copy
 */

template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct eval;

template<typename T> struct eval<T,Dense>
{
  typedef typename plain_matrix_type<T>::type type;
//   typedef typename T::PlainObject type;
//   typedef T::Matrix<typename traits<T>::Scalar,
//                 traits<T>::RowsAtCompileTime,
//                 traits<T>::ColsAtCompileTime,
//                 AutoAlign | (traits<T>::Flags&RowMajorBit ? RowMajor : ColMajor),
//                 traits<T>::MaxRowsAtCompileTime,
//                 traits<T>::MaxColsAtCompileTime
//           > type;
};

template<typename T> struct eval<T,DiagonalShape>
{
  typedef typename plain_matrix_type<T>::type type;
};

// for matrices, no need to evaluate, just use a const reference to avoid a useless copy
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct eval<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Dense>
{
  typedef const Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct eval<Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Dense>
{
  typedef const Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& type;
};


/* similar to plain_matrix_type, but using the evaluator's Flags */
template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct plain_object_eval;

template<typename T>
struct plain_object_eval<T,Dense>
{
  typedef typename plain_matrix_type_dense<T,typename traits<T>::XprKind, evaluator<T>::Flags>::type type;
};


/* plain_matrix_type_column_major : same as plain_matrix_type but guaranteed to be column-major
 */
template<typename T> struct plain_matrix_type_column_major
{
  enum { Rows = traits<T>::RowsAtCompileTime,
         Cols = traits<T>::ColsAtCompileTime,
         MaxRows = traits<T>::MaxRowsAtCompileTime,
         MaxCols = traits<T>::MaxColsAtCompileTime
  };
  typedef Matrix<typename traits<T>::Scalar,
                Rows,
                Cols,
                (MaxRows==1&&MaxCols!=1) ? RowMajor : ColMajor,
                MaxRows,
                MaxCols
          > type;
};

/* plain_matrix_type_row_major : same as plain_matrix_type but guaranteed to be row-major
 */
template<typename T> struct plain_matrix_type_row_major
{
  enum { Rows = traits<T>::RowsAtCompileTime,
         Cols = traits<T>::ColsAtCompileTime,
         MaxRows = traits<T>::MaxRowsAtCompileTime,
         MaxCols = traits<T>::MaxColsAtCompileTime
  };
  typedef Matrix<typename traits<T>::Scalar,
                Rows,
                Cols,
                (MaxCols==1&&MaxRows!=1) ? RowMajor : ColMajor,
                MaxRows,
                MaxCols
          > type;
};

/** \internal The reference selector for template expressions. The idea is that we don't
  * need to use references for expressions since they are light weight proxy
  * objects which should generate no copying overhead. */
template <typename T>
struct ref_selector
{
  typedef typename conditional<
    bool(traits<T>::Flags & NestByRefBit),
    T const&,
    const T
  >::type type;
  
  typedef typename conditional<
    bool(traits<T>::Flags & NestByRefBit),
    T &,
    T
  >::type non_const_type;
};

/** \internal Adds the const qualifier on the value-type of T2 if and only if T1 is a const type */
template<typename T1, typename T2>
struct transfer_constness
{
  typedef typename conditional<
    bool(internal::is_const<T1>::value),
    typename internal::add_const_on_value_type<T2>::type,
    T2
  >::type type;
};


// However, we still need a mechanism to detect whether an expression which is evaluated multiple time
// has to be evaluated into a temporary.
// That's the purpose of this new nested_eval helper:
/** \internal Determines how a given expression should be nested when evaluated multiple times.
  * For example, when you do a * (b+c), Eigen will determine how the expression b+c should be
  * evaluated into the bigger product expression. The choice is between nesting the expression b+c as-is, or
  * evaluating that expression b+c into a temporary variable d, and nest d so that the resulting expression is
  * a*d. Evaluating can be beneficial for example if every coefficient access in the resulting expression causes
  * many coefficient accesses in the nested expressions -- as is the case with matrix product for example.
  *
  * \param T the type of the expression being nested.
  * \param n the number of coefficient accesses in the nested expression for each coefficient access in the bigger expression.
  * \param PlainObject the type of the temporary if needed.
  */
template<typename T, int n, typename PlainObject = typename plain_object_eval<T>::type> struct nested_eval
{
  enum {
    ScalarReadCost = NumTraits<typename traits<T>::Scalar>::ReadCost,
    CoeffReadCost = evaluator<T>::CoeffReadCost,  // NOTE What if an evaluator evaluate itself into a tempory?
                                                  //      Then CoeffReadCost will be small (e.g., 1) but we still have to evaluate, especially if n>1.
                                                  //      This situation is already taken care by the EvalBeforeNestingBit flag, which is turned ON
                                                  //      for all evaluator creating a temporary. This flag is then propagated by the parent evaluators.
                                                  //      Another solution could be to count the number of temps?
    NAsInteger = n == Dynamic ? HugeCost : n,
    CostEval   = (NAsInteger+1) * ScalarReadCost + CoeffReadCost,
    CostNoEval = NAsInteger * CoeffReadCost
  };

  typedef typename conditional<
        ( (int(evaluator<T>::Flags) & EvalBeforeNestingBit) ||
          (int(CostEval) < int(CostNoEval)) ),
        PlainObject,
        typename ref_selector<T>::type
  >::type type;
};

template<typename T>
EIGEN_DEVICE_FUNC
inline T* const_cast_ptr(const T* ptr)
{
  return const_cast<T*>(ptr);
}

template<typename Derived, typename XprKind = typename traits<Derived>::XprKind>
struct dense_xpr_base
{
  /* dense_xpr_base should only ever be used on dense expressions, thus falling either into the MatrixXpr or into the ArrayXpr cases */
};

template<typename Derived>
struct dense_xpr_base<Derived, MatrixXpr>
{
  typedef MatrixBase<Derived> type;
};

template<typename Derived>
struct dense_xpr_base<Derived, ArrayXpr>
{
  typedef ArrayBase<Derived> type;
};

template<typename Derived, typename XprKind = typename traits<Derived>::XprKind, typename StorageKind = typename traits<Derived>::StorageKind>
struct generic_xpr_base;

template<typename Derived, typename XprKind>
struct generic_xpr_base<Derived, XprKind, Dense>
{
  typedef typename dense_xpr_base<Derived,XprKind>::type type;
};

/** \internal Helper base class to add a scalar multiple operator
  * overloads for complex types */
template<typename Derived, typename Scalar, typename OtherScalar, typename BaseType,
         bool EnableIt = !is_same<Scalar,OtherScalar>::value >
struct special_scalar_op_base : public BaseType
{
  // dummy operator* so that the
  // "using special_scalar_op_base::operator*" compiles
  struct dummy {};
  void operator*(dummy) const;
  void operator/(dummy) const;
};

template<typename Derived,typename Scalar,typename OtherScalar, typename BaseType>
struct special_scalar_op_base<Derived,Scalar,OtherScalar,BaseType,true>  : public BaseType
{
  const CwiseUnaryOp<scalar_multiple2_op<Scalar,OtherScalar>, Derived>
  operator*(const OtherScalar& scalar) const
  {
#ifdef EIGEN_SPECIAL_SCALAR_MULTIPLE_PLUGIN
    EIGEN_SPECIAL_SCALAR_MULTIPLE_PLUGIN
#endif
    return CwiseUnaryOp<scalar_multiple2_op<Scalar,OtherScalar>, Derived>
      (*static_cast<const Derived*>(this), scalar_multiple2_op<Scalar,OtherScalar>(scalar));
  }

  inline friend const CwiseUnaryOp<scalar_multiple2_op<Scalar,OtherScalar>, Derived>
  operator*(const OtherScalar& scalar, const Derived& matrix)
  {
#ifdef EIGEN_SPECIAL_SCALAR_MULTIPLE_PLUGIN
    EIGEN_SPECIAL_SCALAR_MULTIPLE_PLUGIN
#endif
    return static_cast<const special_scalar_op_base&>(matrix).operator*(scalar);
  }
  
  const CwiseUnaryOp<scalar_quotient2_op<Scalar,OtherScalar>, Derived>
  operator/(const OtherScalar& scalar) const
  {
#ifdef EIGEN_SPECIAL_SCALAR_MULTIPLE_PLUGIN
    EIGEN_SPECIAL_SCALAR_MULTIPLE_PLUGIN
#endif
    return CwiseUnaryOp<scalar_quotient2_op<Scalar,OtherScalar>, Derived>
      (*static_cast<const Derived*>(this), scalar_quotient2_op<Scalar,OtherScalar>(scalar));
  }
};

template<typename XprType, typename CastType> struct cast_return_type
{
  typedef typename XprType::Scalar CurrentScalarType;
  typedef typename remove_all<CastType>::type _CastType;
  typedef typename _CastType::Scalar NewScalarType;
  typedef typename conditional<is_same<CurrentScalarType,NewScalarType>::value,
                              const XprType&,CastType>::type type;
};

template <typename A, typename B> struct promote_storage_type;

template <typename A> struct promote_storage_type<A,A>
{
  typedef A ret;
};
template <typename A> struct promote_storage_type<A, const A>
{
  typedef A ret;
};
template <typename A> struct promote_storage_type<const A, A>
{
  typedef A ret;
};

/** \internal Specify the "storage kind" of applying a coefficient-wise
  * binary operations between two expressions of kinds A and B respectively.
  * The template parameter Functor permits to specialize the resulting storage kind wrt to
  * the functor.
  * The default rules are as follows:
  * \code
  * A     op A      -> A
  * A     op dense  -> dense
  * dense op B      -> dense
  * A     *  dense  -> A
  * dense *  B      -> B
  * \endcode
  */
template <typename A, typename B, typename Functor> struct cwise_promote_storage_type;

template <typename A, typename Functor>                   struct cwise_promote_storage_type<A,A,Functor>                                      { typedef A     ret; };
template <typename Functor>                               struct cwise_promote_storage_type<Dense,Dense,Functor>                              { typedef Dense ret; };
template <typename ScalarA, typename ScalarB>             struct cwise_promote_storage_type<Dense,Dense,scalar_product_op<ScalarA,ScalarB> >  { typedef Dense ret; };
template <typename A, typename Functor>                   struct cwise_promote_storage_type<A,Dense,Functor>                                  { typedef Dense ret; };
template <typename B, typename Functor>                   struct cwise_promote_storage_type<Dense,B,Functor>                                  { typedef Dense ret; };
template <typename A, typename ScalarA, typename ScalarB> struct cwise_promote_storage_type<A,Dense,scalar_product_op<ScalarA,ScalarB> >      { typedef A     ret; };
template <typename B, typename ScalarA, typename ScalarB> struct cwise_promote_storage_type<Dense,B,scalar_product_op<ScalarA,ScalarB> >      { typedef B     ret; };

/** \internal Specify the "storage kind" of multiplying an expression of kind A with kind B.
  * The template parameter ProductTag permits to specialize the resulting storage kind wrt to
  * some compile-time properties of the product: GemmProduct, GemvProduct, OuterProduct, InnerProduct.
  * The default rules are as follows:
  * \code
  *  K * K            -> K
  *  dense * K        -> dense
  *  K * dense        -> dense
  *  diag * K         -> K
  *  K * diag         -> K
  *  Perm * K         -> K
  * K * Perm          -> K
  * \endcode
  */
template <typename A, typename B, int ProductTag> struct product_promote_storage_type;

template <typename A, int ProductTag> struct product_promote_storage_type<A,                  A,                  ProductTag> { typedef A     ret;};
template <int ProductTag>             struct product_promote_storage_type<Dense,              Dense,              ProductTag> { typedef Dense ret;};
template <typename A, int ProductTag> struct product_promote_storage_type<A,                  Dense,              ProductTag> { typedef Dense ret; };
template <typename B, int ProductTag> struct product_promote_storage_type<Dense,              B,                  ProductTag> { typedef Dense ret; };

template <typename A, int ProductTag> struct product_promote_storage_type<A,                  DiagonalShape,      ProductTag> { typedef A ret; };
template <typename B, int ProductTag> struct product_promote_storage_type<DiagonalShape,      B,                  ProductTag> { typedef B ret; };
template <int ProductTag>             struct product_promote_storage_type<Dense,              DiagonalShape,      ProductTag> { typedef Dense ret; };
template <int ProductTag>             struct product_promote_storage_type<DiagonalShape,      Dense,              ProductTag> { typedef Dense ret; };

template <typename A, int ProductTag> struct product_promote_storage_type<A,                  PermutationStorage, ProductTag> { typedef A ret; };
template <typename B, int ProductTag> struct product_promote_storage_type<PermutationStorage, B,                  ProductTag> { typedef B ret; };
template <int ProductTag>             struct product_promote_storage_type<Dense,              PermutationStorage, ProductTag> { typedef Dense ret; };
template <int ProductTag>             struct product_promote_storage_type<PermutationStorage, Dense,              ProductTag> { typedef Dense ret; };

/** \internal gives the plain matrix or array type to store a row/column/diagonal of a matrix type.
  * \param Scalar optional parameter allowing to pass a different scalar type than the one of the MatrixType.
  */
template<typename ExpressionType, typename Scalar = typename ExpressionType::Scalar>
struct plain_row_type
{
  typedef Matrix<Scalar, 1, ExpressionType::ColsAtCompileTime,
                 ExpressionType::PlainObject::Options | RowMajor, 1, ExpressionType::MaxColsAtCompileTime> MatrixRowType;
  typedef Array<Scalar, 1, ExpressionType::ColsAtCompileTime,
                 ExpressionType::PlainObject::Options | RowMajor, 1, ExpressionType::MaxColsAtCompileTime> ArrayRowType;

  typedef typename conditional<
    is_same< typename traits<ExpressionType>::XprKind, MatrixXpr >::value,
    MatrixRowType,
    ArrayRowType 
  >::type type;
};

template<typename ExpressionType, typename Scalar = typename ExpressionType::Scalar>
struct plain_col_type
{
  typedef Matrix<Scalar, ExpressionType::RowsAtCompileTime, 1,
                 ExpressionType::PlainObject::Options & ~RowMajor, ExpressionType::MaxRowsAtCompileTime, 1> MatrixColType;
  typedef Array<Scalar, ExpressionType::RowsAtCompileTime, 1,
                 ExpressionType::PlainObject::Options & ~RowMajor, ExpressionType::MaxRowsAtCompileTime, 1> ArrayColType;

  typedef typename conditional<
    is_same< typename traits<ExpressionType>::XprKind, MatrixXpr >::value,
    MatrixColType,
    ArrayColType 
  >::type type;
};

template<typename ExpressionType, typename Scalar = typename ExpressionType::Scalar>
struct plain_diag_type
{
  enum { diag_size = EIGEN_SIZE_MIN_PREFER_DYNAMIC(ExpressionType::RowsAtCompileTime, ExpressionType::ColsAtCompileTime),
         max_diag_size = EIGEN_SIZE_MIN_PREFER_FIXED(ExpressionType::MaxRowsAtCompileTime, ExpressionType::MaxColsAtCompileTime)
  };
  typedef Matrix<Scalar, diag_size, 1, ExpressionType::PlainObject::Options & ~RowMajor, max_diag_size, 1> MatrixDiagType;
  typedef Array<Scalar, diag_size, 1, ExpressionType::PlainObject::Options & ~RowMajor, max_diag_size, 1> ArrayDiagType;

  typedef typename conditional<
    is_same< typename traits<ExpressionType>::XprKind, MatrixXpr >::value,
    MatrixDiagType,
    ArrayDiagType 
  >::type type;
};

template<typename ExpressionType>
struct is_lvalue
{
  enum { value = !bool(is_const<ExpressionType>::value) &&
                 bool(traits<ExpressionType>::Flags & LvalueBit) };
};

template<typename T> struct is_diagonal
{ enum { ret = false }; };

template<typename T> struct is_diagonal<DiagonalBase<T> >
{ enum { ret = true }; };

template<typename T> struct is_diagonal<DiagonalWrapper<T> >
{ enum { ret = true }; };

template<typename T, int S> struct is_diagonal<DiagonalMatrix<T,S> >
{ enum { ret = true }; };

template<typename S1, typename S2> struct glue_shapes;
template<> struct glue_shapes<DenseShape,TriangularShape> { typedef TriangularShape type;  };

template<typename T1, typename T2>
bool is_same_dense(const T1 &mat1, const T2 &mat2, typename enable_if<has_direct_access<T1>::ret&&has_direct_access<T2>::ret, T1>::type * = 0)
{
  return (mat1.data()==mat2.data()) && (mat1.innerStride()==mat2.innerStride()) && (mat1.outerStride()==mat2.outerStride());
}

template<typename T1, typename T2>
bool is_same_dense(const T1 &, const T2 &, typename enable_if<!(has_direct_access<T1>::ret&&has_direct_access<T2>::ret), T1>::type * = 0)
{
  return false;
}

template<typename T, typename U> struct is_same_or_void { enum { value = is_same<T,U>::value }; };
template<typename T> struct is_same_or_void<void,T>     { enum { value = 1 }; };
template<typename T> struct is_same_or_void<T,void>     { enum { value = 1 }; };
template<>           struct is_same_or_void<void,void>  { enum { value = 1 }; };

#ifdef EIGEN_DEBUG_ASSIGN
std::string demangle_traversal(int t)
{
  if(t==DefaultTraversal) return "DefaultTraversal";
  if(t==LinearTraversal) return "LinearTraversal";
  if(t==InnerVectorizedTraversal) return "InnerVectorizedTraversal";
  if(t==LinearVectorizedTraversal) return "LinearVectorizedTraversal";
  if(t==SliceVectorizedTraversal) return "SliceVectorizedTraversal";
  return "?";
}
std::string demangle_unrolling(int t)
{
  if(t==NoUnrolling) return "NoUnrolling";
  if(t==InnerUnrolling) return "InnerUnrolling";
  if(t==CompleteUnrolling) return "CompleteUnrolling";
  return "?";
}
std::string demangle_flags(int f)
{
  std::string res;
  if(f&RowMajorBit)                 res += " | RowMajor";
  if(f&PacketAccessBit)             res += " | Packet";
  if(f&LinearAccessBit)             res += " | Linear";
  if(f&LvalueBit)                   res += " | Lvalue";
  if(f&DirectAccessBit)             res += " | Direct";
  if(f&NestByRefBit)                res += " | NestByRef";
  if(f&NoPreferredStorageOrderBit)  res += " | NoPreferredStorageOrderBit";
  
  return res;
}
#endif

} // end namespace internal

// we require Lhs and Rhs to have the same scalar type. Currently there is no example of a binary functor
// that would take two operands of different types. If there were such an example, then this check should be
// moved to the BinaryOp functors, on a per-case basis. This would however require a change in the BinaryOp functors, as
// currently they take only one typename Scalar template parameter.
// It is tempting to always allow mixing different types but remember that this is often impossible in the vectorized paths.
// So allowing mixing different types gives very unexpected errors when enabling vectorization, when the user tries to
// add together a float matrix and a double matrix.
#define EIGEN_CHECK_BINARY_COMPATIBILIY(BINOP,LHS,RHS) \
  EIGEN_STATIC_ASSERT((internal::functor_is_product_like<BINOP>::ret \
                        ? int(internal::scalar_product_traits<LHS, RHS>::Defined) \
                        : int(internal::is_same_or_void<LHS, RHS>::value)), \
    YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    
} // end namespace Eigen

#endif // EIGEN_XPRHELPER_H
