

typedef CwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived> AbsReturnType;
typedef CwiseUnaryOp<internal::scalar_arg_op<Scalar>, const Derived> ArgReturnType;
typedef CwiseUnaryOp<internal::scalar_abs2_op<Scalar>, const Derived> Abs2ReturnType;
typedef CwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived> SqrtReturnType;
typedef CwiseUnaryOp<internal::scalar_rsqrt_op<Scalar>, const Derived> RsqrtReturnType;
typedef CwiseUnaryOp<internal::scalar_sign_op<Scalar>, const Derived> SignReturnType;
typedef CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived> InverseReturnType;
typedef CwiseUnaryOp<internal::scalar_boolean_not_op<Scalar>, const Derived> BooleanNotReturnType;

typedef CwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived> ExpReturnType;
typedef CwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived> LogReturnType;
typedef CwiseUnaryOp<internal::scalar_log10_op<Scalar>, const Derived> Log10ReturnType;
typedef CwiseUnaryOp<internal::scalar_cos_op<Scalar>, const Derived> CosReturnType;
typedef CwiseUnaryOp<internal::scalar_sin_op<Scalar>, const Derived> SinReturnType;
typedef CwiseUnaryOp<internal::scalar_tan_op<Scalar>, const Derived> TanReturnType;
typedef CwiseUnaryOp<internal::scalar_acos_op<Scalar>, const Derived> AcosReturnType;
typedef CwiseUnaryOp<internal::scalar_asin_op<Scalar>, const Derived> AsinReturnType;
typedef CwiseUnaryOp<internal::scalar_atan_op<Scalar>, const Derived> AtanReturnType;
typedef CwiseUnaryOp<internal::scalar_tanh_op<Scalar>, const Derived> TanhReturnType;
typedef CwiseUnaryOp<internal::scalar_sinh_op<Scalar>, const Derived> SinhReturnType;
typedef CwiseUnaryOp<internal::scalar_cosh_op<Scalar>, const Derived> CoshReturnType;
typedef CwiseUnaryOp<internal::scalar_lgamma_op<Scalar>, const Derived> LgammaReturnType;
typedef CwiseUnaryOp<internal::scalar_erf_op<Scalar>, const Derived> ErfReturnType;
typedef CwiseUnaryOp<internal::scalar_erfc_op<Scalar>, const Derived> ErfcReturnType;
typedef CwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived> PowReturnType;
typedef CwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived> SquareReturnType;
typedef CwiseUnaryOp<internal::scalar_cube_op<Scalar>, const Derived> CubeReturnType;
typedef CwiseUnaryOp<internal::scalar_round_op<Scalar>, const Derived> RoundReturnType;
typedef CwiseUnaryOp<internal::scalar_floor_op<Scalar>, const Derived> FloorReturnType;
typedef CwiseUnaryOp<internal::scalar_ceil_op<Scalar>, const Derived> CeilReturnType;
typedef CwiseUnaryOp<internal::scalar_isnan_op<Scalar>, const Derived> IsNaNReturnType;
typedef CwiseUnaryOp<internal::scalar_isinf_op<Scalar>, const Derived> IsInfReturnType;
typedef CwiseUnaryOp<internal::scalar_isfinite_op<Scalar>, const Derived> IsFiniteReturnType;

/** \returns an expression of the coefficient-wise absolute value of \c *this
  *
  * Example: \include Cwise_abs.cpp
  * Output: \verbinclude Cwise_abs.out
  *
  * \sa abs2()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const AbsReturnType
abs() const
{
  return AbsReturnType(derived());
}

/** \returns an expression of the coefficient-wise phase angle of \c *this
  *
  * Example: \include Cwise_arg.cpp
  * Output: \verbinclude Cwise_arg.out
  *
  * \sa abs()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const ArgReturnType
arg() const
{
  return ArgReturnType(derived());
}

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  *
  * Example: \include Cwise_abs2.cpp
  * Output: \verbinclude Cwise_abs2.out
  *
  * \sa abs(), square()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const Abs2ReturnType
abs2() const
{
  return Abs2ReturnType(derived());
}

/** \returns an expression of the coefficient-wise exponential of *this.
  *
  * This function computes the coefficient-wise exponential. The function MatrixBase::exp() in the
  * unsupported module MatrixFunctions computes the matrix exponential.
  *
  * Example: \include Cwise_exp.cpp
  * Output: \verbinclude Cwise_exp.out
  *
  * \sa pow(), log(), sin(), cos()
  */
EIGEN_DEVICE_FUNC
inline const ExpReturnType
exp() const
{
  return ExpReturnType(derived());
}

/** \returns an expression of the coefficient-wise logarithm of *this.
  *
  * This function computes the coefficient-wise logarithm. The function MatrixBase::log() in the
  * unsupported module MatrixFunctions computes the matrix logarithm.
  *
  * Example: \include Cwise_log.cpp
  * Output: \verbinclude Cwise_log.out
  *
  * \sa exp()
  */
EIGEN_DEVICE_FUNC
inline const LogReturnType
log() const
{
  return LogReturnType(derived());
}

/** \returns an expression of the coefficient-wise base-10 logarithm of *this.
  *
  * This function computes the coefficient-wise base-10 logarithm.
  *
  * Example: \include Cwise_log10.cpp
  * Output: \verbinclude Cwise_log10.out
  *
  * \sa log()
  */
EIGEN_DEVICE_FUNC
inline const Log10ReturnType
log10() const
{
  return Log10ReturnType(derived());
}

/** \returns an expression of the coefficient-wise square root of *this.
  *
  * This function computes the coefficient-wise square root. The function MatrixBase::sqrt() in the
  * unsupported module MatrixFunctions computes the matrix square root.
  *
  * Example: \include Cwise_sqrt.cpp
  * Output: \verbinclude Cwise_sqrt.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const SqrtReturnType
sqrt() const
{
  return SqrtReturnType(derived());
}

/** \returns an expression of the coefficient-wise inverse square root of *this.
  *
  * This function computes the coefficient-wise inverse square root.
  *
  * Example: \include Cwise_sqrt.cpp
  * Output: \verbinclude Cwise_sqrt.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const RsqrtReturnType
rsqrt() const
{
  return RsqrtReturnType(derived());
}

/** \returns an expression of the coefficient-wise signum of *this.
  *
  * This function computes the coefficient-wise signum.
  *
  * Example: \include Cwise_sign.cpp
  * Output: \verbinclude Cwise_sign.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const SignReturnType
sign() const
{
  return SignReturnType(derived());
}


/** \returns an expression of the coefficient-wise cosine of *this.
  *
  * This function computes the coefficient-wise cosine. The function MatrixBase::cos() in the
  * unsupported module MatrixFunctions computes the matrix cosine.
  *
  * Example: \include Cwise_cos.cpp
  * Output: \verbinclude Cwise_cos.out
  *
  * \sa sin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const CosReturnType
cos() const
{
  return CosReturnType(derived());
}


/** \returns an expression of the coefficient-wise sine of *this.
  *
  * This function computes the coefficient-wise sine. The function MatrixBase::sin() in the
  * unsupported module MatrixFunctions computes the matrix sine.
  *
  * Example: \include Cwise_sin.cpp
  * Output: \verbinclude Cwise_sin.out
  *
  * \sa cos(), asin()
  */
EIGEN_DEVICE_FUNC
inline const SinReturnType
sin() const
{
  return SinReturnType(derived());
}

/** \returns an expression of the coefficient-wise tan of *this.
  *
  * Example: \include Cwise_tan.cpp
  * Output: \verbinclude Cwise_tan.out
  *
  * \sa cos(), sin()
  */
EIGEN_DEVICE_FUNC
inline const TanReturnType
tan() const
{
  return TanReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc tan of *this.
  *
  * Example: \include Cwise_atan.cpp
  * Output: \verbinclude Cwise_atan.out
  *
  * \sa tan(), asin(), acos()
  */
inline const AtanReturnType
atan() const
{
  return AtanReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc cosine of *this.
  *
  * Example: \include Cwise_acos.cpp
  * Output: \verbinclude Cwise_acos.out
  *
  * \sa cos(), asin()
  */
EIGEN_DEVICE_FUNC
inline const AcosReturnType
acos() const
{
  return AcosReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc sine of *this.
  *
  * Example: \include Cwise_asin.cpp
  * Output: \verbinclude Cwise_asin.out
  *
  * \sa sin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const AsinReturnType
asin() const
{
  return AsinReturnType(derived());
}

/** \returns an expression of the coefficient-wise hyperbolic tan of *this.
  *
  * Example: \include Cwise_tanh.cpp
  * Output: \verbinclude Cwise_tanh.out
  *
  * \sa tan(), sinh(), cosh()
  */
inline const TanhReturnType
tanh() const
{
  return TanhReturnType(derived());
}

/** \returns an expression of the coefficient-wise hyperbolic sin of *this.
  *
  * Example: \include Cwise_sinh.cpp
  * Output: \verbinclude Cwise_sinh.out
  *
  * \sa sin(), tanh(), cosh()
  */
inline const SinhReturnType
sinh() const
{
  return SinhReturnType(derived());
}

/** \returns an expression of the coefficient-wise hyperbolic cos of *this.
  *
  * Example: \include Cwise_cosh.cpp
  * Output: \verbinclude Cwise_cosh.out
  *
  * \sa tan(), sinh(), cosh()
  */
inline const CoshReturnType
cosh() const
{
  return CoshReturnType(derived());
}

/** \returns an expression of the coefficient-wise ln(|gamma(*this)|).
 *
 * Example: \include Cwise_lgamma.cpp
 * Output: \verbinclude Cwise_lgamma.out
 *
 * \sa cos(), sin(), tan()
 */
inline const LgammaReturnType
lgamma() const
{
  return LgammaReturnType(derived());
}

/** \returns an expression of the coefficient-wise Gauss error
 * function of *this.
 *
 * Example: \include Cwise_erf.cpp
 * Output: \verbinclude Cwise_erf.out
 *
 * \sa cos(), sin(), tan()
 */
inline const ErfReturnType
erf() const
{
  return ErfReturnType(derived());
}

/** \returns an expression of the coefficient-wise Complementary error
 * function of *this.
 *
 * Example: \include Cwise_erfc.cpp
 * Output: \verbinclude Cwise_erfc.out
 *
 * \sa cos(), sin(), tan()
 */
inline const ErfcReturnType
erfc() const
{
  return ErfcReturnType(derived());
}

/** \returns an expression of the coefficient-wise power of *this to the given exponent.
  *
  * This function computes the coefficient-wise power. The function MatrixBase::pow() in the
  * unsupported module MatrixFunctions computes the matrix power.
  *
  * Example: \include Cwise_pow.cpp
  * Output: \verbinclude Cwise_pow.out
  *
  * \sa exp(), log()
  */
EIGEN_DEVICE_FUNC
inline const PowReturnType
pow(const Scalar& exponent) const
{
  return PowReturnType(derived(), internal::scalar_pow_op<Scalar>(exponent));
}


/** \returns an expression of the coefficient-wise inverse of *this.
  *
  * Example: \include Cwise_inverse.cpp
  * Output: \verbinclude Cwise_inverse.out
  *
  * \sa operator/(), operator*()
  */
EIGEN_DEVICE_FUNC
inline const InverseReturnType
inverse() const
{
  return InverseReturnType(derived());
}

/** \returns an expression of the coefficient-wise square of *this.
  *
  * Example: \include Cwise_square.cpp
  * Output: \verbinclude Cwise_square.out
  *
  * \sa operator/(), operator*(), abs2()
  */
EIGEN_DEVICE_FUNC
inline const SquareReturnType
square() const
{
  return SquareReturnType(derived());
}

/** \returns an expression of the coefficient-wise cube of *this.
  *
  * Example: \include Cwise_cube.cpp
  * Output: \verbinclude Cwise_cube.out
  *
  * \sa square(), pow()
  */
EIGEN_DEVICE_FUNC
inline const CubeReturnType
cube() const
{
  return CubeReturnType(derived());
}

/** \returns an expression of the coefficient-wise round of *this.
  *
  * Example: \include Cwise_round.cpp
  * Output: \verbinclude Cwise_round.out
  *
  * \sa ceil(), floor()
  */
inline const RoundReturnType
round() const
{
  return RoundReturnType(derived());
}

/** \returns an expression of the coefficient-wise floor of *this.
  *
  * Example: \include Cwise_floor.cpp
  * Output: \verbinclude Cwise_floor.out
  *
  * \sa ceil(), round()
  */
inline const FloorReturnType
floor() const
{
  return FloorReturnType(derived());
}

/** \returns an expression of the coefficient-wise ceil of *this.
  *
  * Example: \include Cwise_ceil.cpp
  * Output: \verbinclude Cwise_ceil.out
  *
  * \sa floor(), round()
  */
inline const CeilReturnType
ceil() const
{
  return CeilReturnType(derived());
}

/** \returns an expression of the coefficient-wise isnan of *this.
  *
  * Example: \include Cwise_isNaN.cpp
  * Output: \verbinclude Cwise_isNaN.out
  *
  * \sa isfinite(), isinf()
  */
inline const IsNaNReturnType
isNaN() const
{
  return IsNaNReturnType(derived());
}

/** \returns an expression of the coefficient-wise isinf of *this.
  *
  * Example: \include Cwise_isInf.cpp
  * Output: \verbinclude Cwise_isInf.out
  *
  * \sa isnan(), isfinite()
  */
inline const IsInfReturnType
isInf() const
{
  return IsInfReturnType(derived());
}

/** \returns an expression of the coefficient-wise isfinite of *this.
  *
  * Example: \include Cwise_isFinite.cpp
  * Output: \verbinclude Cwise_isFinite.out
  *
  * \sa isnan(), isinf()
  */
inline const IsFiniteReturnType
isFinite() const
{
  return IsFiniteReturnType(derived());
}

/** \returns an expression of the coefficient-wise ! operator of *this
  *
  * \warning this operator is for expression of bool only.
  *
  * Example: \include Cwise_boolean_not.cpp
  * Output: \verbinclude Cwise_boolean_not.out
  *
  * \sa operator!=()
  */
EIGEN_DEVICE_FUNC
inline const BooleanNotReturnType
operator!() const
{
  EIGEN_STATIC_ASSERT((internal::is_same<bool,Scalar>::value),
                      THIS_METHOD_IS_ONLY_FOR_EXPRESSIONS_OF_BOOL);
  return BooleanNotReturnType(derived());
}
