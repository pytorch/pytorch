#pragma once

// This file only exists to provide a workaround for a Visual Studio 2017 (version < 15.8) bug.
// Apparently, VS2017<15.8 has a hard time templating on functions,
// and a workaround is to define them in an anonymous namespace near the point of instantiation:
// https://developercommunity.visualstudio.com/content/problem/25334/error-code-c2971-when-specifying-a-function-as-the.html
// This file uses macros to define wrappers for each autocasted op.
// When #included in the anonymous namespace next to the op registrations in autocast_mode.cpp,
// this file provides the local definitions that VS2017 seems to need.
// It's an ugly workaround and unnecessary except for that particular VS version.

namespace autocastVS2017Helper {

#define UPTOa(RET, OP, SIG) RET OP SIG { return at::OP(A                           ); }
#define UPTOb(RET, OP, SIG) RET OP SIG { return at::OP(A, B                        ); }
#define UPTOc(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C                     ); }
#define UPTOd(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D                  ); }
#define UPTOe(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E               ); }
#define UPTOf(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E, F            ); }
#define UPTOg(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E, F, G         ); }
#define UPTOh(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E, F, G, H      ); }
#define UPTOi(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E, F, G, H, I   ); }
#define UPTOj(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E, F, G, H, I, J); }
#define UPTOl(RET, OP, SIG) RET OP SIG { return at::OP(A, B, C, D, E, F, G, H, I, J, K, L); }

UPTOl( Tensor, _convolution               , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, bool        G, IntArrayRef H, int64_t I, bool J, bool K, bool L) )
UPTOh( Tensor, _convolution_nogroup       , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, bool        G, IntArrayRef H) )
UPTOg( Tensor, conv1d                     , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G) )
UPTOg( Tensor, conv2d                     , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G) )
UPTOg( Tensor, conv3d                     , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G) )
UPTOd( Tensor, conv_tbc                   , (const Tensor &A, const Tensor &B, const Tensor &C, int64_t       D) )
UPTOh( Tensor, conv_transpose1d           , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G, IntArrayRef H) )
UPTOh( Tensor, conv_transpose2d           , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G, IntArrayRef H) )
UPTOh( Tensor, conv_transpose3d           , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G, IntArrayRef H) )
UPTOi( Tensor, convolution                , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, bool        G, IntArrayRef H, int64_t I) )
UPTOi( Tensor, cudnn_convolution          , (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G, bool        H, bool    I) )
UPTOj( Tensor, cudnn_convolution_transpose, (const Tensor &A, const Tensor &B, const Tensor &C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, IntArrayRef G, int64_t     H, bool    I, bool J) )
UPTOh( Tensor, cudnn_convolution          , (const Tensor &A, const Tensor &B, IntArrayRef   C, IntArrayRef   D, IntArrayRef E, int64_t     F, bool        G, bool        H) )
UPTOi( Tensor, cudnn_convolution_transpose, (const Tensor &A, const Tensor &B, IntArrayRef   C, IntArrayRef   D, IntArrayRef E, IntArrayRef F, int64_t     G, bool        H, bool    I) )
UPTOb( Tensor, prelu                      , (const Tensor &A, const Tensor &B) )
UPTOe( Tensor, addmm                      , (const Tensor &A, const Tensor &B, const Tensor &C, Scalar        D, Scalar      E) )
UPTOe( Tensor, addmv                      , (const Tensor &A, const Tensor &B, const Tensor &C, Scalar        D, Scalar      E) )
UPTOe( Tensor, addr                       , (const Tensor &A, const Tensor &B, const Tensor &C, Scalar        D, Scalar      E) )
UPTOb( Tensor, matmul                     , (const Tensor &A, const Tensor &B) )
UPTOb( Tensor, mm                         , (const Tensor &A, const Tensor &B) )
UPTOb( Tensor, mv                         , (const Tensor &A, const Tensor &B) )
UPTOc( Tensor, linear                     , (const Tensor &A, const Tensor &B, const Tensor &C) )
UPTOe( Tensor, addbmm                     , (const Tensor &A, const Tensor &B, const Tensor &C, Scalar        D, Scalar      E) )
UPTOe( Tensor, baddbmm                    , (const Tensor &A, const Tensor &B, const Tensor &C, Scalar        D, Scalar      E) )
UPTOb( Tensor, bmm                        , (const Tensor &A, const Tensor &B) )
UPTOa( Tensor, chain_matmul               , (TensorList    A) )

// fp32
UPTOa( Tensor, acos                            , (const Tensor &A) )
UPTOa( Tensor, asin                            , (const Tensor &A) )
UPTOa( Tensor, cosh                            , (const Tensor &A) )
UPTOa( Tensor, erfinv                          , (const Tensor &A) )
UPTOa( Tensor, exp                             , (const Tensor &A) )
UPTOa( Tensor, expm1                           , (const Tensor &A) )
UPTOa( Tensor, log                             , (const Tensor &A) )
UPTOa( Tensor, log10                           , (const Tensor &A) )
UPTOa( Tensor, log2                            , (const Tensor &A) )
UPTOa( Tensor, log1p                           , (const Tensor &A) )
UPTOa( Tensor, reciprocal                      , (const Tensor &A) )
UPTOa( Tensor, rsqrt                           , (const Tensor &A) )
UPTOa( Tensor, sinh                            , (const Tensor &A) )
UPTOa( Tensor, tan                             , (const Tensor &A) )
UPTOb( Tensor, pow                             , (const Tensor &A, Scalar        B) )
UPTOb( Tensor, pow                             , (const Tensor &A, const Tensor &B) )
UPTOb( Tensor, pow                             , (Scalar        A, const Tensor &B) )
UPTOc( Tensor, softplus                        , (const Tensor &A, Scalar        B, Scalar        C) )
UPTOa( Tensor, gelu                            , (const Tensor &A) )
UPTOf( Tensor, layer_norm                      , (const Tensor &A, IntArrayRef   B, const Tensor &C, const Tensor &         D, double        E, bool    F) )
UPTOf( Tensor, group_norm                      , (const Tensor &A, int64_t       B, const Tensor &C, const Tensor &         D, double        E, bool    F) )
UPTOa( Tensor, frobenius_norm                  , (const Tensor &A) )
UPTOc( Tensor, frobenius_norm                  , (const Tensor &A, IntArrayRef   B, bool          C) )
UPTOb( Tensor, nuclear_norm                    , (const Tensor &A, bool          B) )
UPTOc( Tensor, nuclear_norm                    , (const Tensor &A, IntArrayRef   B, bool          C) )
UPTOd( Tensor, cosine_similarity               , (const Tensor &A, const Tensor &B, int64_t       C, double                 D) )
UPTOf( Tensor, poisson_nll_loss                , (const Tensor &A, const Tensor &B, bool          C, bool                   D, double        E, int64_t F) )
UPTOe( Tensor, cosine_embedding_loss           , (const Tensor &A, const Tensor &B, const Tensor &C, double                 D, int64_t       E) )
UPTOe( Tensor, nll_loss                        , (const Tensor &A, const Tensor &B, const Tensor &C, int64_t                D, int64_t       E) )
UPTOe( Tensor, nll_loss2d                      , (const Tensor &A, const Tensor &B, const Tensor &C, int64_t                D, int64_t       E) )
UPTOd( Tensor, hinge_embedding_loss            , (const Tensor &A, const Tensor &B, double        C, int64_t                D) )
UPTOc( Tensor, kl_div                          , (const Tensor &A, const Tensor &B, int64_t       C, bool                   D) )
UPTOc( Tensor, l1_loss                         , (const Tensor &A, const Tensor &B, int64_t       C) )
UPTOc( Tensor, smooth_l1_loss                  , (const Tensor &A, const Tensor &B, int64_t       C) )
UPTOc( Tensor, mse_loss                        , (const Tensor &A, const Tensor &B, int64_t       C) )
UPTOe( Tensor, margin_ranking_loss             , (const Tensor &A, const Tensor &B, const Tensor &C, double                 D, int64_t       E) )
UPTOc( Tensor, multilabel_margin_loss          , (const Tensor &A, const Tensor &B, int64_t       C) )
UPTOc( Tensor, soft_margin_loss                , (const Tensor &A, const Tensor &B, int64_t       C) )
UPTOh( Tensor, triplet_margin_loss             , (const Tensor &A, const Tensor &B, const Tensor &C, double                 D, double        E, double  F, bool G, int64_t H) )
UPTOf( Tensor, multi_margin_loss               , (const Tensor &A, const Tensor &B, Scalar        C, Scalar                 D, const Tensor &E, int64_t F) )
UPTOe( Tensor, binary_cross_entropy_with_logits, (const Tensor &A, const Tensor &B, const Tensor &C, const Tensor &         D, int64_t       E) )
UPTOc( Tensor, dist                            , (const Tensor &A, const Tensor &B, Scalar        C) )
UPTOb( Tensor, pdist                           , (const Tensor &A, double        B) )
UPTOd( Tensor, cdist                           , (const Tensor &A, const Tensor &B, double        C, c10::optional<int64_t> D) )
UPTOd( Tensor, renorm                          , (const Tensor &A, Scalar        B, int64_t       C, Scalar                 D) )
// The macro doesn't like this one so I had to write it manually.
std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor &A, const Tensor &B, const Tensor &C, int64_t D, int64_t E, double F) {
  return at::native_layer_norm(A, B, C, D, E, F);
}

// fp32_set_opt_dtype
UPTOb( Tensor, prod       , (const Tensor &A, c10::optional<ScalarType> B) )
UPTOd( Tensor, prod       , (const Tensor &A, int64_t                   B, bool                      C, c10::optional<ScalarType> D) )
UPTOd( Tensor, prod       , (const Tensor &A, Dimname                   B, bool                      C, c10::optional<ScalarType> D) )
UPTOc( Tensor, softmax    , (const Tensor &A, int64_t                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, softmax    , (const Tensor &A, Dimname                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, log_softmax, (const Tensor &A, int64_t                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, log_softmax, (const Tensor &A, Dimname                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, cumprod    , (const Tensor &A, int64_t                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, cumprod    , (const Tensor &A, Dimname                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, cumsum     , (const Tensor &A, int64_t                   B, c10::optional<ScalarType> C) )
UPTOc( Tensor, cumsum     , (const Tensor &A, Dimname                   B, c10::optional<ScalarType> C) )
UPTOb( Tensor, sum        , (const Tensor &A, c10::optional<ScalarType> B) )
UPTOd( Tensor, sum        , (const Tensor &A, IntArrayRef               B, bool                      C, c10::optional<ScalarType> D) )
UPTOd( Tensor, sum        , (const Tensor &A, DimnameList               B, bool                      C, c10::optional<ScalarType> D) )

// fp32_append_dtype
// The wrapper for templating wraps the REDISPATCH_FUNC, so must match REDISPATCH_SIGNATURE.
UPTOc( Tensor, norm, (const Tensor &A, c10::optional<Scalar> B, ScalarType  C) )
UPTOe( Tensor, norm, (const Tensor &A, c10::optional<Scalar> B, IntArrayRef C, bool D, ScalarType E) )
UPTOe( Tensor, norm, (const Tensor &A, c10::optional<Scalar> B, DimnameList C, bool D, ScalarType E) )

// promote
UPTOd( Tensor, addcdiv  , (const Tensor &A, const Tensor &B, const Tensor &         C, Scalar        D) )
UPTOd( Tensor, addcmul  , (const Tensor &A, const Tensor &B, const Tensor &         C, Scalar        D) )
UPTOb( Tensor, atan2    , (const Tensor &A, const Tensor &B) )
UPTOc( Tensor, cross    , (const Tensor &A, const Tensor &B, c10::optional<int64_t> C) )
UPTOd( Tensor, bilinear , (const Tensor &A, const Tensor &B, const Tensor &         C, const Tensor &D) )
UPTOd( Tensor, tensordot, (const Tensor &A, const Tensor &B, IntArrayRef            C, IntArrayRef   D) )
UPTOb( Tensor, dot      , (const Tensor &A, const Tensor &B) )
UPTOb( bool  , equal    , (const Tensor &A, const Tensor &B) )
UPTOb( Tensor, cat      , (TensorList    A, int64_t       B) )
UPTOb( Tensor, cat      , (TensorList    A, Dimname       B) )
UPTOb( Tensor, _cat     , (TensorList    A, int64_t       B) )
UPTOb( Tensor, stack    , (TensorList    A, int64_t       B) )

#undef UPTOa
#undef UPTOb
#undef UPTOc
#undef UPTOd
#undef UPTOe
#undef UPTOf
#undef UPTOg
#undef UPTOh
#undef UPTOi
#undef UPTOj
#undef UPTOl

} // namespace autocastVS2017Helper
