//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none

// Verify that we can output a cuda device function with [CudaDeviceExport].


//////////////////////////////////////////////////////////////////////////
// Lambda GGX
//////////////////////////////////////////////////////////////////////////

// CUDA-DAG: __device__ float lambdaGGX(float alphaSqr_[[#]], float cosTheta_[[#]])
[CudaDeviceExport]
[BackwardDifferentiable]
float lambdaGGX(const float alphaSqr, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    return 0.5f * (sqrt(1.0f + alphaSqr * tanThetaSqr) - 1.0f);
}

// CUDA-DAG: __device__ void lambdaGGX_bwd(DiffPair_float_[[#]] * alphaSqr_[[#]], DiffPair_float_[[#]] * cosTheta_[[#]], float d_out_[[#]])
[CudaDeviceExport]
void lambdaGGX_bwd(inout DifferentialPair<float> alphaSqr, inout DifferentialPair<float> cosTheta, const float d_out)
{
    __bwd_diff(lambdaGGX)(alphaSqr, cosTheta, d_out);
}

//////////////////////////////////////////////////////////////////////////
// Masking Smith
//////////////////////////////////////////////////////////////////////////

// CUDA-DAG: __device__ float maskingSmithGGXCorrelated(float alphaSqr_[[#]], float cosThetaI_[[#]], float cosThetaO_[[#]])
[CudaDeviceExport]
[BackwardDifferentiable]
float maskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO)
{
    float lambdaI = lambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = lambdaGGX(alphaSqr, cosThetaO);
    return 1.0f / (1.0f + lambdaI + lambdaO);
}

// CUDA-DAG: __device__ void maskingSmithGGXCorrelated_bwd(DiffPair_float_[[#]] * alphaSqr_[[#]], DiffPair_float_[[#]] * cosThetaI_[[#]], DiffPair_float_[[#]] * cosThetaO_[[#]], float d_out_[[#]])
[CudaDeviceExport]
void maskingSmithGGXCorrelated_bwd(inout DifferentialPair<float> alphaSqr,
                                   inout DifferentialPair<float> cosThetaI,
                                   inout DifferentialPair<float> cosThetaO,
                                   const float d_out)
{
    __bwd_diff(maskingSmithGGXCorrelated)(alphaSqr, cosThetaI, cosThetaO, d_out);
}
