//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK): -slang -compute -shaderobj -output-using-type

#define DO_FLOOR
#define MANUAL_DERIVATIVE

#ifndef MANUAL_DERIVATIVE
[BackwardDifferentiable]
#endif
float unusual_norm<let N : uint>(Array<float, N> x)
{
  float result = 0.f;
  #ifndef MANUAL_DERIVATIVE
  [ForceUnroll]
  #endif
  for(uint i = 0; i < N; i++)
  {
  #ifdef DO_FLOOR
    result += pow(floor(x[i]), 4);
  #else
    result += pow(x[i], 4);
  #endif
  }
  return result;
}

#ifdef MANUAL_DERIVATIVE
[BackwardDerivativeOf(unusual_norm)]
void unusual_norm_bwd<let N : uint>(inout DifferentialPair<Array<float, N>> x, float dResult)
{
  Array<float, N> derivatives;
  for(uint i = 0; i < N; i++)
  {
    derivatives[i] = 4.f * dResult * pow(x.p[i], 3);
  }
  x = diffPair(x.p, derivatives);
}
#endif

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=g_out

RWStructuredBuffer<float> g_out;
[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dtid : SV_DispatchThreadID)
{
  Array<float, 5> x;
  for(uint i = 0; i < 5; i++)
  {
    x[i] = float(i + dtid.x);
  }
  
  DifferentialPair<Array<float, 5>> x_pd = diffPair(x, {});
  bwd_diff(unusual_norm)(x_pd, 1.0f);
  for (int i = 0; i < 5; i++)
    g_out[i] = x_pd.d[i];
    // CHECK: 0.0
    // CHECK: 4.0
    // CHECK: 32.0
    // CHECK: 108.0
    // CHECK: 256.0
}
