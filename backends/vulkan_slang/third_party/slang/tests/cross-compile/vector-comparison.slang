// vector-comparison.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -stage fragment -Wno-use-of-non-short-circuiting-operator

// This test ensures that we cross-compile vector comparison operators
// correctly to GLSL

// CHECK: OpFOrdEqual %v4bool
// CHECK: OpFOrdLessThan %v4bool
// CHECK: OpFOrdGreaterThan %v4bool
// CHECK: OpFOrdLessThanEqual %v4bool
// CHECK: OpFOrdGreaterThanEqual %v4bool
// CHECK: OpFUnordNotEqual %v4bool

struct Param
{
    float4 a, b;
};
ParameterBlock<Param> params;
float4 main() : SV_Target
{
    float4 v0 = params.a;
    float4 v1 = params.b;
    float4 result = v0 == v1 ? float4(2.0f) : float4(3.0f);
    result += v0 < v1 ? float4(2.0f) : float4(3.0f);
    result += v0 > v1 ? float4(2.0f) : float4(3.0f);
    result += v0 <= v1 ? float4(2.0f) : float4(3.0f);
    result += v0 >= v1 ? float4(2.0f) : float4(3.0f);
    result += v0 != v1 ? float4(2.0f) : float4(3.0f);
    return result;
}