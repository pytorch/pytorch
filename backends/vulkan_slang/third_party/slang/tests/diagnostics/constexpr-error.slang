//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// Failure to pass compile-time-constant data
// where it is expected.
//
// In this case, the place where compile-time-constant
// data is expected is the texel offset parameter to
// the `Texture2D.Sample` operation.

Texture2D t;
SamplerState s;

cbuffer U
{
	float2 uv;
	uint2 offset;
};

float4 main() : SV_Target
{
    float4 result = 0.0f;

    // Okay, immediate constant
    result += t.Sample(s, uv, int2(0,0));

    // Error: data passed through cbuffer isn't compile-time constant
    // CHECK: ([[# @LINE+1]]): error 40006:
    result += t.Sample(s, uv, offset);

    // Error: data computed via conditional isn't compile-time cosntant
    uint ii = 0;
    if(uv.x > 0.0f)
    {
    	ii = 1;
    }
    // CHECK: ([[# @LINE+1]]): error 40006:
    result += t.Sample(s, uv, int2(ii));

    // Error: data computed in loop isn't compile-time constant
    // (and loop isn't unroll-able)
    // CHECK: ([[# @LINE+1]]): error 40006:
    for(uint jj = 0; jj < uv.y; jj++)
    {
    	result += t.Sample(s, uv, int2(jj));
    }

    return result;
}