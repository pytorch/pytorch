//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main

// Verify that the call to dot is after the conditional branch.

// CHECK: OpBranchConditional
// CHECK: OpDot

[ForceInline]
float test(bool x, float3 a, float3 b) {
    float result = 0;
    if(x) {
        result = dot(a, b);
    }
    return result;
}

float caller(uniform bool x, uniform float3 a, uniform float3 b) {
    return test(x, a, b);
}

RWStructuredBuffer<float> output;

uniform bool branchCheck;
uniform float3 uniformA;
uniform float3 uniformB;

[numthreads(1,1,1)]
[shader("compute")]
void main()
{
    output[0] = caller(branchCheck, uniformA, uniformB);
}
