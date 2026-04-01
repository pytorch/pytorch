//TEST:SIMPLE(filecheck=CHECK): -target spirv

struct T2 {
    float4 _m0;
};

[[vk::binding(0)]]
cbuffer ubo : register(b0)
{
    int in_val;
    // CHECK: ([[# @LINE+1]]): error 31215
    T2 _z0[]; // error 31215: cannot use unsized type 'T2[]' in a constant buffer.
}

[shader("fragment")]
float4 main() : SV_TARGET
{
    uint x = in_val;
    float4 data = _z0[x]._m0;
    return data;
}