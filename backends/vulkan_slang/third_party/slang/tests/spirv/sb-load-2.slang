//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry main -stage compute

struct Test1
{
    float2x3 a;     // 24B
    float3x4 b;     // 48B
    float16_t3x2 c; // 12B
    float16_t2x4 d; // 16B
};

StructuredBuffer<Test1> dp;
RWStructuredBuffer<float4> outputBuffer;

// CHECK: outputBuffer{{.*}}._data[{{.*}}] =

[numthreads(4, 4, 1)]
void main(uint3 GTid : SV_GroupThreadID, 
          uint GI    : SV_GroupIndex)
{
    var tmp = dp[0];
    var rs = tmp.a[0][0] + tmp.a[0][1];
    outputBuffer[GI] = float4(rs);
}