//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly

//CHECK: 1.0 

//TEST_INPUT: type_conformance Sensor:ISensor = 1;

[anyValueSize(16)]
interface ISensor
{
    [Differentiable]
    float4 splat(float4 point);
}

struct Sensor : ISensor
{
    [Differentiable]
    float4 splat(float4 point)
    {
        return point;
    }
}

[Differentiable]
float4 splat(ISensor* obj, float4 point)
{
    return obj->splat(point);
}

//TEST_INPUT: set s = ubuffer(data=[0 0 1 0 0 0 0 0])
uniform ISensor *s;

//TEST_INPUT: set outBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float4> outBuffer;

[shader("compute"), numthreads(1, 1, 1)]
void computeMain(
    uint3 id : SV_DispatchThreadID
)
{
    DifferentialPair<float4> dp;
    bwd_diff(splat)(s, dp, float4(1.0f));
    outBuffer[id.x] = dp.d;
}