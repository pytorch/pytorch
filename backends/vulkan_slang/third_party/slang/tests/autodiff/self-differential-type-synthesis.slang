//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// Test that struct types made up of differentiable members who are self-differential (i.e. their Differential type is the same as their type) 
// are considered self-differential as well. We should be able to assign T.Differential = T and T = T.Differential without errors.
// 1

struct Ray : IDifferentiable {
    float a;
    float3 dir, o;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Ray ray = {};
    Ray.Differential ray2;

    ray.a = 1.f;
    ray.o = float3(3.f, 4.f, 2.5f);

    ray2 = ray;

    float t = 0.f;
    float.Differential dt = 0.f;

    t = dt;

    outputBuffer[0] = t;
    outputBuffer[1] = ray2.o.y;
    outputBuffer[2] = Ray.dadd(ray2, ray2).a;
}
