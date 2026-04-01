// interface-shader-param-ordinary.slang

// This test is for interface-type shader parameters that
// get specialized to types that include "ordinary" data
// but also don't fit into the allocation provided for
// them in the existential-type field itself.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Slang-RHI/WGPU: Too small buffer is bound #5604
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

interface IModifier
{
    int modify(int value);
}

//TEST_INPUT:set gOutputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<int> gOutputBuffer;

//TEST_INPUT:set delta = 65536
uniform int delta;

//TEST_INPUT: globalSpecializationArg MyModifier
//TEST_INPUT:set gModifier = new MyModifier{ ubuffer(data=[4 3 2 1], stride=4), 3 } }
uniform IModifier gModifier;

int test(int val)
{
    return gModifier.modify(val) + delta;
}


[numthreads(4, 1, 1)]
void computeMain(
    int3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    let outputVal = test(inputVal);

    gOutputBuffer[tid] = outputVal;
}

struct MyModifier : IModifier
{
    RWStructuredBuffer<int> data;
    int extra;

    int modify(int val)
    {
        return val*65536 + data[val]*256 + val*extra;
    }
}
