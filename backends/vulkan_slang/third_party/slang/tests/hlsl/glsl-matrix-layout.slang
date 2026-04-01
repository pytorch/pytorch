// Test that storage type lowering correctly handles matrix layout modifiers.

//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type -xslang -fvk-use-gl-layout

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 ], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct
{
    column_major float3x2 mat;
    int otherVal[4];

}

//TEST_INPUT: set gConst = cbuffer(data=[1.0 3.0 5.0 0.0 2.0 4.0 6.0 0.0 100 101 102 103])
// gConst will be:
// --             --
// |   1.0   2.0   |
// |   3.0   4.0   |
// |   5.0   6.0   |
// --             --
ConstantBuffer<MyStruct> gConst;

//TEST_INPUT: set gStructBuffer = ubuffer(data=[1.0 3.0 5.0 0.0 2.0 4.0 6.0 0.0 100 101 102 103 1.0 3.0 5.0 0.0 2.0 4.0 6.0 0.0 100 101 102 103])
RWStructuredBuffer<MyStruct[2]> gStructBuffer;

//TEST_INPUT: set gRawBuffer = ubuffer(data=[1.0 3.0 5.0 0.0 2.0 4.0 6.0 0.0 100 101 102 103])
RWByteAddressBuffer gRawBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        float2 v = float2(2.0, 3.0);
        float3 vm = mul(gConst.mat, v);
        outputBuffer[0] = vm.x; // 8
        outputBuffer[1] = vm.y; // 18
        outputBuffer[2] = vm.z; // 28
    }
    {
        float2 v = float2(2.0, 3.0);
        float3 vm = mul(gStructBuffer[0][0].mat, v);
        outputBuffer[3] = vm.x; // 8
        outputBuffer[4] = vm.y; // 18
        outputBuffer[5] = vm.z; // 28
    }
    {
        float2 v = float2(2.0, 3.0);
        float3 vm = mul(gRawBuffer.Load<MyStruct>(0).mat, v);
        outputBuffer[6] = vm.x; // 8
        outputBuffer[7] = vm.y; // 18
        outputBuffer[8] = vm.z; // 28
        gRawBuffer.Store(0, gRawBuffer.Load<MyStruct>(0));
        vm = mul(gRawBuffer.Load<MyStruct>(0).mat, v);
        outputBuffer[9] = vm.x; // 8
        outputBuffer[10] = vm.y; // 18
        outputBuffer[11] = vm.z; // 28
    }

    {
        int original;
        InterlockedAdd(gStructBuffer[0][0].otherVal[0], 1, original);
        outputBuffer[12] = gStructBuffer[0][0].otherVal[0];
    }
}
