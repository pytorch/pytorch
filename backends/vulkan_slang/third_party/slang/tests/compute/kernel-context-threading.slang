// kernel-context-threading.slang

// This test tests out the slang-ir-explicit-global-context functionality for C++ like targets. 
// In particular these require that the KernelContext is threaded through functions that access globals. 
// Currently this is only really applicable to C++, but for completeness all targets are tested.

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -compile-arg -O3 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -compile-arg -O3 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -compile-arg -O3 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -dx12 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj

//TEST_INPUT:cbuffer(data=[1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0 10.0 20.0 30.0 1.0]):name matrixBuffer
ConstantBuffer<float4x4> matrixBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0], stride=4):out,name rowOrderMatrixOutput
RWStructuredBuffer<float> rowOrderMatrixOutput;

void writeRow2(float4 v, int rowIndex)
{
    int baseIndex = rowIndex * 4;
    
    rowOrderMatrixOutput[baseIndex + 0] = v.x;
    rowOrderMatrixOutput[baseIndex + 1] = v.y;
    rowOrderMatrixOutput[baseIndex + 2] = v.z;
    rowOrderMatrixOutput[baseIndex + 3] = v.w;
}

// Just to test threading works through multiple levels of functions.
void writeRow(float4 v, int rowIndex)
{
    writeRow2(v, rowIndex);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    float4 v = float4(1, 2, 3, 1);

    float4x4 M = matrixBuffer;
    
    float4 r = mul(v, M);
    
    writeRow(M[0], 0);
    writeRow(M[1], 1);
    writeRow(M[2], 2);
    writeRow(M[3], 3);
}
