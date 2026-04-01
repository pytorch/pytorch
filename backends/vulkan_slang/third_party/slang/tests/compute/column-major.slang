// column-major.slang

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -compile-arg -O3 -shaderobj -Xslang -matrix-layout-column-major
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -shaderobj -Xslang -matrix-layout-column-major
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -dx12 -shaderobj -Xslang -matrix-layout-column-major
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -Xslang -matrix-layout-column-major
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -emit-spirv-via-glsl -Xslang -matrix-layout-column-major
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cuda -compute -shaderobj -Xslang -matrix-layout-column-major
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-mtl -compute -shaderobj -Xslang -matrix-layout-column-major

// This data is in column major layout order.... 
//TEST_INPUT:cbuffer(data=[1.0 0.0 0.0 10.0  0.0 1.0 0.0 20.0  0.0 0.0 1.0 30.0  0.0 0.0 0.0 1.0]):name matrixBuffer

ConstantBuffer<float4x4> matrixBuffer;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name output
RWStructuredBuffer<uint> output;

bool floatCheck(float data, float valueToCheckFor)
{
    return data < (valueToCheckFor + 0.001) && data > valueToCheckFor - 0.001;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    float4 v = float4(1, 2, 3, 1);

    float4x4 M1 = matrixBuffer;
    
    float4 r = mul(v, M1);

    float4x4 M2 = mul(M1, M1);

    float4x4 M3 = float4x4(
            1.0, 0.0, 0.0, 10.0, 
            0.0, 1.0, 0.0, 20.0,
            0.0, 0.0, 1.0, 30.0,
            0.0, 0.0, 0.0, 1.0
        );

    output[0] = uint(true
            && floatCheck(r.x, 11)
            && floatCheck(r.y, 22)
            && floatCheck(r.z, 33)
            && floatCheck(r.w, 1)

            && floatCheck(M1[3][0], 10)

            && floatCheck(M2[3][0], 20)
            && floatCheck(M2._41, 20)
            && floatCheck(M2._41_32[0], 20)
            && floatCheck(M2._33_42[0], 1)
            && floatCheck(M2._42_33[0], 40)

            && floatCheck(M3[0][3], 10)
        );
    //BUF: 1
}
