// column-major-with-row-major-operations.slang

// Metal/CPP/CUDA do not deal with packing currently, different results will occur.
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -xslang -DTARGET_WITHOUT_PACKING
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cuda -compute -xslang -DTARGET_WITHOUT_PACKING

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -emit-spirv-via-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-mtl -compute


// CPP/CUDA due to natural layout rules will recieve the following ROW matrix:
// {1,2,3}
// {0,4,5}
// {6,0,7}

// GLSL/SPIRV/HLSL/Metal with cbuffer layout rules will recieve the following ROW/COL matrix:
// {1,2,3}
// {4,5,6}
// {7,8,9}

//TEST_INPUT:cbuffer(data=[1.0 2.0 3.0 0.0  4.0 5.0 6.0 0.0  7.0 8.0 9.0 0.0]):name matrixTestCBuf1
ConstantBuffer<row_major float3x3> matrixTestCBuf1;

// CPP/Metal/CUDA due to not having memory packing will recieve the following COL matrix post-transpose:
// {1,0,6}
// {2,4,0}
// {3,5,7}

//TEST_INPUT:cbuffer(data=[1.0 2.0 3.0 0.0  4.0 5.0 6.0 0.0  7.0 8.0 9.0 0.0]):name matrixTestCBuf2
ConstantBuffer<column_major float3x3> matrixTestCBuf2;

//TEST_INPUT:cbuffer(data=[1.0 2.0 3.0 0.0  4.0 5.0 6.0 0.0]):name NeedsPadding
cbuffer NeedsPadding
{
    float3 data1;
    float3 data2;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name output
RWStructuredBuffer<uint> output;

bool floatCheck(float data, float valueToCheckFor)
{
    return data < (valueToCheckFor + 0.001) && data > valueToCheckFor - 0.001;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    float3x3 matrixTest1;
    matrixTest1 = matrixTestCBuf1;

    float3x3 matrixTest2;
    matrixTest2 = matrixTestCBuf2;

    output[0] = uint(true
#ifdef TARGET_WITHOUT_PACKING
            && floatCheck(matrixTest1[0][0], 1)
            && floatCheck(matrixTest1[0][1], 2)
            && floatCheck(matrixTest1[0][2], 3)
            && floatCheck(matrixTest1[1][0], 0)
            && floatCheck(matrixTest1[1][1], 4)
            && floatCheck(matrixTest1[1][2], 5)
            && floatCheck(matrixTest1[2][0], 6)
            && floatCheck(matrixTest1[2][1], 0)
            && floatCheck(matrixTest1[2][2], 7)

            && floatCheck(matrixTest2[0][0], 1)
            && floatCheck(matrixTest2[0][1], 0)
            && floatCheck(matrixTest2[0][2], 6)
            && floatCheck(matrixTest2[1][0], 2)
            && floatCheck(matrixTest2[1][1], 4)
            && floatCheck(matrixTest2[1][2], 0)
            && floatCheck(matrixTest2[2][0], 3)
            && floatCheck(matrixTest2[2][1], 5)
            && floatCheck(matrixTest2[2][2], 7)

            && floatCheck(data1[0], 1)
            && floatCheck(data1[1], 2)
            && floatCheck(data1[2], 3)
            && floatCheck(data2[0], 0)
            && floatCheck(data2[1], 4)
            && floatCheck(data2[2], 5)
#else
            && floatCheck(matrixTest1[0][0], 1)
            && floatCheck(matrixTest1[0][1], 2)
            && floatCheck(matrixTest1[0][2], 3)
            && floatCheck(matrixTest1[1][0], 4)
            && floatCheck(matrixTest1[1][1], 5)
            && floatCheck(matrixTest1[1][2], 6)
            && floatCheck(matrixTest1[2][0], 7)
            && floatCheck(matrixTest1[2][1], 8)
            && floatCheck(matrixTest1[2][2], 9)

            && floatCheck(matrixTest2[0][0], 1)
            && floatCheck(matrixTest2[0][1], 4)
            && floatCheck(matrixTest2[0][2], 7)
            && floatCheck(matrixTest2[1][0], 2)
            && floatCheck(matrixTest2[1][1], 5)
            && floatCheck(matrixTest2[1][2], 8)
            && floatCheck(matrixTest2[2][0], 3)
            && floatCheck(matrixTest2[2][1], 6)
            && floatCheck(matrixTest2[2][2], 9)

            && floatCheck(data1[0], 1)
            && floatCheck(data1[1], 2)
            && floatCheck(data1[2], 3)
            && floatCheck(data2[0], 4)
            && floatCheck(data2[1], 5)
            && floatCheck(data2[2], 6)
#endif
        );
    //BUF: 1
}
