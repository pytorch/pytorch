//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-dx12 -compute -output-using-type -use-dxil

// Vulkan/GLSL doesn't support 1-dimensional matrix.
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -output-using-type -emit-spirv-via-glsl
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -output-using-type -emit-spirv-directly

//TEST_INPUT: ubuffer(data=[0.0 1.0 2.0 3.0], stride=4):name=inputBuffer
RWStructuredBuffer<float> inputBuffer;

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    float v0 = inputBuffer[0];
    float v1 = inputBuffer[1];
    float v2 = inputBuffer[2];
    float v3 = inputBuffer[3];
    float result = 0.f;

    {
        float1x1 a = v0;
        float1x1 b = v1;
        b += a;
        result += b[0][0];

        float1x1 c = v2;
        float1x1 d = v3;
        d -= c;
        result += d[0][0];

        float1x1 e = v0;
        float1x1 f = v1;
        f *= e;
        result += f[0][0];

        float1x1 g = v2;
        float1x1 h = v3;
        g /= h;
        result += g[0][0];
    }
    {
        float1x2 a = float1x2(v0, v1);
        float1x2 b = float1x2(v2, v3);
        b += a;
        result += b[0][0] + b[0][1];

        float1x2 c = float1x2(v0, v1);
        float1x2 d = float1x2(v2, v3);
        d -= c;
        result += d[0][0] + d[0][1];

        float1x2 e = float1x2(v0, v1);
        float1x2 f = float1x2(v2, v3);
        f *= e;
        result += f[0][0] + f[0][1];

        float1x2 g = float1x2(v0, v1);
        float1x2 h = float1x2(v2, v3);
        g /= h;
        result += g[0][0] + g[0][1];
    }
    {
        float2x1 a = float2x1(v0, v1);
        float2x1 b = float2x1(v2, v3);
        b += a;
        result += b[0][0] + b[0][1];

        float2x1 c = float2x1(v0, v1);
        float2x1 d = float2x1(v2, v3);
        d -= c;
        result += d[0][0] + d[0][1];

        float2x1 e = float2x1(v0, v1);
        float2x1 f = float2x1(v2, v3);
        f *= e;
        result += f[0][0] + f[0][1];

        float2x1 g = float2x1(v0, v1);
        float2x1 h = float2x1(v2, v3);
        g /= h;
        result += g[0][0] + g[0][1];
    }

    //CHK:24
    outputBuffer[0] = int(result);
}
