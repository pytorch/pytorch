
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

static const uint32_t N_LATENT_DIMS = 4;
static const uint32_t kDecoderInputCount = 6;
struct LatentTexture
{
    static const uint32_t kLatentDimsCount = N_LATENT_DIMS;
    static const uint32_t kLatentTextureCount = N_LATENT_DIMS / 4;

    [BackwardDifferentiable]
    void getCodeStochastic(float2 uv, out float code[kLatentDimsCount])
    {
        return getCode(uint2(1,2), code);
    }

    void getCode(uint2 texel, out float code[kLatentDimsCount])
    {
        for (uint i = 0; i < kLatentTextureCount; ++i)
        {
            for (uint j = 0; j < 4; ++j)
            {
                code[i * 4 + j] = j;
            }
        }
    }
    [BackwardDerivativeOf(getCode)]
    void bwd_getCode(uint2 texel, float d_out[kLatentDimsCount])
    {
        outputBuffer[0] = d_out[0];
    }
}

static LatentTexture gLatents;

[BackwardDifferentiable]
void test(float arr[10], out float result[3])
{
    float sum = 0;
    [ForceUnroll]
    for (int i = 0; i < LatentTexture.kLatentDimsCount + kDecoderInputCount; i++)
        sum += arr[i];
    result[0] = sum;
    result[1] = sum;
    result[2] = sum;
}

[BackwardDifferentiable]
float evalDecoder()
{
    // Latent code.
    float latentCode[LatentTexture.kLatentDimsCount];
    gLatents.getCodeStochastic(float2(1,2), latentCode);

    // Model input.
    float input[kDecoderInputCount + LatentTexture.kLatentDimsCount];
    input[0] = 0;
    input[1] = 1;
    input[2] = 2;
    input[3] = 3;
    input[4] = 4;
    input[5] = 5;
    [ForceUnroll]
    for (int i = 0; i < LatentTexture.kLatentDimsCount; i++)
    {
        input[kDecoderInputCount + i] = latentCode[i];
    }

    float res[3];
    test(input, res);
    return res[0] + res[1] + res[2];
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    __bwd_diff(evalDecoder)(1.0);
}
