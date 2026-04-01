// compute-simple.slang

static const uint THREADGROUP_SIZE_X = 8;
static const uint THREADGROUP_SIZE_Y = THREADGROUP_SIZE_X;

struct ImageProcessingOptions
{
    float3 tintColor;
    float blurRadius;

    bool useLookupTable;
    StructuredBuffer<float4> lookupTable;
}

[shader("compute")]
[numthreads(THREADGROUP_SIZE_X, THREADGROUP_SIZE_Y)]
void processImage(
    uint3 threadID : SV_DispatchThreadID,
    uniform Texture2D inputImage,
    uniform RWTexture2D outputImage,
    uniform ImageProcessingOptions options)
{
    /* actual logic would go here */
}
