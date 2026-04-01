// static-const-array.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -slang -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<float> outputBuffer;


static const matrix<float, 2, 3> kMatrix = { 1, 2, 4, -1, -2, -3 };
static const float3 kVector = { -1, -2, -3 };

static const matrix<float, 2, 3> kArray[2] =
{
    matrix<float, 2, 3>(0, 1, 2, 3, 4, 5),
    matrix<float, 2, 3>(float3(6, 7, 8), float3(9, 10, 11)),
};

float test(int inVal, int index)
{
    matrix<float, 2, 3> mat = kArray[index] + kMatrix;
    mat[0] =+ kVector;
    
    float2 a = { inVal, inVal + 1};
    float3 v = mul(a, mat);
    return v.x + v.y + v.z;
}

[numthreads(8, 1, 1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
    int inVal = tid.x;
    float outVal = test(inVal, inVal & 1);
    outputBuffer[inVal] = outVal;
}
