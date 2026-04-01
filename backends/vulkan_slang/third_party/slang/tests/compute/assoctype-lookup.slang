// assoctype-nested.slang

// Confirm that an associated type can be correctly looked up.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

interface IBoneWeightSet
{
    associatedtype PackedType;
};

struct StandardBoneWeightSet : IBoneWeightSet
{
    struct PackedType
    {
        uint boneIds;
        uint boneWeights;
    };
    PackedType field;
};

interface IVertexFormat
{
    associatedtype BoneWeightSet : IBoneWeightSet;
};

struct VertexFormat<TBoneWeightSet : IBoneWeightSet> : IVertexFormat
{
    typedef TBoneWeightSet BoneWeightSet;
    TBoneWeightSet.PackedType weightSet;
};

struct OutputType
{
    VertexFormat<StandardBoneWeightSet>.BoneWeightSet.PackedType field;
};

int test(int val)
{
    OutputType rs;
    rs.field.boneIds = 256;
    return int(rs.field.boneIds);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int inputVal = int(tid);
    int outputVal = test(inputVal);
    gOutputBuffer[tid] = outputVal;
}
