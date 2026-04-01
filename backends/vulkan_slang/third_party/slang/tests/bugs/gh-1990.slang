// Stack overflow bug in type checking when using auto type deduction in a generic method.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0 -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute  -output-using-type

[anyValueSize(12)]
interface IBSDF
{
    uint getLobes();
}

[anyValueSize(16)]
interface IMaterial
{
    associatedtype BSDF : IBSDF;
    BSDF setupBSDF();
}

struct StandardBSDF : IBSDF
{
    uint getLobes()
    {
        return 0;
    }
}

struct StandardMaterial : IMaterial
{
    typedef StandardBSDF BSDF;
    StandardBSDF setupBSDF() { StandardBSDF g; return g; }
};

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: type_conformance StandardMaterial:IMaterial = 0

interface IInterface
{
    void randomStuff();
};
struct Impl : IInterface
{
    void randomStuff(){}
};

void test<R : IInterface>(R r, int id)
{
    float result = 0.0;
    // Use of auto type deduction from a generic method leads to stack overflow during
    // type checking for `bsdf.getLobes`.
    let bsdf = createDynamicObject<IMaterial, int>(id, 0).setupBSDF();
    result = bsdf.getLobes();
    gOutputBuffer[0] = float(result);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Impl impl;
    test(impl, int(dispatchThreadID.x));
}
