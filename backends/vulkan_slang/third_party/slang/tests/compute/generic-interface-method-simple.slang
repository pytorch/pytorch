//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IVertexInterpolant
{
    float2 getUV();
}

interface IBRDFPattern
{
    __generic<TVertexInterpolant:IVertexInterpolant>
    float evalPattern(TVertexInterpolant interpolants);
}

struct StandardVertexInterpolant : IVertexInterpolant
{
    float2 getUV() { return float2(0.5); }
};

struct MaterialPattern1 : IBRDFPattern
{
    float base;
    __generic<TVertexInterpolant:IVertexInterpolant>
    float evalPattern(TVertexInterpolant interpolants)
    {
        float rs = base + interpolants.getUV().x;
        return rs;
    }
};

__generic<TPattern : IBRDFPattern, TInterpolant: IVertexInterpolant>
float test(TPattern pattern, TInterpolant vertInterps)
{
    float rs = pattern.evalPattern<TInterpolant>(vertInterps);    
    return rs;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    StandardVertexInterpolant vertInterp;
    MaterialPattern1 mp1;
    mp1.base = 0.5;
	float outVal = test<MaterialPattern1, StandardVertexInterpolant>(mp1, vertInterp);
	outputBuffer[dispatchThreadID.x] = outVal;
}