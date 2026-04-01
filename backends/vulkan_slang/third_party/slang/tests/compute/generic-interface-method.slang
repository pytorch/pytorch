//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct DisneyBRDFPattern
{
    float3 baseColor;
    float3 normal;
    float specular, metallic, roughness;
    float opacity;
    float3 emmissive;
    float ambientOcclusion;  
};

struct VertexPosition
{
    float3 pos;
    float3 normal;
    float2 uv;
};

struct CameraView
{
    float3 camPos;
    float3 camDir;
};

interface IVertexInterpolant
{
    float4 getVertexColor(int index);
    int getVertexColorCount();
    float2 getUV(int index);
    int getUVCount();
}

interface IDisneyBRDFPattern
{
    __generic<TVertexInterpolant:IVertexInterpolant>
    DisneyBRDFPattern evalPattern(
        CameraView cam, 
        VertexPosition vWorld,  
        VertexPosition vObject, 
        TVertexInterpolant interpolants);
}

struct StandardVertexInterpolant : IVertexInterpolant
{
    float4 getVertexColor(int index) { return float4(0.0); }
    int getVertexColorCount() { return 0;}
    float2 getUV(int index) { return float2(0.0); }
    int getUVCount() {return 1; }
};

struct MaterialPattern1 : IDisneyBRDFPattern
{
    __generic<TVertexInterpolant:IVertexInterpolant>
    DisneyBRDFPattern evalPattern(
        CameraView cam, 
        VertexPosition vWorld,  
        VertexPosition vObject, 
        TVertexInterpolant interpolants)
    {
        DisneyBRDFPattern rs;
        rs.baseColor = float3(0.5);
        rs.opacity = 1.0;
        return rs;
    }
};

__generic<TVertexInterpolant:IVertexInterpolant, TPattern : IDisneyBRDFPattern>
float test(TVertexInterpolant vertInterps, TPattern pattern)
{
    CameraView cam;
    VertexPosition vW, vO;
    DisneyBRDFPattern rs = pattern.evalPattern(cam, vW, vO, vertInterps);    
    return rs.baseColor.x;
}
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    StandardVertexInterpolant vertInterp;
    MaterialPattern1 mp1;
	float outVal = test<StandardVertexInterpolant, MaterialPattern1>(vertInterp, mp1);
	outputBuffer[dispatchThreadID.x] = outVal;
}