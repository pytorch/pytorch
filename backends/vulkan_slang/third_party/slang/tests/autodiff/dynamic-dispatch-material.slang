// Test calling differentiable function through dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT: set g_materials = new StructuredBuffer<MaterialDataBlob>[new MaterialDataBlob{new MaterialHeader{[0, 0, 0, 0]}, new MaterialPayload{[1.0, 1.2, 0.3, 0.5]}}];
RWStructuredBuffer<MaterialDataBlob> g_materials;

public struct ShadingInput
{
    public float scale;
}

struct MaterialHeader
{
    uint4 header;
};
struct MaterialPayload
{
    float4 data;
};
struct MaterialDataBlob
{
    MaterialHeader header;   // 16B
    MaterialPayload payload; // 16B
};

interface IMaterial : IDifferentiable
{
    associatedtype MaterialInstance : IMaterialInstance;

    [Differentiable]
    MaterialInstance setupMaterialInstance( ShadingInput input );
}

interface IMaterialInstance : IDifferentiable
{
    [Differentiable]
    float eval( float x );
}


[BackwardDerivative(getMaterial_bwd)]
IMaterial getMaterial(int id)
{
    return createDynamicObject<IMaterial, MaterialDataBlob>(id, g_materials[id]);
}

void getMaterial_bwd(int id, IDifferentiable d)
{
    // Something random
    outputBuffer[id] = 2.f;
}

struct Material1: IMaterial
{
    typedef MaterialInstance1 MaterialInstance;

    MaterialHeader header;
    float a;
    float b;
    float c;

    [Differentiable]
    MaterialInstance1 setupMaterialInstance( ShadingInput input )
    {
        MaterialInstance1 instance;
        instance.a = a * input.scale;
        instance.b = b * input.scale;
        instance.c = c * input.scale;
        return instance;
    }

}
struct MaterialInstance1: IMaterialInstance
{
    float a;
    float b;
    float c;

    [Differentiable]
    float eval( float x )
    {
        return a * x * x + b * x + c;
    }
}

struct Material2: IMaterial
{
    typedef MaterialInstance2 MaterialInstance;

    MaterialHeader header;
    float a;
    float b;

    [Differentiable]
    MaterialInstance2 setupMaterialInstance( ShadingInput input )
    {
        MaterialInstance2 instance;
        instance.a = a * input.scale * input.scale;
        instance.b = b * input.scale * input.scale;
        return instance;
    }

}
public struct MaterialInstance2: IMaterialInstance
{
    float a;
    float b;

    [Differentiable]
    public float eval( float x )
    {
        return a * x + b;
    }
}

[Differentiable]
public float shade(int material, ShadingInput input, float x)
{
    IMaterial m = getMaterial(material);
    IMaterialInstance mi = m.setupMaterialInstance(input);
    return mi.eval(x);
}

//TEST_INPUT: type_conformance Material1:IMaterial = 0
//TEST_INPUT: type_conformance Material2:IMaterial = 1

[shader("compute")]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = shade(0, {0.5}, 0.6);

    // TODO: VERIFY
    DifferentialPair<float> dpx = diffPair(3.0);
    bwd_diff(shade)(0, {0.5}, dpx, 1.0);

    outputBuffer[3] = dpx.d;
}