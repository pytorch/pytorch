//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;


struct SpatialVertex : IDifferentiable
{
    float x;
};

struct MaterialVertex
{
    float x;
};

//TEST_INPUT:ubuffer(data=[2.0 2.0 2.0 2.0 2.0], stride=4):name=pathVertices
RWStructuredBuffer<MaterialVertex> pathVertices;

[Differentiable]
SpatialVertex transform(float p, MaterialVertex m)
{
    return { p * m.x };
}

[Differentiable]
float test_simple_loop(float y)
{
    SpatialVertex vShade[2];
    int pathLength = 1;

    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        if (!(pathVertices[i].x > 1.4))
        {
            pathLength = i;
            break;
        }

        vShade[i] = transform(y, pathVertices[i]);
    }

    return vShade[0].x + vShade[1].x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpy = dpfloat(1.0, 1.0);

        var dpresult = fwd_diff(test_simple_loop)(dpy);
        outputBuffer[0] = pathVertices[0].x; // CHECK: 2.0
        outputBuffer[1] = dpresult.d; // CHECK: 4.0
    }
}
