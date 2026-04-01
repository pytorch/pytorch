//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:ubuffer(data=[0.0 1.0 2.0 3.0], stride=4):name=endpointBuffer
RWStructuredBuffer<float> endpointBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=endpointDifferentialBuffer
RWStructuredBuffer<float> endpointDifferentialBuffer;

struct LineSegment : IDifferentiable
{
    float x0;
    float x1;

    [BackwardDifferentiable]
    __init(float _x0, float _x1)
    {
        x0 = _x0;
        x1 = _x1;
    }
};

[BackwardDerivative(d_loadLineSegment)]
[ForwardDerivative(fwd_loadLineSegment)]
LineSegment loadLineSegment(uint id)
{
    return {endpointBuffer[id * 2], endpointBuffer[id * 2 + 1]};
}

[BackwardDerivative(d_fwd_loadLineSegment)]
DifferentialPair<LineSegment> fwd_loadLineSegment(uint id)
{
    return DifferentialPair<LineSegment>(loadLineSegment(id), LineSegment.dzero());
}

void d_loadLineSegment(uint id, LineSegment.Differential d_ls)
{
    endpointDifferentialBuffer[id * 2] += d_ls.x0;
    endpointDifferentialBuffer[id * 2 + 1] += d_ls.x1;
}

void d_fwd_loadLineSegment(uint id, DifferentialPair<LineSegment>.Differential dp_ls)
{
    endpointDifferentialBuffer[id * 2] += dp_ls.p.x0;
    endpointDifferentialBuffer[id * 2 + 1] += dp_ls.p.x1;
}

[BackwardDifferentiable]
float something()
{
    LineSegment ls = __fwd_diff(loadLineSegment)(1).p;
    return ls.x0 + ls.x1;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        LineSegment ls = __fwd_diff(loadLineSegment)(0).p;
        outputBuffer[0] = ls.x0; // Expect: 0
        outputBuffer[1] = ls.x1; // Expect: 1
    }

    {
        LineSegment.Differential d_ls = __fwd_diff(loadLineSegment)(0).d;
        outputBuffer[2] = d_ls.x1; // Expect: 0
    }

    {
        // Expect: 2.0 in endpointDifferentialBuffer[2]
        // Expect: 2.0 in endpointDifferentialBuffer[3]
        __bwd_diff(something)(2.0); 
    }
}