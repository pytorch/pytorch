//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[1], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// This test isn't actually testing the output, but rather that the compiler doesn't crash upon 
// encountering a specific loop pattern. ('Data' is non-differentiable here, so the expected output is 0)
// 

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

struct P
{
    bool terminated;
    bool isTerminated() { return terminated; }
    bool isHit() { return !terminated; }
};

struct Data
{
    __init(float dataIn)
    {
        this.t = dataIn;
    }
    float t;
};

void updateData(Data data)
{
    data.t = data.t * data.t;
}

[BackwardDifferentiable]
float test_simple_while(float y, uint n)
{
    Data d = Data(y);
    P p;
    p.terminated = false;
    int i = n;

    if (p.isTerminated())
        return d.t;

    [MaxIters(4)]
    while (!p.isTerminated())
    {
        updateData(d);
        p.terminated = (i-- == 0);
        if (p.isTerminated())
            break;

        if (!p.isHit())
            break;
    }
    return d.t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_while)(dpa, 2, 1.0f);
        outputBuffer[0] = dpa.d;
        //BUF: 0
    }
}
