//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -target hlsl -profile cs_6_5


RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

// CHECK: function 'test_loop_with_continue' never returns
[BackwardDifferentiable]
float test_loop_with_continue(float y)
{
    float t = y;
    
    // OOPS! No `++i`! 
    for (int i = 0; i < 3;)
    {
        if (t > 4.0)
            continue;

        t = t * t;
    }

    return t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 0.0);

        __bwd_diff(test_loop_with_continue)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 32.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_loop_with_continue)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 0.0131072
    }
}