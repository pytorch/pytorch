//TEST:SIMPLE(filecheck=CHECK): -target cuda -line-directive-mode none

//CHECK: struct s_bwd_prop_function_Intermediates{{[_0-9]+}}
//CHECK: {
//CHECK:     MyDiffPtr{{[_0-9]+}} {{[_A-Za-z0-9]+}};
//CHECK:     MyDiffPtr{{[_0-9]+}} {{[_A-Za-z0-9]+}};
//CHECK: };

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyDiffPtr
{
    uint offset;
    uint d_offset;

    [BackwardDerivative(__bwd_foo)]
    float foo() 
    { 
        return outputBuffer[offset] * outputBuffer[offset];
    }

    void __bwd_foo(float grad) 
    {
        outputBuffer[d_offset] = 2.f * outputBuffer[offset] * grad;
    }
};

[Differentiable]
float function(MyDiffPtr *i)
{
    return i[0].foo() + i[1].foo();
}

[numthreads(1, 1, 1), shader("compute")]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    MyDiffPtr s[2] = {{0, 2}, {1, 3}};
    __bwd_diff(function)(&s[0], 1.0f);
}