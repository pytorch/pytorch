//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// ----- MyPtrType definition -----
struct MyPtrType : IDifferentiablePtrType
{
    typealias Differential = MyPtrType;

    RWStructuredBuffer<float> buffer;
    uint offset;

    float load(uint idx) { return buffer[offset + idx]; }
    void accumulate(uint idx, float value) { buffer[offset + idx] += value; }
}

[BackwardDerivative(load_bwd)]
float load(MyPtrType b, uint idx)
{
    return b.load(idx);
}

void load_bwd(DifferentialPtrPair<MyPtrType> b, uint idx, float grad)
{
    b.d.accumulate(idx, grad);
}

// ------
[Differentiable]
float reduce(MyPtrType a)
{
    return load(a, 0) + load(a, 1);
}

[Differentiable]
float test(MyPtrType b)
{
    return reduce(b);
}

[numthreads(1, 1, 1)]
void computeMain(uint id : SV_DispatchThreadID)
{
    outputBuffer[0] = 1; // CHECK: 1
    outputBuffer[1] = 2; // CHECK: 2

    // Denote the first two elements in the buffer as the primal buffer and the last two elements for the derivative.
    var b = DifferentialPtrPair<MyPtrType>( { outputBuffer, 0 }, { outputBuffer, 2 } );

    bwd_diff(test)(b, 1.5f);

    // Check locations [2] and [3] in the buffer
    // CHECK: 1.5
    // CHECK: 1.5
}