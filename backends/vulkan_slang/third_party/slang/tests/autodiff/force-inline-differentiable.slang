//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK): -slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// A general test for differentiable, force-inlined functions.
//
// This test was added to expose a bug in legalizeDefUse() where an IRVar can be be hoisted to 
// after its uses (when it is already in the same block, but gets reinserted), and cause 
// a validation bug.
// 

__generic<T : __BuiltinFloatingPointType, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
vector<T, R> column(matrix<T, R, C> m, int i) {
    vector<T, R> result;
    [ForceUnroll]
    for (int j = 0; j < R; j++) {
        result[j] = m[j][i];
    }
    return result;
}

struct Data {
    float4x4 m;
    float foo() {
        return length(column(m, 0).xyz);
    }
};

[numthreads(1, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    Data data;
    // Initialize matrix with some values
    data.m = float4x4(
        3.0, 0.0, 0.0, 0.0,
        0.0, 5.0, 0.0, 0.0,
        0.0, 0.0, 7.0, 0.0,
        0.0, 0.0, 0.0, 11.0);
    
    // Calculate and store result
    outputBuffer[0] = data.foo();
    
    // CHECK: type: float
    // CHECK: 3.0
}