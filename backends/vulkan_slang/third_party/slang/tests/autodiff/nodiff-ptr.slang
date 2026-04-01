
[Differentiable]
float sumOfSquares(float x, float y, no_diff float4* test)
{
    return x * x + y * y * (test->x + test->y + test->z);
}

//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type -compile-arg -skip-spirv-validation -emit-spirv-directly

//TEST_INPUT: set ptr = ubuffer(data=[1.0 2.0 3.0], stride=4)
uniform float* ptr;

//TEST_INPUT:ubuffer(data=[0.0 0.0 0.0 0.0 0.0], stride=4):out, name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    float4* testPtr = (float4*)ptr;

    let result = sumOfSquares(2.0, 3.0, testPtr);

    // Use forward differentiation to compute the gradient of the output w.r.t. x only.
    let diffX = fwd_diff(sumOfSquares)(diffPair(2.0, 1.0), diffPair(3.0, 0.0), testPtr);

    // Create a differentiable pair to pass in the primal value and to receive the gradient.
    var dpX = diffPair(2.0);
    var dpY = diffPair(3.0);

    // Propagate the gradient of the output (1.0f) to the input parameters.
    bwd_diff(sumOfSquares)(dpX, dpY, testPtr, 1.0);

    outputBuffer[0] = result;     // 2^2 + 3^2 * (1 + 2 + 3) = 58
    outputBuffer[1] = diffX.d;    // 2*x * dx + 2*y * dy * (1 + 2 + 3) = 4
    outputBuffer[2] = diffX.p;    // 2^2 + 3^2 * (1 + 2 + 3) = 58
    outputBuffer[3] = dpX.d;      // 2*x = 4

    outputBuffer[4] = dpY.d;      // 2*y * (1 + 2 +3) = 36
}
