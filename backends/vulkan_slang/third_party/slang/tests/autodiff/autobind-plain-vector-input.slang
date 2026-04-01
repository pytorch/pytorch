//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none

[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy(float3 input, TensorView<float> output)
{
    // CUDA: __global__ void __kernel__plain_copy(_VectorStorage_float3_0 input_0, TensorView output_0)
    // TORCH: void __kernel__plain_copy(_VectorStorage_float3_0 _0, TensorView _1);

    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input.x;
    output[1] = input.y;
    output[2] = input.z;
}
